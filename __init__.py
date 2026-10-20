"""
Auto-Subtitles VSE Add-on for Blender
Transcribes selected audio/video strips using faster-whisper and places
TEXT strips in the VSE timeline, one block per spoken phrase.

Runtime dependencies live in a private venv (see bootstrap.py) built with uv
from Blender's own Python; the transcription worker runs in that venv so no ML
library is ever loaded inside Blender's process.
"""

import bpy
import os
import json
import queue
import tempfile
import threading
import subprocess
from pathlib import Path

try:
    from . import bootstrap, presets
except ImportError:  # direct execution / legacy layout
    import bootstrap
    import presets


def _get_strips_all(scene):
    se = scene.sequence_editor
    if se is None:
        return []
    for attr in ("strips_all", "sequences_all", "strips", "sequences"):
        col = getattr(se, attr, None)
        if col is not None:
            return col
    return []


def _resolve_audio_path(strip):
    if strip.type == "SOUND":
        raw = getattr(strip.sound, "filepath", None)
        if raw:
            return str(Path(bpy.path.abspath(raw)).resolve())
    elif strip.type == "MOVIE":
        raw = getattr(strip, "filepath", None)
        if raw:
            return str(Path(bpy.path.abspath(raw)).resolve())
    return None


def _collect_selected_clips(scene):
    fps = scene.render.fps / scene.render.fps_base
    strips = _get_strips_all(scene)
    sound_paths = set()
    clips = []

    def _clip_dict(strip, path):
        fs = int(strip.frame_final_start)
        fe = int(strip.frame_final_end)
        trim_fr = int(getattr(strip, "frame_offset_start", 0))
        return {
            "name":            f"{strip.name} [ch{strip.channel} f{fs}]",
            "audio_path":      path,
            "frame_start":     fs,
            "frame_end":       fe,
            "trim_start_s":    trim_fr / fps,
            "trim_duration_s": (fe - fs) / fps,
            "fps":             fps,
        }

    for strip in strips:
        if not strip.select or strip.type != "SOUND":
            continue
        path = _resolve_audio_path(strip)
        if not path:
            continue
        sound_paths.add(path)
        clips.append(_clip_dict(strip, path))

    for strip in strips:
        if not strip.select or strip.type != "MOVIE":
            continue
        path = _resolve_audio_path(strip)
        if not path or path in sound_paths:
            continue
        clips.append(_clip_dict(strip, path))

    return clips


def _redraw_all():
    screen = getattr(bpy.context, "screen", None)
    if not screen:
        return
    for area in screen.areas:
        if area.type == "SEQUENCE_EDITOR":
            area.tag_redraw()
            for region in area.regions:
                if region.type == "UI":
                    region.tag_redraw()


class _JobState:
    __slots__ = ("running", "progress", "status", "error",
                 "result_path", "args_path", "proc", "queue")

    def __init__(self):
        self.running     = False
        self.progress    = 0.0
        self.status      = ""
        self.error       = ""
        self.result_path = ""
        self.args_path   = ""
        self.proc        = None
        self.queue       = queue.Queue()

    def cleanup(self, *paths):
        for path in (paths or (self.args_path, self.result_path)):
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass


class _SetupState:
    __slots__ = ("running", "progress", "status", "error", "queue")

    def __init__(self):
        self.running  = False
        self.progress = 0.0
        self.status   = ""
        self.error    = ""
        self.queue    = queue.Queue()


_job   = _JobState()
_setup = _SetupState()

_pending_run_args = None


# ---------------------------------------------------------------------------
# Add-on properties
# ---------------------------------------------------------------------------

MODELS = [
    ("tiny",     "Tiny",     "Fastest, lowest accuracy"),
    ("base",     "Base",     "Good balance of speed and accuracy"),
    ("small",    "Small",    ""),
    ("medium",   "Medium",   ""),
    ("large-v2", "Large-v2", ""),
    ("large-v3", "Large-v3", "Most accurate, slowest"),
]

LANGUAGES = [
    ("auto", "Auto-detect", ""),
    ("en",   "English",    ""),
    ("es",   "Spanish",    ""),
    ("fr",   "French",     ""),
    ("de",   "German",     ""),
    ("it",   "Italian",    ""),
    ("pt",   "Portuguese", ""),
    ("nl",   "Dutch",      ""),
    ("pl",   "Polish",     ""),
    ("ru",   "Russian",    ""),
    ("ja",   "Japanese",   ""),
    ("zh",   "Chinese",    ""),
    ("ko",   "Korean",     ""),
    ("ar",   "Arabic",     ""),
    ("hi",   "Hindi",      ""),
    ("tr",   "Turkish",    ""),
    ("sv",   "Swedish",    ""),
    ("da",   "Danish",     ""),
    ("fi",   "Finnish",    ""),
    ("no",   "Norwegian",  ""),
    ("uk",   "Ukrainian",  ""),
    ("cs",   "Czech",      ""),
    ("ro",   "Romanian",   ""),
    ("hu",   "Hungarian",  ""),
    ("el",   "Greek",      ""),
]

DEVICES = [
    ("auto", "Auto",       "Use the GPU when available, otherwise the CPU"),
    ("cuda", "CUDA (GPU)", "Force the NVIDIA GPU"),
    ("cpu",  "CPU",        "Force the CPU"),
]

CASES = [
    ("default", "Default",    "As transcribed"),
    ("upper",   "UPPERCASE",  ""),
    ("lower",   "lowercase",  ""),
    ("title",   "Title Case", ""),
]

_ENUM_VALUES = {
    "as_model":     {value for value, _label, _desc in MODELS},
    "as_language":  {value for value, _label, _desc in LANGUAGES},
    "as_text_case": {value for value, _label, _desc in CASES},
}

_INT_RANGES = {
    "as_channel":    (1, 128),
    "as_max_chars":  (10, 120),
    "as_max_lines":  (1, 5),
    "as_max_gap_fr": (1, 500),
}


def _presets_path():
    return presets.default_path(bootstrap.data_dir())


def _validated_preset(values):
    return presets.validate(
        values, _ENUM_VALUES, _INT_RANGES, ("as_rm_punct",)
    )


def _apply_preset(scene, values):
    return presets.apply_values(scene, _validated_preset(values))


def _register_props():
    S = bpy.types.Scene
    S.as_model     = bpy.props.EnumProperty(name="Model",    items=MODELS,    default="base")
    S.as_language  = bpy.props.EnumProperty(name="Language", items=LANGUAGES, default="auto")
    S.as_channel   = bpy.props.IntProperty(name="Text Channel", default=5, min=1, max=128)
    S.as_max_chars = bpy.props.IntProperty(name="Max Chars/Line", default=42, min=10, max=120)
    S.as_max_lines = bpy.props.IntProperty(name="Max Lines", default=2, min=1, max=5)
    S.as_max_gap_fr = bpy.props.IntProperty(
        name="Max Gap (frames)",
        description=(
            "How long a subtitle lingers after the last word before disappearing. "
            "Also splits a new block when silence exceeds this length. "
            "If the next subtitle starts sooner, it replaces this one immediately."
        ),
        default=36, min=1, max=500,
    )
    S.as_rm_punct  = bpy.props.BoolProperty(name="Remove Punctuation", default=False)
    S.as_text_case = bpy.props.EnumProperty(name="Text Case", items=CASES, default="default")


def _unregister_props():
    S = bpy.types.Scene
    for p in ("as_model", "as_language", "as_channel",
              "as_max_chars", "as_max_lines", "as_max_gap_fr",
              "as_rm_punct", "as_text_case"):
        if hasattr(S, p):
            try:
                delattr(S, p)
            except Exception:
                pass


def _timer_callback():
    global _job
    if not _job.running:
        return None

    changed = False
    while True:
        try:
            msg = _job.queue.get_nowait()
        except queue.Empty:
            break
        changed = True
        kind = msg.get("kind")
        if kind == "progress":
            _job.progress = msg["value"]
            _job.status   = msg["text"]
        elif kind == "error":
            _job.error   = msg["text"]
            _job.status  = "Error – see console"
            _job.running = False
            _job.cleanup()
        elif kind == "done":
            _job.progress = 1.0
            _job.status   = "Creating strips…"
            _create_strips_from_result(_job.result_path)
            _job.running = False
            _job.status  = "Done!"

    if changed:
        _redraw_all()

    return 0.25


def _setup_timer():
    global _setup
    if not _setup.running:
        return None

    changed = False
    while True:
        try:
            msg = _setup.queue.get_nowait()
        except queue.Empty:
            break
        changed = True
        kind = msg.get("kind")
        if kind == "progress":
            _setup.progress = msg["value"]
            _setup.status   = msg["text"]
        elif kind == "error":
            _setup.error   = msg["text"]
            _setup.status  = "Setup failed"
            _setup.running = False
            _finish_pending(False)
        elif kind == "done":
            _setup.progress = 1.0
            _setup.status   = "Ready"
            _setup.running  = False
            _finish_pending(True)

    if changed:
        _redraw_all()

    # Unregister as soon as setup ends so a later re-run can register again.
    return 0.25 if _setup.running else None


def _finish_pending(ok):
    global _pending_run_args
    pending, _pending_run_args = _pending_run_args, None
    if not pending:
        return
    args_path, res_path, device = pending
    if ok:
        _start_worker(args_path, res_path, device)
    else:
        _job.cleanup(args_path, res_path)


def _apply_case(text, case):
    if case == "upper": return text.upper()
    if case == "lower": return text.lower()
    if case == "title": return text.title()
    return text


def _create_strips_from_result(result_path):
    scene = bpy.context.scene
    if not scene.sequence_editor:
        scene.sequence_editor_create()
    se   = scene.sequence_editor
    ch   = scene.as_channel
    case = scene.as_text_case

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Auto-Subtitles] Failed to read result JSON: {e}")
        return

    subtitles = data.get("subtitles", [])
    print(f"[Auto-Subtitles] Creating {len(subtitles)} subtitle strip(s)…")

    for i, sub in enumerate(subtitles):
        text   = _apply_case(sub["text"], case)
        fs     = int(round(sub["frame_start"]))
        fe     = int(round(sub["frame_end"]))
        if fe <= fs:
            fe = fs + 1
        length = fe - fs

        try:
            ts = se.strips.new_effect(
                name=f"sub_{i:04d}", type="TEXT",
                channel=ch, frame_start=fs, length=length,
            )
        except TypeError:
            try:
                ts = se.strips.new_effect(
                    name=f"sub_{i:04d}", type="TEXT",
                    channel=ch, frame_start=fs,
                )
            except Exception as err:
                print(f"[Auto-Subtitles] Could not create strip {i}: {err}")
                continue

        if hasattr(ts, "frame_final_end"):
            ts.frame_final_end = fe

        ts.text       = text
        ts.wrap_width = 0.9
        ts.font_size  = 60
        for attr, val in (
            ("alignment_x", "CENTER"),
            ("anchor_x",    "CENTER"),
            ("anchor_y",    "BOTTOM"),
            ("location",    (0.5, 0.05)),
        ):
            if hasattr(ts, attr):
                setattr(ts, attr, val)
        ts.use_shadow = True
        ts.use_box    = True
        ts.box_margin = 0.01

    _job.cleanup()

    print("[Auto-Subtitles] Strips created successfully.")


def _start_worker(args_path, result_path, device="cpu"):
    global _job
    _job.queue       = queue.Queue()  # drop stale messages from a prior run
    _job.running     = True
    _job.progress    = 0.0
    _job.status      = "Starting…"
    _job.error       = ""
    _job.proc        = None
    _job.args_path   = args_path
    _job.result_path = result_path

    bpy.app.timers.register(_timer_callback, first_interval=0.25)
    _launch_worker(args_path, result_path, device=device)


def _launch_worker(args_path, result_path, device="cpu"):
    global _job
    vpy = bootstrap.venv_python()
    if not Path(vpy).exists():
        _job.queue.put({"kind": "error",
                        "text": "Speech engine not installed – click Generate Subtitles."})
        return
    env = bootstrap.worker_env(device)
    try:
        if not bootstrap.online_access():
            env["HF_HUB_OFFLINE"] = "1"
    except Exception:
        pass

    def _run():
        worker = str(Path(__file__).parent / "transcribe_worker.py")
        # -I = isolated mode: ignores PYTHONPATH/PYTHONHOME and user site-packages.
        cmd    = [str(vpy), "-I", worker, args_path, result_path]
        print(f"[Auto-Subtitles] Launching worker: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
        except Exception as e:
            _job.queue.put({"kind": "error", "text": str(e)})
            return

        # Assign the handle BEFORE the running check: unregister() reads
        # _job.proc on the main thread, so assigning first guarantees it either
        # kills this proc or we see running=False ourselves and kill it.
        _job.proc = proc
        if not _job.running:
            # unregistered (disable/uninstall) while we were launching
            _terminate_proc(proc)
            _job.proc = None
            _job.cleanup(args_path, result_path)
            return

        def _read_stdout():
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                print(f"[worker] {line}")
                if line.startswith("PROGRESS:"):
                    parts = line.split(":", 2)
                    try:
                        _job.queue.put({
                            "kind":  "progress",
                            "value": float(parts[1]),
                            "text":  parts[2] if len(parts) > 2 else "",
                        })
                    except Exception:
                        pass
                elif line.startswith("ERROR:"):
                    _job.queue.put({"kind": "error", "text": line[6:]})

        reader = threading.Thread(target=_read_stdout, daemon=True)
        reader.start()
        proc.wait()
        reader.join()
        _job.proc = None

        if not _job.running:
            # Cancelled – timer is gone and cancel() has returned, so clean the
            # paths this worker owned (not _job.*, which a new job may have set).
            _job.cleanup(args_path, result_path)
        elif proc.returncode != 0:
            _job.queue.put({"kind": "error",
                            "text": f"Worker exited with code {proc.returncode}"})
        else:
            _job.queue.put({"kind": "done"})

    threading.Thread(target=_run, daemon=True).start()


def _addon_key():
    return __package__ or __name__


def _get_device():
    """Current compute device from the add-on preferences (default: auto)."""
    try:
        prefs = bpy.context.preferences.addons[_addon_key()].preferences
        return prefs.device
    except Exception:
        return "auto"


class AS_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = _addon_key()

    device: bpy.props.EnumProperty(
        name="Compute Device",
        description="Device to use for transcription",
        items=DEVICES,
        default="auto",
    )

    def draw(self, context):
        layout = self.layout
        layout.row().prop(self, "device", expand=True)
        if self.device == "cuda":
            layout.label(text="Needs the NVIDIA driver – cuBLAS is installed by setup.", icon="INFO")
        layout.separator()
        layout.operator("as.purge_data", icon="TRASH")


def _begin_setup(pending_run=None):
    global _pending_run_args
    if _setup.running:
        return False
    _pending_run_args = pending_run
    _setup.queue    = queue.Queue()
    _setup.running  = True
    _setup.progress = 0.0
    _setup.status   = "Preparing…"
    _setup.error    = ""

    bpy.app.timers.register(_setup_timer, first_interval=0.25)
    _run_setup_thread()
    return True


def _run_setup_thread():
    def _work():
        try:
            bootstrap.setup(lambda v, t: _setup.queue.put(
                {"kind": "progress", "value": v, "text": t}))
            _setup.queue.put({"kind": "done"})
        except Exception as e:
            _setup.queue.put({"kind": "error", "text": str(e)})

    threading.Thread(target=_work, daemon=True).start()


class AS_OT_Transcribe(bpy.types.Operator):
    bl_idname      = "as.transcribe"
    bl_label       = "Generate Subtitles"
    bl_description = "Transcribe selected audio/video strips and create subtitle text strips"

    def execute(self, ctx):
        global _job
        if _job.running or _setup.running:
            self.report({"WARNING"}, "Already running – see the N-panel")
            return {"CANCELLED"}

        scene = ctx.scene
        clips = _collect_selected_clips(scene)
        if not clips:
            self.report({"ERROR"}, "No SOUND or MOVIE clips selected")
            return {"CANCELLED"}

        device = _get_device()
        fps = scene.render.fps / scene.render.fps_base
        args = {
            "clips":       clips,
            "model":       scene.as_model,
            "language":    None if scene.as_language == "auto" else scene.as_language,
            "device":      device,
            "max_chars":   scene.as_max_chars,
            "max_lines":   scene.as_max_lines,
            "max_gap_s":   scene.as_max_gap_fr / fps,
            "rm_punct":    scene.as_rm_punct,
            "fps":         fps,
        }

        fd_a, args_path = tempfile.mkstemp(suffix="_as_args.json",   prefix="blender_",
                                           dir=bootstrap.tmp_dir())
        fd_r, res_path  = tempfile.mkstemp(suffix="_as_result.json", prefix="blender_",
                                           dir=bootstrap.tmp_dir())
        os.close(fd_a)
        os.close(fd_r)

        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(args, f)

        if bootstrap.quick_ready():
            _start_worker(args_path, res_path, device)
            self.report({"INFO"},
                        f"Transcribing {len(clips)} clip(s) – see N-panel for progress")
            return {"FINISHED"}

        if not bootstrap.online_access():
            self.report({"ERROR"},
                        "Enable online access: Preferences > Get Extensions")
            _job.cleanup(args_path, res_path)
            return {"CANCELLED"}

        if not _begin_setup((args_path, res_path, device)):
            _job.cleanup(args_path, res_path)
            self.report({"WARNING"}, "Already running – see the N-panel")
            return {"CANCELLED"}
        self.report({"INFO"},
                    "First run: downloading the speech engine (~3.2 GB), then transcribing…")
        return {"FINISHED"}


def _terminate_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


class AS_OT_Cancel(bpy.types.Operator):
    bl_idname      = "as.cancel"
    bl_label       = "Cancel"
    bl_description = "Cancel the running setup or transcription"

    def execute(self, ctx):
        global _job, _pending_run_args
        if _setup.running:
            _setup.running = False
            _setup.status  = "Cancelled"
            pending, _pending_run_args = _pending_run_args, None
            if pending:
                _job.cleanup(pending[0], pending[1])
            bootstrap.terminate_active()
            self.report({"INFO"}, "Setup cancelled")
            return {"FINISHED"}

        if not _job.running:
            self.report({"WARNING"}, "Nothing running")
            return {"CANCELLED"}

        _job.running = False
        _job.status  = "Cancelled"

        _terminate_proc(_job.proc)
        _job.proc = None
        _job.cleanup()

        self.report({"INFO"}, "Transcription cancelled")
        return {"FINISHED"}


class AS_OT_PurgeData(bpy.types.Operator):
    bl_idname      = "as.purge_data"
    bl_label       = "Remove Downloaded Data"
    bl_description = "Delete the venv, downloaded dependencies and speech models (frees disk space; they re-download automatically on the next Generate)"
    bl_options     = {"REGISTER"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, ctx):
        if _job.running or _setup.running:
            self.report({"WARNING"}, "Setup or transcription is running – cancel it first")
            return {"CANCELLED"}
        bootstrap.terminate_active()
        ok = bootstrap.purge_data()
        if ok:
            self.report({"INFO"}, "Downloaded data removed")
        else:
            self.report({"WARNING"},
                        "Some files are still in use and could not be removed")
        return {"FINISHED"}


class AS_OT_SavePreset(bpy.types.Operator):
    bl_idname      = "as.save_preset"
    bl_label       = "Save Preset"
    bl_description = "Save the current subtitle settings as a preset"

    name: bpy.props.StringProperty(name="Preset Name")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, ctx):
        name = self.name.strip()
        if not name:
            self.report({"ERROR"}, "Preset name cannot be empty")
            return {"CANCELLED"}

        path = _presets_path()
        try:
            store = presets.load_presets(path, strict=True)
        except Exception as exc:
            self.report({"ERROR"}, f"Could not read presets: {exc}")
            return {"CANCELLED"}

        existed = name in store
        store[name] = presets.snapshot(ctx.scene)
        try:
            presets.save_presets(path, store)
        except Exception as exc:
            self.report({"ERROR"}, f"Could not save preset: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Preset '{name}' {'updated' if existed else 'saved'}")
        return {"FINISHED"}


class AS_OT_LoadPreset(bpy.types.Operator):
    bl_idname      = "as.load_preset"
    bl_label       = "Load Preset"
    bl_description = "Apply a saved subtitle settings preset"
    bl_options     = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty(options={"HIDDEN"})

    def execute(self, ctx):
        store = presets.load_presets(_presets_path())
        values = store.get(self.name)
        if values is None:
            self.report({"WARNING"}, f"Preset '{self.name}' not found")
            return {"CANCELLED"}

        applied = _apply_preset(ctx.scene, values)
        if not applied:
            self.report({"WARNING"}, f"Preset '{self.name}' has no valid settings")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Preset '{self.name}' loaded")
        return {"FINISHED"}


class AS_OT_DeletePreset(bpy.types.Operator):
    bl_idname      = "as.delete_preset"
    bl_label       = "Delete Preset"
    bl_description = "Delete a saved subtitle settings preset"

    name: bpy.props.StringProperty(options={"HIDDEN"})

    def invoke(self, context, event):
        if not self.name:
            return {"CANCELLED"}
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, ctx):
        path = _presets_path()
        try:
            store = presets.load_presets(path, strict=True)
        except Exception as exc:
            self.report({"ERROR"}, f"Could not read presets: {exc}")
            return {"CANCELLED"}

        if self.name not in store:
            self.report({"WARNING"}, f"Preset '{self.name}' not found")
            return {"CANCELLED"}

        del store[self.name]
        try:
            presets.save_presets(path, store)
        except Exception as exc:
            self.report({"ERROR"}, f"Could not delete preset: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Preset '{self.name}' deleted")
        return {"FINISHED"}


class AS_MT_PresetMenu(bpy.types.Menu):
    bl_idname = "AS_MT_preset_menu"
    bl_label  = "Presets"

    def draw(self, ctx):
        layout = self.layout
        store = presets.load_presets(_presets_path())
        if not store:
            layout.label(text="No presets saved", icon="INFO")
            return
        for name in sorted(store):
            op = layout.operator("as.load_preset", text=name)
            op.name = name
        layout.separator()
        layout.menu("AS_MT_delete_preset_menu", text="Delete Preset…")


class AS_MT_DeletePresetMenu(bpy.types.Menu):
    bl_idname = "AS_MT_delete_preset_menu"
    bl_label  = "Delete Preset"

    def draw(self, ctx):
        layout = self.layout
        store = presets.load_presets(_presets_path())
        if not store:
            layout.label(text="No presets saved", icon="INFO")
            return
        for name in sorted(store):
            op = layout.operator("as.delete_preset", text=name)
            op.name = name


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class AS_PT_Panel(bpy.types.Panel):
    bl_label       = "Auto-Subtitles"
    bl_idname      = "AS_PT_Panel"
    bl_space_type  = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category    = "Auto-Subtitles"

    def draw(self, ctx):
        layout = self.layout
        scene  = ctx.scene
        global _job, _setup

        try:
            ready = bootstrap.quick_ready()
        except Exception:
            ready = False

        if _setup.running:
            box = layout.box()
            box.label(text="Setting up speech engine…", icon="TIME")
            box.label(text=f"{int(_setup.progress * 100)}%  {_setup.status}")
            box.operator("as.cancel", icon="CANCEL")
            return

        if _setup.error:
            box = layout.box()
            box.label(text="Setup failed – click Generate to retry", icon="ERROR")
            box.label(text=_setup.error[:70])

        if _job.running:
            box = layout.box()
            box.label(text="Running…", icon="TIME")
            box.label(text=f"{int(_job.progress * 100)}%  {_job.status}")
            box.operator("as.cancel", icon="CANCEL")
            return

        if _job.error:
            layout.box().label(text=_job.error[:80], icon="ERROR")

        if _job.status == "Done!":
            layout.box().label(text="Subtitles created", icon="CHECKMARK")

        # --- Presets ----------------------------------------------------
        row = layout.row(align=True)
        row.menu("AS_MT_preset_menu", text="Presets", icon="PRESET")
        row.operator("as.save_preset", text="", icon="ADD")

        layout.separator()

        # --- Transcription ----------------------------------------------
        col = layout.column(align=True)
        col.label(text="Transcription", icon="SPEAKER")
        col.prop(scene, "as_model")
        col.prop(scene, "as_language")

        layout.separator()

        col = layout.column(align=True)
        col.label(text="Output", icon="TEXT")
        col.prop(scene, "as_channel")
        col.prop(scene, "as_max_chars")
        col.prop(scene, "as_max_lines")
        col.prop(scene, "as_max_gap_fr")

        layout.separator()

        col = layout.column(align=True)
        col.label(text="Formatting", icon="FONT_DATA")
        col.prop(scene, "as_text_case")
        col.prop(scene, "as_rm_punct")

        layout.separator()

        if not ready:
            note = layout.column(align=True)
            note.label(text="Setup downloads the speech engine (~3.2 GB) and models.", icon="INFO")
            note.label(text="The first run may take a few minutes.")

        n_clips = len(_collect_selected_clips(scene))

        row = layout.row()
        row.enabled = n_clips > 0
        row.operator("as.transcribe", icon="REC", text="Generate Subtitles")

        layout.label(text=f"{n_clips} clip(s) selected", icon="INFO")


CLASSES = [AS_AddonPreferences, AS_OT_Transcribe, AS_OT_Cancel,
           AS_OT_PurgeData, AS_OT_SavePreset, AS_OT_LoadPreset,
           AS_OT_DeletePreset, AS_MT_PresetMenu, AS_MT_DeletePresetMenu,
           AS_PT_Panel]


def register():
    # One-time move of pre-1.3 downloads into the extension's user-data dir,
    # so Blender's uninstall removes them. Never block enabling on failure.
    try:
        bootstrap.migrate_legacy_cache()
    except Exception as e:
        print(f"[Auto-Subtitles] legacy cache migration failed: {e}")
    _register_props()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    print("[Auto-Subtitles] Add-on registered.")


def unregister():
    global _job, _setup, _pending_run_args
    _job.running   = False
    _setup.running = False
    # Drop any run waiting on a first-run setup so its temp files don't leak
    # into a later enable.
    pending, _pending_run_args = _pending_run_args, None
    if pending:
        _job.cleanup(pending[0], pending[1])
    # Kill subprocesses first: Blender deletes our user-data dir right after
    # this, and locked files (Windows) would make that delete fail.
    _terminate_proc(_job.proc)
    _job.proc = None
    bootstrap.terminate_active()
    _job.cleanup()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    _unregister_props()
    print("[Auto-Subtitles] Add-on unregistered.")


if __name__ == "__main__":
    register()
