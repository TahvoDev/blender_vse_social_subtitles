"""
Auto-Subtitles VSE Add-on for Blender 5.2
Transcribes selected audio/video strips using faster-whisper and places
TEXT strips in the VSE timeline, one block per spoken phrase.
"""

bl_info = {
    "name": "Auto-Subtitles",
    "author": "",
    "version": (1, 1, 0),
    "blender": (5, 2, 0),
    "location": "Video Sequence Editor > N-Panel > Auto-Subtitles",
    "description": "Transcribe audio/video strips and generate subtitle text strips",
    "category": "Sequencer",
}

import bpy
import os
import sys
import site
import json
import queue
import threading
import subprocess
import tempfile
import importlib
from pathlib import Path


# ---------------------------------------------------------------------------
# sys.path patch – Blender 5.2 strips user site-packages at startup
# ---------------------------------------------------------------------------

def _patch_sys_path():
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.insert(0, user_site)
    importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# VSE helpers
# ---------------------------------------------------------------------------

def _get_strips_all(scene):
    """Return the strip collection regardless of Blender API version."""
    se = scene.sequence_editor
    if se is None:
        return []
    for attr in ("strips_all", "sequences_all", "strips", "sequences"):
        col = getattr(se, attr, None)
        if col is not None:
            return col
    return []


def _resolve_audio_path(strip):
    """Return absolute audio path for SOUND or MOVIE strips, or None."""
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
    """
    Scan strips_all for selected SOUND/MOVIE strips.
    Prefer SOUND strips; only fall back to MOVIE if no paired SOUND exists.

    For each clip we capture:
      frame_start     – absolute VSE frame where the visible clip begins
      frame_end       – absolute VSE frame where the visible clip ends
      trim_start_s    – seconds into the source file where the clip starts
      trim_duration_s – duration of the visible segment in seconds
    """
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


# ---------------------------------------------------------------------------
# Global job state
# ---------------------------------------------------------------------------

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
        """Remove temp files (default: args/result). Safe to call repeatedly."""
        for path in (paths or (self.args_path, self.result_path)):
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass


_job = _JobState()


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
    ("nb",   "Norwegian",  ""),
    ("uk",   "Ukrainian",  ""),
    ("cs",   "Czech",      ""),
    ("ro",   "Romanian",   ""),
    ("hu",   "Hungarian",  ""),
    ("el",   "Greek",      ""),
]

DEVICES = [
    ("cuda", "CUDA (GPU)", ""),
    ("cpu",  "CPU",        ""),
]

CASES = [
    ("default", "Default",    "As transcribed"),
    ("upper",   "UPPERCASE",  ""),
    ("lower",   "lowercase",  ""),
    ("title",   "Title Case", ""),
]


def _register_props():
    S = bpy.types.Scene
    S.as_model     = bpy.props.EnumProperty(name="Model",    items=MODELS,    default="base")
    S.as_language  = bpy.props.EnumProperty(name="Language", items=LANGUAGES, default="auto")
    S.as_device    = bpy.props.EnumProperty(name="Device",   items=DEVICES,   default="cuda")
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
    for p in ("as_model", "as_language", "as_device", "as_channel",
              "as_max_chars", "as_max_lines", "as_max_gap_fr",
              "as_rm_punct", "as_text_case"):
        if hasattr(S, p):
            try:
                delattr(S, p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Timer – drains worker queue, updates UI
# ---------------------------------------------------------------------------

def _timer_callback():
    global _job
    if not _job.running:
        return None  # unregister timer

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
        for area in bpy.context.screen.areas:
            if area.type == "SEQUENCE_EDITOR":
                area.tag_redraw()
            for region in area.regions:
                if region.type == "UI":
                    region.tag_redraw()

    return 0.25


# ---------------------------------------------------------------------------
# Strip creation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Worker launcher
# ---------------------------------------------------------------------------

def _find_cuda_lib_dirs():
    """Find CUDA library dirs. Falls back to torch.cuda.is_available()."""
    # Primary: PyTorch's built-in detection
    try:
        import torch
        return [torch.cuda.is_available()]  # boolean list for compatibility
    except ImportError:
        pass
    # Fallback: search common CUDA paths
    import glob
    candidates = []
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if d:
            candidates.append(d)
    for pattern in (
        "/usr/local/cuda*/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/opt/cuda/lib64",
        "/usr/local/lib",
    ):
        candidates.extend(glob.glob(pattern))
    try:
        import subprocess as _sp
        out = _sp.check_output(["ldconfig", "-p"], text=True, stderr=_sp.DEVNULL)
        for line in out.splitlines():
            if "libcublas" in line and "=>" in line:
                candidates.append(str(Path(line.split("=>")[-1].strip()).parent))
    except Exception:
        pass
    try:
        import ctranslate2 as _ct2
        candidates.append(str(Path(_ct2.__file__).parent))
    except Exception:
        pass
    seen, result = set(), []
    for d in candidates:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            result.append(d)
    return result


def _build_worker_env(device):
    env = os.environ.copy()
    if device == "cuda":
        cuda_dirs = _find_cuda_lib_dirs()
        if cuda_dirs:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(cuda_dirs + ([existing] if existing else []))
            print(f"[Auto-Subtitles] LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH'][:120]}…")
    return env


def _launch_worker(args_path, result_path, device="cpu"):
    global _job
    env = _build_worker_env(device)

    def _run():
        worker = str(Path(__file__).parent / "transcribe_worker.py")
        cmd    = [sys.executable, worker, args_path, result_path]
        print(f"[Auto-Subtitles] Launching worker: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
        except Exception as e:
            _job.queue.put({"kind": "error", "text": str(e)})
            return

        _job.proc = proc

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


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class AS_OT_Transcribe(bpy.types.Operator):
    bl_idname      = "as.transcribe"
    bl_label       = "Generate Subtitles"
    bl_description = "Transcribe selected audio/video strips and create subtitle text strips"

    def execute(self, ctx):
        global _job
        if _job.running:
            self.report({"WARNING"}, "Transcription already running")
            return {"CANCELLED"}

        scene = ctx.scene
        clips = _collect_selected_clips(scene)
        if not clips:
            self.report({"ERROR"}, "No SOUND or MOVIE strips selected")
            return {"CANCELLED"}

        _patch_sys_path()

        fps = scene.render.fps / scene.render.fps_base
        args = {
            "clips":       clips,
            "model":       scene.as_model,
            "language":    None if scene.as_language == "auto" else scene.as_language,
            "device":      scene.as_device,
            "max_chars":   scene.as_max_chars,
            "max_lines":   scene.as_max_lines,
            "max_gap_s":   scene.as_max_gap_fr / fps,  # convert frames → seconds for worker
            "rm_punct":    scene.as_rm_punct,
            "fps":         fps,
        }

        fd_a, args_path = tempfile.mkstemp(suffix="_as_args.json",   prefix="blender_")
        fd_r, res_path  = tempfile.mkstemp(suffix="_as_result.json", prefix="blender_")
        os.close(fd_a)
        os.close(fd_r)

        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(args, f)

        _job.queue       = queue.Queue()  # drop stale messages from a prior run
        _job.running     = True
        _job.progress    = 0.0
        _job.status      = "Starting…"
        _job.error       = ""
        _job.proc        = None
        _job.args_path   = args_path
        _job.result_path = res_path

        bpy.app.timers.register(_timer_callback, first_interval=0.25)
        _launch_worker(args_path, res_path, device=scene.as_device)

        self.report({"INFO"}, f"Transcribing {len(clips)} clip(s) – see N-panel for progress")
        return {"FINISHED"}


class AS_OT_Cancel(bpy.types.Operator):
    bl_idname      = "as.cancel"
    bl_label       = "Cancel"
    bl_description = "Cancel the running transcription"

    def execute(self, ctx):
        global _job
        if not _job.running:
            self.report({"WARNING"}, "No transcription running")
            return {"CANCELLED"}

        _job.running = False
        _job.status  = "Cancelled"

        proc = _job.proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        _job.proc = None
        _job.cleanup()

        self.report({"INFO"}, "Transcription cancelled")
        return {"FINISHED"}


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
        global _job

        # Status box while running
        if _job.running:
            box = layout.box()
            box.label(text="⏳ Running…", icon="TIME")
            box.label(text=f"{int(_job.progress * 100)}%  {_job.status}")
            box.operator("as.cancel", icon="CANCEL")
            return

        if _job.error:
            layout.box().label(text="❌ " + _job.error[:80], icon="ERROR")

        if _job.status == "Done!":
            layout.box().label(text="✅ Subtitles created!", icon="CHECKMARK")

        # Transcription
        col = layout.column(align=True)
        col.label(text="Transcription", icon="SPEAKER")
        col.prop(scene, "as_model")
        col.prop(scene, "as_language")
        col.prop(scene, "as_device")

        layout.separator()

        # Output
        col = layout.column(align=True)
        col.label(text="Output", icon="TEXT")
        col.prop(scene, "as_channel")
        col.prop(scene, "as_max_chars")
        col.prop(scene, "as_max_lines")
        col.prop(scene, "as_max_gap_fr")

        layout.separator()

        # Formatting
        col = layout.column(align=True)
        col.label(text="Formatting", icon="FONT_DATA")
        col.prop(scene, "as_text_case")
        col.prop(scene, "as_rm_punct")

        layout.separator()

        layout.operator("as.transcribe", icon="REC", text="Generate Subtitles")

        strips = _get_strips_all(scene)
        sel = [s for s in strips if s.select and s.type in ("SOUND", "MOVIE")]
        layout.label(text=f"{len(sel)} strip(s) selected", icon="INFO")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

CLASSES = [AS_OT_Transcribe, AS_OT_Cancel, AS_PT_Panel]


def register():
    _patch_sys_path()
    _register_props()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    print("[Auto-Subtitles] Add-on registered.")


def unregister():
    global _job
    _job.running = False
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    _unregister_props()
    print("[Auto-Subtitles] Add-on unregistered.")


if __name__ == "__main__":
    register()
