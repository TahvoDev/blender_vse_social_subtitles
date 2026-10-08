# Auto-Subtitles VSE Add-on for Blender

Transcribes selected audio/video strips and places TEXT strips in the VSE
timeline, using **faster-whisper** for transcription and a **wav2vec2 CTC
forced aligner** for accurate word-level timestamps.

One-click setup: the add-on installs its own runtime dependencies on first use,
so there is nothing to `pip install` by hand and no Administrator/root access
required.

---

## Requirements

- Blender **5.0+** (Video Sequence Editor)
- Internet access for the one-time dependency download (the add-on declares the
  `network` permission and honours **Preferences → Get Extensions → Allow
  Online Access**)
- NVIDIA GPU + driver optional — falls back to CPU automatically

No system `ffmpeg` is required (audio is decoded with the bundled PyAV).

---

## Installation

1. Zip the add-on folder so `blender_manifest.toml` and `__init__.py` are at the
   archive root:
   ```bash
   zip -r auto_subtitles.zip blender_manifest.toml __init__.py bootstrap.py \
       transcribe_worker.py align_wav2vec2.py requirements.in
   ```
2. In Blender: **Edit → Preferences → Add-ons → Install…** → select the zip.
3. Enable **"Auto-Subtitles"**.

---

## Usage

1. Open the **Video Sequence Editor**, select one or more **SOUND** or **MOVIE**
   strips.
2. Open the **N-panel** (`N`) → **Auto-Subtitles** tab.
3. Click **Set up dependencies** (one-time; downloads ~3.2 GB into a private
   venv). Progress is shown in the panel.
4. Configure options, then click **Generate Subtitles**.
5. Progress appears in the panel; TEXT strips are created when done. Model
   weights download on first use.

---

## How it works

```
Blender process                     worker process (private venv)
  __init__.py                         transcribe_worker.py
   │ json / subprocess only   ─────▶  │ faster-whisper + wav2vec2 aligner
  bootstrap.py                         │ (ctranslate2, onnxruntime, torch, PyAV)
   └─ uv: venv + pinned wheels         └─ never loads ML libs in Blender
```

- `bootstrap.py` uses [uv](https://github.com/astral-sh/uv) to create a venv
  from Blender's own Python and install the pinned dependencies in
  `requirements.in`. **Everything the add-on downloads — the venv, uv and its
  wheel cache, and the whisper/wav2vec2 models — lives in the per-extension
  user data dir** (`bpy.utils.extension_path_user`), so it never writes into
  the add-on folder and survives add-on updates.
- The transcription worker runs in that venv as a separate process, so heavy
  ML libraries (torch, ctranslate2, …) never load inside Blender. This avoids
  the native-library/`torch._dynamo` crashes that occur when such packages are
  imported into Blender's process.
- Audio is decoded with PyAV, seeking straight to the clip window (only the
  clip is held in memory) and peak-normalised for the aligner; word timestamps
  are then refined by the wav2vec2 aligner (pure PyTorch CTC forced alignment —
  no `torchaudio`).

**Note:** a Blender upgrade that changes Blender's Python version invalidates
the venv. Just click **Reinstall** — it rebuilds in one step.

---

## UI Options

| Option | Description |
|---|---|
| Model | Whisper model size: `tiny` → `large-v3`. Larger = more accurate, slower. |
| Language | Source language, or "Auto-detect" |
| Text Channel | VSE channel number for output TEXT strips |
| Max Chars/Line | Maximum characters per subtitle line (default 42) |
| Max Lines | Maximum lines per subtitle block (default 2) |
| Max Gap (frames) | How long a subtitle lingers after the last word; also splits blocks on longer silences |
| Text Case | Default / UPPERCASE / lowercase / Title Case |
| Remove Punctuation | Strip punctuation from output text |

### Device (Add-on Preferences)

The compute device is a global, set-and-forget setting, like Cycles' device
preference — **Edit → Preferences → Add-ons → Auto-Subtitles**:

- `Auto` (default) — use the GPU when available, otherwise the CPU
- `CUDA (GPU)` — force the NVIDIA GPU
- `CPU` — force the CPU

The device is only in the preferences (not the VSE side panel).

---

## Uninstall

Removing the extension (**Edit → Preferences → Add-ons → Extensions →
Remove**) deletes everything it downloaded: Blender automatically removes the
extension's user data dir, which contains the venv, uv, the wheel cache and
the speech models. Nothing is written into the add-on folder itself.

To free disk space without removing the extension, use **Remove Downloaded
Data** in **Edit → Preferences → Add-ons → Auto-Subtitles** (then re-run
**Set up dependencies** to reinstall).

Notes:

- Versions ≤ 1.2 kept uv and its wheel cache in `~/.cache/auto_subtitles`;
  the first run of 1.3+ migrates that into the user data dir (removed on
  uninstall).
- Versions ≤ 1.2 downloaded models into the shared `~/.cache/huggingface`.
  Those files are left alone on purpose (other tools may use the same
  models); delete that directory yourself if you want them gone.

---

## Troubleshooting

**"Dependencies not installed"** — click **Set up dependencies** in the N-panel.

**Setup fails / downloads blocked** — enable **Preferences → Get Extensions →
Allow Online Access**, then retry. Setup failures and full command output are
printed to Blender's system console.

**"No SOUND or MOVIE clips selected"** — make sure strips are highlighted in
the VSE before clicking Generate.

**CUDA errors / falling back to CPU** — GPU transcription needs only the
NVIDIA driver: `libcublas.so.12` ships with the add-on (setup verifies that it
loads), so no system CUDA toolkit or cuDNN install is required. The fallback
message in the N-panel shows which library failed.

- Works on Fedora/RHEL (`/usr/lib64`), Debian/Ubuntu (multiarch), Arch
  (`/opt/cuda`) and Windows, **x86_64 only**.
- **linux aarch64 is CPU-only**: the ctranslate2 wheel there has no GPU
  support, so the cuBLAS package is not installed either.
- **flatpak/snap** Blender sandboxes without `libcuda.so.1` → CPU only; use
  the native Blender build for GPU.
- After an add-on upgrade, click **Set up dependencies** again (the pinned
  requirements changed).
- To force CPU anyway, set Device to "CPU" in the add-on preferences.

**Worker exits with a non-zero code** — open Blender's system console
(Window → Toggle System Console, or launch Blender from a terminal) to see the
worker traceback.

---

## Development

Run the self-checks (no dependencies required; torch tests are skipped if
absent):

```bash
python3 test_worker.py
python3 test_paths.py
```
