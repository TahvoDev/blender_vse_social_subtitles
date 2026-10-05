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
- NVIDIA GPU + CUDA optional — falls back to CPU automatically

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
3. Click **Set up dependencies** (one-time; downloads ~700 MB into a private
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
  `requirements.in`. The venv lives outside the add-on (in the per-extension
  user data dir), so it never writes into the add-on folder and survives add-on
  updates.
- The transcription worker runs in that venv as a separate process, so heavy
  ML libraries (torch, ctranslate2, …) never load inside Blender. This avoids
  the native-library/`torch._dynamo` crashes that occur when such packages are
  imported into Blender's process.
- Audio is decoded once with faster-whisper's PyAV-based `decode_audio`, then
  word timestamps are refined by the wav2vec2 aligner (pure PyTorch CTC forced
  alignment — no `torchaudio`).

**Note:** a Blender upgrade that changes Blender's Python version invalidates
the venv. Just click **Reinstall** — it rebuilds in one step.

---

## UI Options

| Option | Description |
|---|---|
| Model | Whisper model size: `tiny` → `large-v3`. Larger = more accurate, slower. |
| Language | Source language, or "Auto-detect" |
| Device | `CUDA` (GPU) or `CPU` |
| Text Channel | VSE channel number for output TEXT strips |
| Max Chars/Line | Maximum characters per subtitle line (default 42) |
| Max Lines | Maximum lines per subtitle block (default 2) |
| Max Gap (frames) | How long a subtitle lingers after the last word; also splits blocks on longer silences |
| Text Case | Default / UPPERCASE / lowercase / Title Case |
| Remove Punctuation | Strip punctuation from output text |

---

## Troubleshooting

**"Dependencies not installed"** — click **Set up dependencies** in the N-panel.

**Setup fails / downloads blocked** — enable **Preferences → Get Extensions →
Allow Online Access**, then retry. Setup failures and full command output are
printed to Blender's system console.

**"No SOUND or MOVIE strips selected"** — make sure strips are highlighted in
the VSE before clicking Generate.

**CUDA errors / falling back to CPU** — switch Device to "CPU", or install a
matching CUDA driver + cuDNN. The default PyPI `torch` wheel is CUDA-capable on
Linux.

**Worker exits with a non-zero code** — open Blender's system console
(Window → Toggle System Console, or launch Blender from a terminal) to see the
worker traceback.

---

## Development

Run the self-checks (no dependencies required; torch tests are skipped if
absent):

```bash
python3 test_worker.py
```
