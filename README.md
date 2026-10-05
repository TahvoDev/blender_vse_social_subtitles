# Auto-Subtitles VSE Add-on for Blender 5.2

Transcribes selected audio/video strips and places TEXT strips in the VSE timeline.
Word timestamps come from **faster-whisper**, optionally refined with a **wav2vec2
CTC forced aligner** (`torchaudio` / HuggingFace `transformers`) when installed.

---

## Requirements

```bash
pip install faster-whisper
```

- Python 3.14, Blender 5.2
- ffmpeg installed system-wide (for audio decoding)
- NVIDIA GPU + CUDA optional (falls back to CPU)
- **Do NOT install WhisperX** — its torch/_dynamo deps crash Blender

Optional, for more accurate word timestamps (wav2vec2 forced alignment):

```bash
pip install numpy torch torchaudio transformers
```

Without these the add-on silently falls back to faster-whisper's own word timestamps.

---

## Installation

1. Zip the add-on folder so `__init__.py` sits at the top level of the archive:
   ```bash
   zip -r auto_subtitles.zip blender_vse_social_subtitles/
   ```
2. In Blender: **Edit → Preferences → Add-ons → Install…** → select the zip
3. Enable **"Auto-Subtitles"**

---

## Usage

1. Open the **Video Sequence Editor**
2. Select one or more **SOUND** or **MOVIE** strips in the timeline
3. Open the **N-panel** (press `N`) → **Auto-Subtitles** tab
4. Configure options (see below)
5. Click **Generate Subtitles**
6. Watch progress in the panel — TEXT strips appear when done

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
| Text Case | Default / UPPERCASE / lowercase / Title Case |
| Remove Punctuation | Strip punctuation from output text |
| Max Gap (frames) | How long a subtitle lingers after its last word, and the silence length that starts a new block (default 36). Gap-filling is always applied. |

---

## Architecture notes

- All transcription runs in a **subprocess** (`transcribe_worker.py`) so Blender never freezes
- A daemon reader thread streams stdout progress lines back to Blender
- A `bpy.app.timers` callback (250 ms) drains the queue and redraws the UI
- Results are passed via a temp JSON file, never through memory
- Word timing uses faster-whisper's word timestamps, optionally refined by a wav2vec2
  CTC forced aligner (`align_wav2vec2.py`); if torch/torchaudio/transformers are missing
  or alignment fails, it falls back to faster-whisper's timestamps

### Strip selection gotcha

`context.selected_sequences` is always empty when an operator fires from the
N-panel sidebar in Blender 5.2. The add-on scans `scene.sequence_editor.strips_all`
directly and checks `strip.select` on each strip.

### MOVIE + SOUND pairing

When you add an `.mp4` to the VSE, Blender creates two linked strips. The add-on
prefers the **SOUND** strip (has a `.sound.filepath`) and only falls back to the
MOVIE strip if no paired SOUND strip exists for that file.

---

## Troubleshooting

**"No SOUND or MOVIE strips selected"**
- Make sure strips are selected (highlighted) in the VSE timeline before clicking the button.

**"faster-whisper not found"**
- Run `pip install faster-whisper` and restart Blender.
- The add-on patches `sys.path` to include `~/.local/lib/python3.14/site-packages` at register time.

**Word timestamps look coarse**
- Install the optional aligner deps: `pip install numpy torch torchaudio transformers`.
- The worker log prints `(no aligner …)` or `alignment failed` when it falls back.

**CUDA errors / falling back to CPU**
- Change Device to "CPU" in the panel.
- Verify `ctranslate2` was built with CUDA support: `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"`.

**Worker exits with non-zero code**
- Open Blender's **System Console** (Window → Toggle System Console on Windows,
  or launch from terminal on Linux/macOS) to see the full traceback from the worker.
