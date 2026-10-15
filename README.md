# Auto-Subtitles VSE Add-on for Blender

Transcribes selected audio/video strips and places TEXT strips in the VSE
timeline, using **faster-whisper** for transcription and a **wav2vec2 CTC
forced aligner** for accurate word-level timestamps.

Everything is automatic: on the first **Generate Subtitles** the add-on
installs its own runtime dependencies, so there is no separate setup step,
nothing to `pip install` by hand and no Administrator/root access required.

---

## Requirements

- Blender **5.0+** (Video Sequence Editor)
- Internet access for the one-time dependency download (the add-on declares the
  `network` permission and honours **Preferences → Get Extensions → Allow
  Online Access**)
- NVIDIA GPU + driver optional — falls back to CPU automatically

---

## Installation

### From the extension repository (recommended — supports updates)

This repository ships a static Blender extensions index (`repo/index.json`), so
you can add it once and then install and update Auto-Subtitles from Blender.

1. **Edit → Preferences → Get Extensions → Repositories → `+` → Add Remote Repository**.
2. Set **URL** to:
   ```
   https://raw.githubusercontent.com/TahvoDev/blender_vse_social_subtitles/main/repo/index.json
   ```
3. Give it a name (e.g. `TahvoDev`), then add the repository.
4. Find **Auto-Subtitles** in the list and click **Install**.

The index is refreshed automatically on each release, so Blender offers updates here as new versions are published.

### From a release zip

1. Download the latest `auto_subtitles-<version>.zip` from the
   [**Releases**](https://github.com/TahvoDev/blender_vse_social_subtitles/releases/latest)
   page.
2. In Blender: **Edit → Preferences → Add-ons → Install…** → select the zip.
3. Enable **"Auto-Subtitles"**.

<details>
<summary>Build from source instead</summary>

Zip the add-on folder so `blender_manifest.toml` and `__init__.py` are at the
archive root, then install that zip the same way:

```bash
zip -r auto_subtitles.zip blender_manifest.toml __init__.py bootstrap.py \
    transcribe_worker.py align_wav2vec2.py requirements.in LICENSE THIRDPARTY.md
```

</details>

---

## Usage

1. Open the **Video Sequence Editor**, select one or more **SOUND** or **MOVIE**
   strips.
2. Open the **N-panel** (`N`) → **Auto-Subtitles** tab.
3. Configure options, then click **Generate Subtitles**.
4. On the first run the add-on downloads the speech engine (~3.2 GB) and the
   model weights into a private venv, showing progress in the panel, then
   transcribes automatically. This one-time step takes a few minutes; later
   runs are fast.
5. Progress appears in the panel; TEXT strips are created when done.

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
  `requirements.in`. **Everything the add-on writes — the venv, uv and its
  wheel cache, the whisper/wav2vec2 models, and the transient job scratch
  files — lives in the per-extension user data dir**
  (`bpy.utils.extension_path_user`), so it never writes into
  the add-on folder, the shared system temp, or `~/.cache`, and survives
  add-on updates.
- The transcription worker runs in that venv as a separate process, so heavy
  ML libraries (torch, ctranslate2, …) never load inside Blender. This avoids
  the native-library/`torch._dynamo` crashes that occur when such packages are
  imported into Blender's process.
- Audio is decoded with PyAV, seeking straight to the clip window (only the
  clip is held in memory) and peak-normalised for the aligner; word timestamps
  are then refined by the wav2vec2 aligner (pure PyTorch CTC forced alignment —
  no `torchaudio`).

**Note:** a Blender upgrade that changes Blender's Python version invalidates
the venv. The next **Generate Subtitles** rebuilds it automatically.

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

### NVIDIA / GPU support

GPU acceleration is optional and needs only the NVIDIA driver — no system CUDA
toolkit or cuDNN. Two CUDA stacks are involved:

- **Transcription** (ctranslate2, via faster-whisper) uses its CUDA-12 build;
  `libcublas.so.12` is provided by the pinned `nvidia-cublas-cu12` wheel
  installed into the add-on's venv (`float16` on GPU, `int8` on CPU).
- **Word alignment** (the wav2vec2 aligner) runs on torch, which is built
  against CUDA 13. On Linux that pulls the NVIDIA CUDA 13 runtime (~2–3 GB)
  from PyPI.

Driver minimums (NVIDIA's documented figures): transcription (CUDA 12)
supports NVIDIA driver **R525 or newer** (≥ 525.60.13 Linux, ≥ 527.41
Windows); GPU word alignment (torch / CUDA 13) needs **R580 or newer**
(≥ 580.65.06 Linux, R580-series on Windows).

Because the two stacks target different CUDA versions, an older driver can
still transcribe on the GPU while the aligner cannot initialise CUDA; in that
case alignment silently falls back to faster-whisper's word timestamps and
transcription stays on the GPU. Set **Device → CPU** to avoid the GPU entirely.

Note: the CUDA runtime is installed on Linux and Windows even on machines
without an NVIDIA GPU (the PyPI torch wheel is the CUDA build), so the
download is the same size either way.

---

## Uninstall

Removing the extension (**Edit → Preferences → Add-ons → Extensions →
Remove**) deletes everything it wrote: Blender automatically removes the
extension's user data dir, which contains the venv, uv, the wheel cache, the
speech models and the job scratch files. Nothing is written into the add-on
folder itself, the shared system temp, or the shared HuggingFace cache.

To free disk space without removing the extension, use **Remove Downloaded
Data** in **Edit → Preferences → Add-ons → Auto-Subtitles** (it re-downloads
automatically on the next **Generate Subtitles**).

Notes:

- Versions ≤ 1.2 kept uv and its wheel cache in `~/.cache/auto_subtitles`;
  the first run of 1.3+ migrates that into the user data dir (removed on
  uninstall).
- Versions ≤ 1.2 downloaded models into the shared `~/.cache/huggingface`.
  Those files are left alone on purpose (other tools may use the same
  models); delete that directory yourself if you want them gone.

---

## Troubleshooting

**The speech engine is missing** — just click **Generate Subtitles**; it
re-downloads automatically.

**Download fails / blocked** — enable **Preferences → Get Extensions → Allow
Online Access**, then click **Generate Subtitles** again to retry. Failures and
full command output are printed to Blender's system console.

**"No SOUND or MOVIE clips selected"** — make sure strips are highlighted in
the VSE before clicking Generate.

**CUDA errors / falling back to CPU** — GPU transcription needs only the
NVIDIA driver: `libcublas.so.12` is installed by the add-on (the pinned
`nvidia-cublas-cu12` wheel, verified at setup), so no system CUDA toolkit or
cuDNN install is required. The fallback message in the N-panel shows which
library failed.

- Works on Fedora/RHEL (`/usr/lib64`), Debian/Ubuntu (multiarch), Arch
  (`/opt/cuda`) and Windows, **x86_64 only**.
- **linux aarch64 is CPU-only**: the ctranslate2 wheel there has no GPU
  support, so the cuBLAS package is not installed either.
- **flatpak/snap** Blender sandboxes without `libcuda.so.1` → CPU only; use
  the native Blender build for GPU.
- After an add-on upgrade the next **Generate Subtitles** re-installs the
  runtime automatically (the pinned requirements changed).
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
