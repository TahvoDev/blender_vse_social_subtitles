# Third-Party Dependencies

This add-on downloads the following libraries at runtime (not bundled in the extension):

## Runtime Python Dependencies (via uv from PyPI)
- faster-whisper 1.2.1 — MIT License
- ctranslate2 4.8.2 — MIT License
- onnxruntime 1.30.0 — MIT License
- av (PyAV) 19.0.1 — BSD 3-Clause License
- numpy 2.5.3 — BSD 3-Clause License
- torch 2.11.0 — BSD 3-Clause License
- transformers 4.57.6 — Apache License 2.0
- nvidia-cublas-cu12 12.9.2.10 — NVIDIA CUDA EULA (platform-specific GPU support)

## Other Downloads
- uv binary — Downloaded from GitHub releases (Astral/uv) to manage the virtual environment. Installed to the extension's user data directory.
- Model weights — Downloaded from Hugging Face Hub by faster-whisper and transformers (e.g. Whisper models, wav2vec2 aligner models). Model weights are subject to their respective model licenses.

All downloads occur only with user consent and respect `bpy.app.online_access` (Blender's "Allow Online Access" preference). Downloaded content is stored in the extension's user data directory and can be removed via the add-on's "Remove Downloaded Data" operator.
