# Third-Party Dependencies

This add-on downloads the following libraries at runtime (not bundled in the extension).

## Runtime Python Dependencies (via uv from PyPI)
- faster-whisper 1.2.1 — MIT License
- ctranslate2 4.8.2 — MIT License
- onnxruntime 1.30.0 — MIT License
- av (PyAV) 19.0.1 — BSD 3-Clause License
- numpy 2.5.3 — BSD 3-Clause License
- torch 2.11.0 — BSD 3-Clause License
- transformers 4.57.6 — Apache License 2.0
- nvidia-cublas-cu12 12.9.2.10 — NVIDIA proprietary (CUDA 12 runtime for ctranslate2; see "NVIDIA / CUDA" below)

### NVIDIA / CUDA (GPU support)

GPU acceleration is optional; the add-on runs on the CPU without any NVIDIA
software. Where the pinned dependencies enable it, uv downloads two separate
NVIDIA CUDA runtime stacks from PyPI. Both are NVIDIA proprietary
(`LicenseRef-NVIDIA-Proprietary`, NVIDIA Software License Agreement / CUDA
Toolkit EULA) and are installed on the user's machine, not redistributed by
this add-on.

- **cTranslate2 / faster-whisper — CUDA 12.** `nvidia-cublas-cu12 12.9.2.10`
  supplies `libcublas.so.12` + `libcublasLt.so.12` for ctranslate2's CUDA-12
  build, and pulls `nvidia-cuda-nvrtc-cu12`. Pinned on Windows and linux
  x86_64 only (see `requirements.in`).
- **torch / wav2vec2 aligner — CUDA 13.** On Linux the PyPI `torch` wheel
  transitively installs the whole NVIDIA CUDA 13 stack:
  `cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2`,
  `cuda-bindings`, `nvidia-cudnn-cu13`, `nvidia-cusparselt-cu13`,
  `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`, and `triton`. This is pulled on
  Linux even on machines without a GPU. On Windows the CUDA runtime is bundled
  inside the `torch` wheel instead of separate packages.

Driver minimums (NVIDIA documented): CUDA 12.x runs on driver **R525+**
(≥ 525.60.13 Linux / ≥ 527.41 Windows); CUDA 13.x runs on driver **R580+**
(≥ 580.65.06 Linux).

Other transitive dependencies (e.g. `huggingface-hub`, `tokenizers`,
`flatbuffers`, `protobuf`, `sympy`, `networkx`, `jinja2`, `filelock`, `fsspec`)
are governed by their own licenses.

## Build Tool
- uv binary — Astral/uv, distributed under Apache License 2.0 OR MIT License. Downloaded from GitHub releases to manage the virtual environment; installed to the extension's user data directory.

## Model Weights
- Whisper model weights — Downloaded from Hugging Face Hub by faster-whisper. Model weights are subject to their respective model licenses.
- Wav2vec2 aligner model weights — Downloaded from Hugging Face Hub by transformers (including facebook/wav2vec2-base-960h, facebook/wav2vec2-large-960h-lv60-self (default fallback), jonatasgrosman/wav2vec2-large-xlsr-53-* variants, kresnik/wav2vec2-large-xlsr-korean, theainerd/Wav2Vec2-large-xlsr-hindi, mpoyraz/wav2vec2-xls-r-300m-cv7-turkish, Yehor/wav2vec2-xls-r-300m-uk-with-small-lm, comodoro/wav2vec2-xls-r-300m-cs-250, gigant/romanian-wav2vec2, KBLab/wav2vec2-large-voxrex-swedish, saattrupdan/wav2vec2-xls-r-300m-ftspeech, NbAiLab/nb-wav2vec2-1b-bokmaal-v2, imvladikon/wav2vec2-xls-r-300m-hebrew). These are subject to their respective model licenses.

## Adapted Source
Portions of `align_wav2vec2.py` adapt the CTC forced-alignment logic from two
BSD-2-Clause projects. Their copyright notices and license terms are reproduced
below as required by the license.

### PyTorch audio (forced-alignment tutorial)
https://github.com/pytorch/audio

```
BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### WhisperX
https://github.com/m-bain/whisperX

```
BSD 2-Clause License

Copyright (c) 2024, Max Bain

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

All downloads are user-initiated: the one-time dependency setup refuses to run
while `bpy.app.online_access` is disabled (Blender's "Allow Online Access"
preference), and model weights are fetched by the worker on first use.
Downloaded content is stored in the extension's user data directory and can be
removed via the add-on's "Remove Downloaded Data" operator.

## Compliance Notes

- **Online access:** the one-time dependency setup refuses to run while
  `bpy.app.online_access` is disabled. Model weights are downloaded on first
  use, after the user starts a transcription.
- **File access:** nothing is written to the add-on directory (which may be
  read-only for system repositories). The uv binary, venv, wheel cache and
  model weights live in the extension user-data directory created with
  `bpy.utils.extension_path_user`, and are removed on uninstall. The manifest
  declares `network` and `files` permissions accordingly.

**Not eligible for extensions.blender.org as-is.** Blender's extension
repository requires add-ons to be self-contained: third-party Python modules
must be bundled as [Python wheels](https://docs.blender.org/manual/en/dev/advanced/extensions/addons.html#bundle-dependencies)
or vendored, and add-ons "must not install Python modules, PIP packages,
Python-wheels etc." at run time. This add-on deliberately installs its runtime
into a private venv on first use because the dependency set (~3.2 GB) is too
large to bundle, so it must be distributed directly (e.g. GitHub releases)
rather than listed on the official repository.
