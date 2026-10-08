"""
test_paths.py – self-check for the clean-uninstall write scope.

Verifies (without Blender) that:
  * migrate_legacy_cache() moves the pre-1.3 uv cache into the extension
    user-data dir and removes the legacy dir,
  * neither it nor purge_data() ever touches the shared HuggingFace cache,
  * purge_data() removes everything the add-on downloaded.

Run: python3 test_paths.py
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Exercise the XDG branches on any dev machine.
_orig_platform = sys.platform
sys.platform = "linux"

tmp = Path(tempfile.mkdtemp(prefix="as_paths_"))
os.environ["XDG_DATA_HOME"] = str(tmp / "data")
os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")

try:
    import bootstrap

    data = bootstrap.data_dir()               # <XDG_DATA_HOME>/auto_subtitles
    legacy = bootstrap.legacy_cache_dir()     # <XDG_CACHE_HOME>/auto_subtitles
    hf_shared = tmp / "cache" / "huggingface"  # shared, must never be touched

    assert data == tmp / "data" / "auto_subtitles", data
    assert legacy == tmp / "cache" / "auto_subtitles", legacy

    # Fake pre-1.3 legacy cache + shared HF cache + current user data.
    (legacy / "bin").mkdir(parents=True)
    (legacy / "uv-cache" / "wheels").mkdir(parents=True)
    (legacy / "bin" / "uv").write_text("fake-uv")
    (legacy / "uv-cache" / "wheels" / "torch.whl").write_text("fake-whl")
    (hf_shared / "hub").mkdir(parents=True)
    (hf_shared / "hub" / "blob").write_text("shared-model")
    (data / "venv").mkdir(parents=True)
    (data / "venv" / "python").write_text("fake-python")

    # 1. Migration moves uv + wheel cache, removes legacy, keeps HF cache.
    bootstrap.migrate_legacy_cache()
    assert not legacy.exists(), "legacy cache dir must be removed"
    assert (data / "bin" / "uv").read_text() == "fake-uv"
    assert (data / "uv-cache" / "wheels" / "torch.whl").read_text() == "fake-whl"
    assert (data / "venv" / "python").exists(), "migration must not touch user data"
    assert (hf_shared / "hub" / "blob").exists(), "shared HF cache must be untouched"

    # 2. Migration is idempotent.
    bootstrap.migrate_legacy_cache()

    # 3. Paths used by the app live under the user-data dir.
    assert bootstrap.venv_dir() == data / "venv"
    assert bootstrap.hf_dir() == data / "hf"
    assert bootstrap._uv_path() == data / "bin" / "uv"

    # 4. purge_data removes everything the add-on owns, nothing else.
    bootstrap.purge_data()
    assert not data.exists(), "user-data dir must be removed"
    assert not legacy.exists(), "legacy cache dir must be removed"
    assert (hf_shared / "hub" / "blob").exists(), "shared HF cache must be untouched"

    # 5. purge is safe to call when nothing was ever downloaded.
    bootstrap.purge_data()

    print("test_paths: OK")
finally:
    sys.platform = _orig_platform
    shutil.rmtree(tmp, ignore_errors=True)
