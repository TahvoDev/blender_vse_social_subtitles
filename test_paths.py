"""
test_paths.py – self-check for the clean-uninstall write scope.

Verifies (without Blender) that:
  * migrate_legacy_cache() moves the pre-1.3 uv cache into the extension
    user-data dir and removes the legacy dir,
  * neither it nor purge_data() ever touches the shared HuggingFace cache,
  * every runtime path (venv, models, uv, job scratch) is under the
    user-data dir, and purge_data() removes everything the add-on downloaded.

Run: python3 test_paths.py
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Refuse to run inside Blender: data_dir() would resolve to the real
# extension data dir and purge_data() would delete the real downloads.
try:
    import bpy  # noqa: F401
    raise SystemExit("run this test outside Blender")
except ImportError:
    pass

# Exercise the XDG branches on any dev machine.
_orig_platform = sys.platform
sys.platform = "linux"


def _assert_temp_calls_are_scoped():
    """Every runtime tempfile.mkstemp/mkdtemp must be scoped with dir=<...>.

    Guards the whole point of this change: no runtime scratch may land in the
    shared system temp again.
    """
    import ast
    for name in ("bootstrap.py", "__init__.py", "transcribe_worker.py",
                 "align_wav2vec2.py"):
        tree = ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"),
                         filename=name)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("mkstemp", "mkdtemp")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tempfile"):
                assert any(k.arg == "dir" for k in node.keywords), \
                    f"{name}:{node.lineno} tempfile.{node.func.attr} without dir="


tmp = Path(tempfile.mkdtemp(prefix="as_paths_"))
os.environ["XDG_DATA_HOME"] = str(tmp / "data")
os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")

try:
    import bootstrap

    data = bootstrap.data_dir()
    legacy = bootstrap.legacy_cache_dir()
    hf_shared = tmp / "cache" / "huggingface"

    assert data == tmp / "data" / "auto_subtitles", data
    assert legacy == tmp / "cache" / "auto_subtitles", legacy

    (legacy / "bin").mkdir(parents=True)
    (legacy / "uv-cache" / "wheels").mkdir(parents=True)
    (legacy / "bin" / "uv").write_text("fake-uv")
    (legacy / "uv-cache" / "wheels" / "torch.whl").write_text("fake-whl")
    (hf_shared / "hub").mkdir(parents=True)
    (hf_shared / "hub" / "blob").write_text("shared-model")
    (data / "venv").mkdir(parents=True)
    (data / "venv" / "python").write_text("fake-python")

    bootstrap.migrate_legacy_cache()
    assert not legacy.exists(), "legacy cache dir must be removed"
    assert (data / "bin" / "uv").read_text() == "fake-uv"
    assert (data / "uv-cache" / "wheels" / "torch.whl").read_text() == "fake-whl"
    assert (data / "venv" / "python").exists(), "migration must not touch user data"
    assert (hf_shared / "hub" / "blob").exists(), "shared HF cache must be untouched"

    bootstrap.migrate_legacy_cache()

    assert bootstrap.venv_dir() == data / "venv"
    assert bootstrap.hf_dir() == data / "hf"
    assert bootstrap._uv_path() == data / "bin" / "uv"

    assert bootstrap.tmp_dir() == data / "tmp"
    assert bootstrap.tmp_dir().is_dir()
    (bootstrap.tmp_dir() / "blender_job.json").write_text("scratch")

    pinned = {"HF_HOME": "/elsewhere", "HF_HUB_CACHE": "/elsewhere/hub",
              "TRANSFORMERS_CACHE": "/elsewhere/tf",
              "PYTORCH_TRANSFORMERS_CACHE": "/elsewhere/pt",
              "PYTORCH_PRETRAINED_BERT_CACHE": "/elsewhere/bert"}
    env = bootstrap.pin_model_caches(pinned)
    assert env["HF_HOME"] == str(data / "hf")
    assert env["HF_HUB_CACHE"] == str(data / "hf" / "hub")
    for key in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                "PYTORCH_PRETRAINED_BERT_CACHE"):
        assert key not in env, f"inherited {key} must be stripped"

    scratch = bootstrap.tmp_dir() / "blender_job.json"
    assert scratch.exists(), "job scratch must still exist before purge"
    assert bootstrap.purge_data() is True
    assert not data.exists(), "user-data dir must be removed"
    assert not scratch.exists(), "job scratch must be gone"
    assert not legacy.exists(), "legacy cache dir must be removed"
    assert (hf_shared / "hub" / "blob").exists(), "shared HF cache must be untouched"

    assert bootstrap.purge_data() is True

    nv = data / "venv" / "lib" / "python3.11" / "site-packages" / "nvidia"
    cub = nv / "cublas" / "lib"
    cub.mkdir(parents=True)
    (cub / "libcublas.so.12").write_text("")
    (cub / "libcublasLt.so.12").write_text("")
    decoy = nv / "cu13" / "lib"
    decoy.mkdir(parents=True)
    (decoy / "libcublas.so.13").write_text("")
    dirs = bootstrap.find_cuda_lib_dirs()
    assert str(cub) in dirs, (cub, dirs)
    assert str(decoy) not in dirs, "torch cu13 libs must not be exposed"
    assert len(dirs) == len(set(dirs)), "duplicate candidate dirs"

    _saved_ld = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(cub), str(cub), str(tmp / "nope")])
    try:
        dirs = bootstrap.find_cuda_lib_dirs()
        assert dirs.count(str(cub)) == 1, dirs
        assert str(tmp / "nope") not in dirs, dirs
    finally:
        if _saved_ld is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = _saved_ld

    for dev in ("cuda", "auto"):
        env = bootstrap.worker_env(dev)
        assert str(cub) in env.get("LD_LIBRARY_PATH", "").split(os.pathsep), (dev, env.get("LD_LIBRARY_PATH"))
        assert "PYTHONPATH" not in env and "PYTHONHOME" not in env, dev
    env = bootstrap.worker_env("cpu")
    assert str(cub) not in (env.get("LD_LIBRARY_PATH") or ""), env.get("LD_LIBRARY_PATH")

    os.environ["LOCALAPPDATA"] = str(tmp / "localappdata")
    _saved_platform = sys.platform
    sys.platform = "win32"
    try:
        wbin = (Path(os.environ["LOCALAPPDATA"]) / "auto_subtitles" / "venv" /
                "Lib" / "site-packages" / "nvidia" / "cublas" / "bin")
        wbin.mkdir(parents=True)
        (wbin / "cublas64_12.dll").write_text("")
        wdirs = bootstrap.find_cuda_lib_dirs()
        assert str(wbin) in wdirs, (wbin, wdirs)
        wenv = bootstrap.worker_env("auto")
        parts = wenv["PATH"].split(os.pathsep)
        assert str(wbin) in parts, wenv["PATH"]
        for comp in os.environ.get("PATH", "").split(os.pathsep):
            if comp:
                assert comp in parts, (comp, wenv["PATH"])
        assert str(wbin) not in (wenv.get("LD_LIBRARY_PATH") or "")
    finally:
        sys.platform = _saved_platform
        os.environ.pop("LOCALAPPDATA", None)

    _assert_temp_calls_are_scoped()

    print("test_paths: OK")
finally:
    sys.platform = _orig_platform
    shutil.rmtree(tmp, ignore_errors=True)
