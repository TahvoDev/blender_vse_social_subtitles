"""
bootstrap.py – one-click dependency setup for Auto-Subtitles.

Creates a private virtual environment based on Blender's own Python and
installs the pinned runtime dependencies into it using `uv`
(https://github.com/astral-sh/uv). The transcription worker is then launched
with that venv's interpreter, so heavy ML libraries never load inside
Blender's process.

Every file the add-on writes lives under the extension's per-user data dir
(``bpy.utils.extension_path_user``): the venv, uv, the uv wheel cache, the
speech models and the transient job scratch dir. Nothing is written into the
add-on folder (Blender replaces it on update) or the shared system temp.

All network access is gated on Blender's "Allow Online Access" preference
(bpy.app.online_access).
"""

import os
import sys
import json
import glob
import hashlib
import platform
import shutil
import tarfile
import tempfile
import zipfile
import subprocess
import urllib.request
from pathlib import Path

UV_VERSION = "0.12.23"
_UV_ASSETS = {
    ("Linux", "x86_64"): (
        "uv-x86_64-unknown-linux-gnu.tar.gz",
        "9167d72b3319674b6303c4cbe071854bba13ebdf3d76b1a7cbdc175471fb66d6",
    ),
    ("Linux", "aarch64"): (
        "uv-aarch64-unknown-linux-gnu.tar.gz",
        "6524bd338177ed50d035d39354e12545e993bbeba2ecbddf0480c5b3a81d313f",
    ),
    ("Darwin", "x86_64"): (
        "uv-x86_64-apple-darwin.tar.gz",
        "960da44cb4b73685206ddd250b19e0a117fa41095710c1038f081f5cb613efb4",
    ),
    ("Darwin", "arm64"): (
        "uv-aarch64-apple-darwin.tar.gz",
        "50487ae565ccd96e499056b4674d438f4c53170202617b4c759defe0c6a1b544",
    ),
    ("Windows", "AMD64"): (
        "uv-x86_64-pc-windows-msvc.zip",
        "75d05de6762778c31ee183398de7dd15093fad0ed90b1f236d8205ea5ec00c90",
    ),
    ("Windows", "ARM64"): (
        "uv-aarch64-pc-windows-msvc.zip",
        "13294e232ececbe709c06b74e6ced06f2a225ea5591476685362f22be56a50d5",
    ),
}

ADDON_DIR   = Path(__file__).parent
REQUIREMENTS = ADDON_DIR / "requirements.in"


def clean_env() -> dict:
    """
    A sanitized environment for the venv interpreter.

    Blender and the host shell can export PYTHONPATH/PYTHONHOME (and a user
    site-packages) that leak unrelated packages into the venv, shadowing the
    pinned versions. Strip them so the venv is fully isolated.
    """
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def online_access() -> bool:
    try:
        import bpy
        return bool(getattr(bpy.app, "online_access", True))
    except Exception:
        return True


def data_dir() -> Path:
    """Persistent per-user data dir; never the (possibly read-only) add-on dir."""
    try:
        import bpy
        pkg = __package__ or "auto_subtitles"
        base = bpy.utils.extension_path_user(pkg, path="", create=True)
        if base:
            return Path(base)
    except Exception:
        pass

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    p = Path(base) / "auto_subtitles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def hf_dir() -> Path:
    return data_dir() / "hf"


def tmp_dir() -> Path:
    """
    Scratch dir for transient job files (uv scratch, args/result JSON).

    Kept inside the user-data dir so no add-on file ever lands in the shared
    system temp: uninstall and purge_data() clean it up with everything else.
    """
    p = data_dir() / "tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pin_model_caches(env: dict) -> dict:
    """
    Point every model cache at the user-data dir (mutates env, returns it).

    Sets HF_HOME/HF_HUB_CACHE for faster-whisper and transformers. The
    deprecated TRANSFORMERS_CACHE* vars are stripped instead of set: an
    inherited value would win over HF_HUB_CACHE (and make transformers warn),
    while an unset one falls back to the pinned HF_HUB_CACHE.
    """
    hub = str(hf_dir() / "hub")
    env["HF_HOME"] = str(hf_dir())
    env["HF_HUB_CACHE"] = hub
    for key in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                "PYTORCH_PRETRAINED_BERT_CACHE"):
        env.pop(key, None)
    return env


def legacy_cache_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "auto_subtitles"


def migrate_legacy_cache():
    """
    Move the pre-1.3 uv binary/wheel cache into the extension user-data dir.

    Everything persistent must live in the user-data dir so Blender's
    uninstall (which deletes that dir) leaves nothing behind. Idempotent;
    never touches ~/.cache/huggingface, which is shared with other apps.
    """
    legacy = legacy_cache_dir()
    if not legacy.exists():
        return
    for name in ("bin", "uv-cache"):
        src = legacy / name
        if not src.exists():
            continue
        dst = data_dir() / name
        try:
            if dst.exists():
                # A previous migration already produced a current copy; it
                # wins over possibly-stale legacy data.
                shutil.rmtree(src, ignore_errors=True)
            else:
                shutil.move(str(src), str(dst))
        except Exception as e:
            print(f"[bootstrap] could not migrate {src}: {e}")
            shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(legacy, ignore_errors=True)


def purge_data() -> bool:
    target = data_dir()
    legacy = legacy_cache_dir()
    shutil.rmtree(target, ignore_errors=True)
    shutil.rmtree(legacy, ignore_errors=True)
    return not target.exists() and not legacy.exists()


def venv_dir() -> Path:
    return data_dir() / "venv"


def venv_python() -> Path:
    if sys.platform == "win32":
        return venv_dir() / "Scripts" / "python.exe"
    return venv_dir() / "bin" / "python"


def find_cuda_lib_dirs() -> list:
    """
    Directories that may contain the CUDA libraries ctranslate2 dlopens.

    System candidates first (a toolkit install wins over the pip copy), then
    the add-on venv's own nvidia/cublas tree — the only place pip-provided
    libcublas.so.12 lives, and the only place visible to flatpak/snap Blender
    where host /usr is not.
    """
    candidates = []
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
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
        out = subprocess.check_output(["ldconfig", "-p"], text=True,
                                      stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "libcublas" in line and "=>" in line:
                candidates.append(str(Path(line.split("=>")[-1].strip()).parent))
    except Exception:
        pass
    # ct2 needs only libcublas.so.12 + libcublasLt.so.12, both in
    # nvidia/cublas/lib. Deliberately not nvidia/*/lib: that would also put
    # torch's CUDA 13 libs on the loader path for no benefit.
    if sys.platform == "win32":
        candidates.extend(glob.glob(
            str(venv_dir() / "Lib/site-packages/nvidia/cublas/bin")))
    else:
        candidates.extend(glob.glob(
            str(venv_dir() / "lib/python*/site-packages/nvidia/cublas/lib")))
    seen, result = set(), []
    for d in candidates:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            result.append(d)
    return result


def worker_env(device: str = "auto") -> dict:
    """
    Environment for the transcription worker.

    Anything but "cpu" gets the CUDA lib dirs on the loader path: the
    preference default is "auto" and the worker resolves auto → cuda itself,
    so gating on "cuda" here would leave the default config un-augmented.
    """
    env = clean_env()
    pin_model_caches(env)
    if device == "cpu":
        return env
    key = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    cuda_dirs = find_cuda_lib_dirs()
    existing = env.get(key, "")
    merged = cuda_dirs + [d for d in existing.split(os.pathsep)
                          if d and d not in cuda_dirs]
    if merged:
        env[key] = os.pathsep.join(merged)
        print(f"[Auto-Subtitles] {key}: {env[key][:120]}…")
    return env


def _sentinel_path() -> Path:
    return venv_dir() / ".auto_subtitles.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint() -> dict:
    return {
        "blender_exe": sys.executable,
        "blender_py": f"{sys.version_info.major}.{sys.version_info.minor}",
        "requirements_sha": _sha256_file(REQUIREMENTS) if REQUIREMENTS.exists() else "",
    }


def quick_ready() -> bool:
    if not venv_python().exists():
        return False
    try:
        return json.loads(_sentinel_path().read_text()) == _fingerprint()
    except Exception:
        return False


def _probe(timeout: int = 300):
    """
    Import the key packages in the venv and return (ok, detail).

    detail is the captured stdout/stderr (with tracebacks), plus the list of
    modules that failed, so a setup failure is diagnosable from the UI/console.

    Also *loads* libcublas when the venv ships one (see worker_env): a
    present-but-unloadable library would otherwise only surface as a silent
    CPU fallback at the first transcription. That condition prefixes detail
    with "WARNING:" but does not make the probe fail — CPU still works.
    """
    code = (
        "import importlib, traceback\n"
        "mods = ['faster_whisper', 'torch', 'transformers']\n"
        "bad = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        mod = importlib.import_module(m)\n"
        "        print('OK', m, getattr(mod, '__version__', ''))\n"
        "    except Exception:\n"
        "        bad.append(m)\n"
        "        traceback.print_exc()\n"
        "print('PROBE_BAD:' + ','.join(bad))\n"
        "import ctypes, glob, os, sys\n"
        "if sys.platform == 'win32':\n"
        "    cub = glob.glob(os.path.join(sys.prefix, 'Lib', 'site-packages',"
        " 'nvidia', 'cublas', 'bin'))\n"
        "    lib = 'cublas64_12.dll'\n"
        "else:\n"
        "    cub = glob.glob(os.path.join(sys.prefix, 'lib', 'python*',"
        " 'site-packages', 'nvidia', 'cublas', 'lib'))\n"
        "    lib = 'libcublas.so.12'\n"
        "if cub:\n"
        "    try:\n"
        "        ctypes.CDLL(lib)\n"
        "        print('OK', lib, 'from', cub[0])\n"
        "    except OSError as e:\n"
        "        print('CUBLAS_FAIL:' + str(e))\n"
    )
    global _active_proc
    try:
        proc = subprocess.Popen(
            [str(venv_python()), "-I", "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=worker_env("auto"),
        )
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    _active_proc = proc
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        return False, f"probe timed out after {timeout}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        if _active_proc is proc:
            _active_proc = None

    out = ((stdout or "") + (stderr or "")).strip()
    bad = ""
    cublas_err = ""
    for line in out.splitlines():
        if line.startswith("PROBE_BAD:"):
            bad = line[len("PROBE_BAD:"):]
        elif line.startswith("CUBLAS_FAIL:"):
            cublas_err = line[len("CUBLAS_FAIL:"):]
    ok = (proc.returncode == 0) and (bad == "")
    if bad:
        out = f"failed to import: {bad}\n\n{out}"
    if cublas_err:
        out = f"WARNING: libcublas present but failed to load: {cublas_err}\n\n{out}"
        print(f"[bootstrap] cublas load check failed: {cublas_err}")
    return ok, out


def probe_ready(timeout: int = 300) -> bool:
    if not quick_ready():
        return False
    ok, _ = _probe(timeout=timeout)
    return ok


def _uv_path() -> Path:
    return data_dir() / "bin" / ("uv.exe" if sys.platform == "win32" else "uv")


def _download(url: str, dest: Path, progress=None, lo: float = 0.0, hi: float = 1.0):
    req = urllib.request.Request(url, headers={"User-Agent": "auto-subtitles"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(lo + (hi - lo) * done / total,
                             f"Downloading… {done * 100 // total}%")


def ensure_uv(progress=None) -> str:
    found = shutil.which("uv")
    if found:
        return found

    target = _uv_path()
    if target.exists():
        try:
            if _run_tracked([str(target), "--version"]) != 0:
                raise RuntimeError("uv self-check failed")
            return str(target)
        except Exception:
            pass

    key = (platform.system(), platform.machine())
    asset = _UV_ASSETS.get(key)
    if asset is None:
        raise RuntimeError(
            f"No bundled uv build for {key[0]}/{key[1]}. "
            f"Install uv manually (https://docs.astral.sh/uv/), then retry."
        )
    if not online_access():
        raise RuntimeError(
            "Blender online access is disabled. Enable it in "
            "Preferences → Get Extensions → Allow Online Access, then retry."
        )

    fname, sha = asset
    url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{fname}"
    if progress:
        progress(0.03, f"Downloading uv {UV_VERSION}…")

    tmpdir = Path(tempfile.mkdtemp(prefix="as_uv_", dir=tmp_dir()))
    try:
        tarpath = tmpdir / fname
        _download(url, tarpath, progress, 0.03, 0.20)
        digest = _sha256_file(tarpath)
        if digest != sha:
            raise RuntimeError(
                f"uv checksum mismatch for {fname}:\n"
                f"  expected {sha}\n"
                f"  got      {digest}"
            )

        if progress:
            progress(0.21, "Installing uv…")
        target.parent.mkdir(parents=True, exist_ok=True)
        want = "uv.exe" if sys.platform == "win32" else "uv"
        if fname.endswith(".zip"):
            with zipfile.ZipFile(tarpath) as zf:
                name = next((n for n in zf.namelist()
                             if os.path.basename(n) == want), None)
                if name is None:
                    raise RuntimeError("uv binary not found in downloaded archive")
                with zf.open(name) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        else:
            with tarfile.open(tarpath, "r:gz") as tf:
                member = next((m for m in tf.getmembers()
                               if m.isfile() and os.path.basename(m.name) == want), None)
                if member is None:
                    raise RuntimeError("uv binary not found in downloaded archive")
                src = tf.extractfile(member)
                with open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        try:
            target.chmod(0o755)
        except Exception:
            pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return str(target)


_active_proc = None


def _run_tracked(cmd, timeout=30):
    global _active_proc
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _active_proc = proc
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    finally:
        if _active_proc is proc:
            _active_proc = None


def terminate_active():
    proc = _active_proc
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def _run(cmd, env, on_line=None):
    global _active_proc
    print(f"[bootstrap] $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    _active_proc = proc
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"[bootstrap] {line}")
                if on_line:
                    on_line(line)
        proc.wait()
    finally:
        if _active_proc is proc:
            _active_proc = None
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}: {' '.join(cmd)}")


def setup(progress=None):
    def p(v, t):
        if progress:
            progress(v, t)

    if not online_access():
        raise RuntimeError(
            "Blender online access is disabled. Enable it in "
            "Preferences → Get Extensions → Allow Online Access, then retry."
        )
    if not REQUIREMENTS.exists():
        raise RuntimeError(f"requirements.in not found at {REQUIREMENTS}")

    # One-time move of the pre-1.3 uv binary/cache from ~/.cache into the
    # extension's user-data dir (Blender wipes that dir on uninstall).
    migrate_legacy_cache()

    uv = ensure_uv(p)

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(data_dir() / "uv-cache")
    env["UV_NO_PROGRESS"] = "1"
    env["UV_PYTHON_PREFERENCE"] = "only-system"

    vd = venv_dir()
    if vd.exists():
        shutil.rmtree(vd, ignore_errors=True)

    p(0.25, "Creating virtual environment…")
    _run([uv, "venv",
          "--python", sys.executable,
          "--python-preference", "only-system",
          str(vd)], env)
    if not venv_python().exists():
        raise RuntimeError(f"virtual environment not created at {vd}")

    # Belt-and-suspenders: make sure the venv never inherits the base
    # interpreter's site-packages (which could shadow our pinned versions).
    cfg = vd / "pyvenv.cfg"
    try:
        text = cfg.read_text()
        patched = text.replace("include-system-site-packages = true",
                               "include-system-site-packages = false")
        patched = patched.replace("system-site-packages = true",
                                  "system-site-packages = false")
        if patched != text:
            cfg.write_text(patched)
    except Exception:
        pass

    p(0.35, "Installing dependencies (this can take several minutes)…")
    seen = [0]

    def _line(line):
        seen[0] += 1
        p(0.35 + 0.55 * (seen[0] / (seen[0] + 8.0)), line.strip()[:60])

    _run([uv, "pip", "install",
          "--python", str(venv_python()),
          "-r", str(REQUIREMENTS)], env, on_line=_line)

    p(0.95, "Verifying installation…")
    ok, detail = _probe()
    if not ok:
        print("[bootstrap] import check failed:\n" + detail)
        raise RuntimeError(
            "Dependencies installed but the import check failed.\n"
            f"venv: {venv_python()}\n"
            f"{detail[:800]}"
        )

    # Only mark the venv usable once the import check has passed. Otherwise a
    # half-installed venv would satisfy quick_ready() and never be retried.
    _sentinel_path().write_text(json.dumps(_fingerprint()))

    p(1.0, "Setup done, but the GPU library check failed – see console."
        if detail.startswith("WARNING:") else "Dependencies ready.")
