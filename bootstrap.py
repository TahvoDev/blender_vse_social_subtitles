"""
bootstrap.py – one-click dependency setup for Auto-Subtitles.

Creates a private virtual environment based on Blender's own Python and
installs the pinned runtime dependencies into it using `uv`
(https://github.com/astral-sh/uv). The transcription worker is then launched
with that venv's interpreter, so heavy ML libraries never load inside
Blender's process.

All network access is gated on Blender's "Allow Online Access" preference
(bpy.app.online_access).
"""

import os
import sys
import json
import hashlib
import platform
import shutil
import tarfile
import tempfile
import zipfile
import subprocess
import urllib.request
from pathlib import Path

# Pinned uv release + verified sha256 of each archive.
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


# ---------------------------------------------------------------------------
# Environment / paths
# ---------------------------------------------------------------------------

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
    """Honour Blender's 'Allow Online Access' preference."""
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
    """Where the worker downloads speech models (HF_HOME); wiped on uninstall."""
    return data_dir() / "hf"


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
    """Pre-1.3 location of the uv binary + wheel cache (migrated, then removed)."""
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
    """Remove everything this add-on downloaded. True when nothing is left."""
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


def _sentinel_path() -> Path:
    return venv_dir() / ".auto_subtitles.json"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

def quick_ready() -> bool:
    """Cheap check: venv exists and matches the current Blender interpreter."""
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
    )
    global _active_proc
    try:
        proc = subprocess.Popen(
            [str(venv_python()), "-I", "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=clean_env(),
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
    for line in out.splitlines():
        if line.startswith("PROBE_BAD:"):
            bad = line[len("PROBE_BAD:"):]
    ok = (proc.returncode == 0) and (bad == "")
    if bad:
        out = f"failed to import: {bad}\n\n{out}"
    return ok, out


def probe_ready(timeout: int = 300) -> bool:
    """Full check: quick_ready() plus a real import of the key packages."""
    if not quick_ready():
        return False
    ok, _ = _probe(timeout=timeout)
    return ok


# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------

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
    """Return a usable uv executable, downloading a pinned build if needed."""
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

    tmpdir = Path(tempfile.mkdtemp(prefix="as_uv_"))
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


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_active_proc = None


def _run_tracked(cmd, timeout=30):
    """Run a short-lived child, tracked so terminate_active() can kill it."""
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
    """Kill a running setup subprocess (before purge/uninstall)."""
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
    """(Re)create the venv and install the pinned dependencies."""
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

    _sentinel_path().write_text(json.dumps(_fingerprint()))

    p(0.95, "Verifying installation…")
    ok, detail = _probe()
    if not ok:
        print("[bootstrap] import check failed:\n" + detail)
        raise RuntimeError(
            "Dependencies installed but the import check failed.\n"
            f"venv: {venv_python()}\n"
            f"{detail[:800]}"
        )
    p(1.0, "Dependencies ready.")
