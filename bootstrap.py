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
import subprocess
import urllib.request
from pathlib import Path

# Pinned uv release + verified sha256 of the linux-gnu archives.
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
}

ADDON_DIR   = Path(__file__).parent
REQUIREMENTS = ADDON_DIR / "requirements.in"


# ---------------------------------------------------------------------------
# Environment / paths
# ---------------------------------------------------------------------------

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


def probe_ready(timeout: int = 180) -> bool:
    """Full check: quick_ready() plus a real import of the key packages."""
    if not quick_ready():
        return False
    try:
        r = subprocess.run(
            [str(venv_python()), "-c", "import faster_whisper, torch, transformers"],
            capture_output=True, timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False


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
            subprocess.run([str(target), "--version"], check=True, capture_output=True)
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
        with tarfile.open(tarpath, "r:gz") as tf:
            member = next((m for m in tf.getmembers()
                           if m.isfile() and (m.name == "uv" or m.name.endswith("/uv"))), None)
            if member is None:
                raise RuntimeError("uv binary not found in downloaded archive")
            src = tf.extractfile(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out)
        target.chmod(0o755)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return str(target)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _run(cmd, env, on_line=None):
    print(f"[bootstrap] $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"[bootstrap] {line}")
            if on_line:
                on_line(line)
    proc.wait()
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
    if not probe_ready():
        raise RuntimeError(
            "Dependencies installed but the import check failed. "
            "See the Blender system console for details."
        )
    p(1.0, "Dependencies ready.")
