"""
presets.py – persistent user presets for Auto-Subtitles.

The store is a small JSON file in the extension's user-data dir. It is global
across .blend files and survives add-on updates; Blender removes it on
uninstall. This module has no Blender dependency so it can be tested with a
bare Python interpreter.
"""

import json
import os
import tempfile
from pathlib import Path


SCHEMA_VERSION = 1
FILENAME = "presets.json"

SETTINGS = (
    "as_model",
    "as_language",
    "as_channel",
    "as_max_chars",
    "as_max_lines",
    "as_max_gap_fr",
    "as_rm_punct",
    "as_text_case",
)


def default_path(data_dir) -> Path:
    return Path(data_dir) / FILENAME


def load_presets(path) -> dict:
    """Return {name: settings}; tolerate missing, corrupt or old files."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(f"[Auto-Subtitles] Could not read presets: {exc}")
        return {}

    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        print(f"[Auto-Subtitles] Could not parse presets: {exc}")
        return {}

    if not isinstance(data, dict) or data.get("schema") != SCHEMA_VERSION:
        print("[Auto-Subtitles] Ignoring presets with an unsupported format")
        return {}

    stored = data.get("presets")
    if not isinstance(stored, dict):
        return {}

    presets = {}
    for name, values in stored.items():
        if not isinstance(name, str) or not isinstance(values, dict):
            continue
        presets[name] = {
            key: values[key] for key in SETTINGS if key in values
        }
    return presets


def save_presets(path, presets) -> None:
    """Atomically replace the preset file with {name: settings}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA_VERSION, "presets": presets}
    text = json.dumps(payload, indent=2, ensure_ascii=False)

    fd, tmp = tempfile.mkstemp(
        prefix="presets_", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def snapshot(source, keys=SETTINGS) -> dict:
    """Capture settings from a Blender scene-like object."""
    return {
        key: getattr(source, key) for key in keys if hasattr(source, key)
    }


def apply_values(target, values) -> int:
    """Assign known settings; return how many were applied."""
    applied = 0
    for key in SETTINGS:
        if key not in values or not hasattr(target, key):
            continue
        try:
            setattr(target, key, values[key])
            applied += 1
        except Exception as exc:
            print(f"[Auto-Subtitles] Could not apply preset field {key}: {exc}")
    return applied
