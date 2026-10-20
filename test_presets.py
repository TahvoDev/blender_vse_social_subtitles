"""
Self-checks for preset storage.

Runs with a bare Python interpreter (no Blender or third-party deps):

    python3 test_presets.py
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeScene:
    as_model = "base"
    as_language = "auto"
    as_channel = 5
    as_max_chars = 42
    as_max_lines = 2
    as_max_gap_fr = 36
    as_rm_punct = False
    as_text_case = "default"
    as_anim_entrance = "none"
    as_anim_pop_start = 0.88
    as_anim_pop_overshoot = 1.15
    as_anim_pop_frames = 6
    as_anim_fade_frames = 8


def test_snapshot_fields():
    p = _load("presets")
    values = p.snapshot(FakeScene())
    assert set(values) == set(p.SETTINGS), values
    assert values["as_model"] == "base"
    assert values["as_max_gap_fr"] == 36
    assert values["as_anim_entrance"] == "none"
    assert values["as_anim_pop_start"] == 0.88


def test_round_trip_and_order():
    p = _load("presets")
    with tempfile.TemporaryDirectory(prefix="as_presets_") as tmp:
        path = Path(tmp) / "presets.json"
        store = {
            "Second": {"as_model": "small", "as_max_chars": 24},
            "First": {"as_model": "base", "as_rm_punct": True},
        }
        p.save_presets(path, store)
        loaded = p.load_presets(path)
        assert loaded == store, loaded
        assert list(loaded) == ["Second", "First"], list(loaded)


def test_overwrite_preserves_other_entries():
    p = _load("presets")
    with tempfile.TemporaryDirectory(prefix="as_presets_") as tmp:
        path = Path(tmp) / "presets.json"
        p.save_presets(path, {
            "A": {"as_model": "tiny"},
            "B": {"as_model": "base"},
        })
        store = p.load_presets(path)
        store["B"] = {"as_model": "large-v3"}
        p.save_presets(path, store)
        loaded = p.load_presets(path)
        assert loaded["A"] == {"as_model": "tiny"}
        assert loaded["B"] == {"as_model": "large-v3"}


def test_missing_file_is_empty():
    p = _load("presets")
    with tempfile.TemporaryDirectory(prefix="as_presets_") as tmp:
        assert p.load_presets(Path(tmp) / "missing.json") == {}


def test_corrupt_file_is_empty():
    p = _load("presets")
    with tempfile.TemporaryDirectory(prefix="as_presets_") as tmp:
        path = Path(tmp) / "presets.json"
        path.write_text("{not json", encoding="utf-8")
        assert p.load_presets(path) == {}


def test_wrong_schema_is_empty():
    p = _load("presets")
    with tempfile.TemporaryDirectory(prefix="as_presets_") as tmp:
        path = Path(tmp) / "presets.json"
        path.write_text(json.dumps({
            "schema": 99,
            "presets": {"A": {"as_model": "base"}},
        }), encoding="utf-8")
        assert p.load_presets(path) == {}
        try:
            p.load_presets(path, strict=True)
        except ValueError:
            pass
        else:
            raise AssertionError("strict load must reject an unsupported schema")


def test_strict_load_rejects_corrupt_file():
    p = _load("presets")
    with tempfile.TemporaryDirectory(prefix="as_presets_") as tmp:
        path = Path(tmp) / "presets.json"
        path.write_text("{not json", encoding="utf-8")
        try:
            p.load_presets(path, strict=True)
        except ValueError:
            pass
        else:
            raise AssertionError("strict load must reject corrupt JSON")


def test_validate_clamps_and_rejects():
    p = _load("presets")
    enums = {
        "as_model": {"base", "small"},
        "as_text_case": {"upper"},
    }
    ranges = {
        "as_channel": (1, 128),
        "as_max_chars": (10, 120),
    }
    values = {
        "as_model": "bogus",
        "as_channel": 999,
        "as_max_chars": "x",
        "as_rm_punct": 1,
        "as_text_case": "upper",
    }
    clean = p.validate(values, enums, ranges, ("as_rm_punct",))
    assert clean == {"as_channel": 128, "as_text_case": "upper"}, clean
    assert p.validate(
        {"as_rm_punct": True}, {}, {}, ("as_rm_punct",)
    ) == {"as_rm_punct": True}


def test_validate_clamps_floats():
    p = _load("presets")
    floats = {
        "as_anim_pop_start": (0.1, 1.0),
        "as_anim_pop_overshoot": (1.0, 3.0),
    }
    clean = p.validate({
        "as_anim_pop_start": 0.05,
        "as_anim_pop_overshoot": 9.0,
    }, {}, {}, (), floats)
    assert clean == {
        "as_anim_pop_start": 0.1,
        "as_anim_pop_overshoot": 3.0,
    }, clean
    assert p.validate({"as_anim_pop_start": True}, {}, {}, (), floats) == {}
    assert p.validate({"as_anim_pop_start": "x"}, {}, {}, (), floats) == {}


def test_unknown_and_malformed_entries_are_dropped():
    p = _load("presets")
    with tempfile.TemporaryDirectory(prefix="as_presets_") as tmp:
        path = Path(tmp) / "presets.json"
        path.write_text(json.dumps({
            "schema": 1,
            "presets": {
                "Good": {"as_model": "base", "unknown": 1},
                "Bad": "not a dict",
                "Empty": {},
            },
        }), encoding="utf-8")
        loaded = p.load_presets(path)
        assert loaded == {"Good": {"as_model": "base"}, "Empty": {}}, loaded


def test_atomic_save_leaves_no_temp_file():
    p = _load("presets")
    with tempfile.TemporaryDirectory(prefix="as_presets_") as tmp:
        path = Path(tmp) / "presets.json"
        p.save_presets(path, {"A": {"as_model": "base"}})
        assert list(Path(tmp).iterdir()) == [path], list(Path(tmp).iterdir())
        assert json.loads(path.read_text(encoding="utf-8"))["schema"] == 1


def test_apply_values_ignores_unknown_keys():
    p = _load("presets")
    scene = FakeScene()
    applied = p.apply_values(scene, {
        "as_channel": 9,
        "unknown": 1,
    })
    assert applied == 1, applied
    assert scene.as_channel == 9


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"ok    {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
