"""
Self-checks for caption entrance planning.

Runs with a bare Python interpreter (no Blender required):

    python3 test_animations.py
"""

import importlib.util
import os
import sys
from types import SimpleNamespace


HERE = os.path.dirname(os.path.abspath(__file__))


class FakeStrip:
    def __init__(self, name="sub_0000"):
        self.name = name
        self.font_size = 0
        self.blend_alpha = 1.0
        self.inserts = []

    def keyframe_insert(self, data_path, frame):
        self.inserts.append((data_path, frame))
        return True


class FakeKey:
    interpolation = "LINEAR"
    handle_left_type = "FREE"
    handle_right_type = "FREE"


class FakeFCurve:
    def __init__(self, data_path):
        self.data_path = data_path
        self.keyframe_points = [FakeKey(), FakeKey()]
        self.updated = False

    def update(self):
        self.updated = True


class FakeAction:
    def __init__(self, fcurves, layered=True):
        self.is_action_layered = layered
        self.fcurves = fcurves
        if layered:
            bag = SimpleNamespace(fcurves=fcurves)
            strip = SimpleNamespace(channelbags=[bag])
            self.layers = [SimpleNamespace(strips=[strip])]


def _scene(action):
    return SimpleNamespace(animation_data=SimpleNamespace(action=action))


def _load(name):
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_none_and_invalid_entrances_are_empty():
    a = _load("animations")
    assert a.plan_entrance(60, 100, 160, "none") == {}
    assert a.plan_entrance(60, 100, 160, "bogus") == {}
    assert a.plan_entrance(60, 100, 100, "pop") == {}


def test_pop_plan_has_start_overshoot_and_rest():
    a = _load("animations")
    plan = a.plan_entrance(
        60, 100, 160, "pop",
        pop_start=0.88, pop_overshoot=1.15, pop_frames=6,
    )
    assert plan == {"font_size": [(100, 53), (102, 69), (106, 60)]}, plan


def test_fade_plan_goes_transparent_to_opaque():
    a = _load("animations")
    plan = a.plan_entrance(60, 100, 160, "fade", fade_frames=8)
    assert plan == {"blend_alpha": [(100, 0.0), (108, 1.0)]}, plan


def test_both_combines_pop_and_fade():
    a = _load("animations")
    plan = a.plan_entrance(60, 100, 160, "both", pop_frames=6, fade_frames=8)
    assert set(plan) == {"font_size", "blend_alpha"}, plan
    assert plan["font_size"][-1] == (106, 60), plan
    assert plan["blend_alpha"][-1] == (108, 1.0), plan


def test_short_strip_clamps_to_its_end():
    a = _load("animations")
    plan = a.plan_entrance(60, 100, 103, "both", pop_frames=30, fade_frames=30)
    for keys in plan.values():
        assert all(100 <= frame <= 103 for frame, _value in keys), plan
        assert keys[-1][0] == 103, plan


def test_one_frame_strip_has_no_room_for_animation():
    a = _load("animations")
    plan = a.plan_entrance(60, 100, 101, "both", pop_frames=6, fade_frames=8)
    assert plan["font_size"] == [(100, 53), (101, 60)], plan
    assert plan["blend_alpha"] == [(100, 0.0), (101, 1.0)], plan


def test_find_strip_fcurves_matches_exact_strip_name():
    a = _load("animations")
    own = FakeFCurve('sequence_editor.strips_all["sub_0000"].font_size')
    decoy = FakeFCurve('sequence_editor.strips_all["sub_00000"].font_size')
    scene = _scene(FakeAction([own, decoy]))
    found = a.find_strip_fcurves(scene, FakeStrip("sub_0000"), "font_size")
    assert found == [own], found


def test_find_strip_fcurves_supports_legacy_actions():
    a = _load("animations")
    own = FakeFCurve('sequence_editor.strips_all["sub_0000"].font_size')
    scene = _scene(FakeAction([own], layered=False))
    assert a.find_strip_fcurves(scene, FakeStrip(), "font_size") == [own]


def test_apply_entrance_sorts_keys_restores_rest_and_eases():
    a = _load("animations")
    curve = FakeFCurve('sequence_editor.strips_all["sub_0000"].font_size')
    scene = _scene(FakeAction([curve]))
    strip = FakeStrip()
    a.apply_entrance(strip, {"font_size": [(106, 60), (100, 53), (102, 69)]}, scene)
    assert strip.inserts == [
        ("font_size", 100), ("font_size", 102), ("font_size", 106),
    ], strip.inserts
    assert strip.font_size == 60, strip.font_size
    assert curve.updated
    for key in curve.keyframe_points:
        assert key.interpolation == "BEZIER"
        assert key.handle_left_type == "AUTO_CLAMPED"
        assert key.handle_right_type == "AUTO_CLAMPED"


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
