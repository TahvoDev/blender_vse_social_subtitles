"""Caption entrance animations for generated VSE text strips."""

ENTRANCES = ("none", "pop", "fade", "both")


def _dedupe(keys):
    out = []
    for frame, value in keys:
        if out and out[-1][0] == frame:
            out[-1] = (frame, value)
        else:
            out.append((frame, value))
    return out


def plan_entrance(rest_size, start, end, entrance,
                  pop_start=0.88, pop_overshoot=1.15,
                  pop_frames=6, fade_frames=8):
    """Return {data_path: [(frame, value), ...]} for one text strip."""
    start, end = int(start), int(end)
    rest = max(1, int(rest_size))
    plan = {}
    if end <= start or entrance not in ENTRANCES or entrance == "none":
        return plan

    if entrance in ("pop", "both"):
        pop_end = min(start + max(2, int(pop_frames)), end)
        peak = start + max(1, round((pop_end - start) * 0.4))
        if peak >= pop_end:
            peak = pop_end - 1
        keys = [(start, max(1, round(rest * pop_start)))]
        if peak > start:
            keys.append((peak, max(1, round(rest * pop_overshoot))))
        keys.append((pop_end, rest))
        plan["font_size"] = _dedupe(keys)

    if entrance in ("fade", "both"):
        fade_end = min(start + max(1, int(fade_frames)), end)
        plan["blend_alpha"] = [(start, 0.0), (fade_end, 1.0)]

    return plan


def find_strip_fcurves(scene, strip, data_path):
    """Find the fcurves driving one property of a VSE strip."""
    anim = getattr(scene, "animation_data", None)
    action = getattr(anim, "action", None) if anim else None
    if action is None:
        return []

    safe_name = strip.name.replace("\\", "\\\\").replace('"', '\\"')
    needle = f'["{safe_name}"].{data_path}'
    fcurves = []

    if getattr(action, "is_action_layered", False):
        for layer in getattr(action, "layers", ()):
            for key_strip in getattr(layer, "strips", ()):
                for channelbag in getattr(key_strip, "channelbags", ()):
                    fcurves.extend(
                        fc for fc in channelbag.fcurves
                        if needle in fc.data_path
                    )
        return fcurves

    for fc in getattr(action, "fcurves", ()) or ():
        if needle in fc.data_path:
            fcurves.append(fc)
    return fcurves


def ease_strip_fcurves(scene, strip, data_path):
    for fc in find_strip_fcurves(scene, strip, data_path):
        for key in fc.keyframe_points:
            key.interpolation = "BEZIER"
            key.handle_left_type = "AUTO_CLAMPED"
            key.handle_right_type = "AUTO_CLAMPED"
        fc.update()


def apply_entrance(strip, plan, scene):
    """Insert the planned keyframes on a strip and ease them."""
    for data_path, keys in plan.items():
        if not keys:
            continue
        keys = sorted(keys)
        for frame, value in keys:
            setattr(strip, data_path, value)
            strip.keyframe_insert(data_path=data_path, frame=frame)
        setattr(strip, data_path, keys[-1][1])
        ease_strip_fcurves(scene, strip, data_path)
