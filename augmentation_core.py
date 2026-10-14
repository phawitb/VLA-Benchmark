"""Shared augmentation engine for datasets, real-robot evaluation and LIBERO.

Robot parameters retain SO101 units and are rejected by the LIBERO worker.
"""
import random
import numpy as np

_INTENSITY_SCALE = {1: .25, 2: .5, 3: 1., 4: 1.5, 5: 2.}

def _augp_camera(opts, intensity=3, fixed=None, *, rng=None, np_rng=None):
    rng = rng if rng is not None else random
    np_rng = np_rng if np_rng is not None else np.random
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if fixed:
        if opts.get("perspective") and fixed.get("persp_mag") is not None:
            m = float(fixed["persp_mag"])
            p["persp"] = np.array([[m, m], [-m, m], [m, -m], [-m, -m]])
        if opts.get("affine"):
            p["tx"] = float(fixed.get("tx", 0))
            p["ty"] = float(fixed.get("ty", 0))
            p["shear"] = float(fixed.get("shear", 0))
            p["scale"] = float(fixed.get("scale", 1.0))
        if opts.get("rotation") and fixed.get("angle") is not None:
            p["angle"] = float(fixed["angle"])
        return p or None
    if opts.get("perspective"):
        p["persp"] = np_rng.uniform(-0.05 * s, 0.05 * s, (4, 2))
    if opts.get("affine"):
        p["tx"] = rng.uniform(-0.04 * s, 0.04 * s)
        p["ty"] = rng.uniform(-0.04 * s, 0.04 * s)
        p["shear"] = rng.uniform(-0.03 * s, 0.03 * s)
        p["scale"] = rng.uniform(1 - 0.05 * s, 1 + 0.05 * s)
    if opts.get("rotation"):
        p["angle"] = rng.uniform(-4 * s, 4 * s)
    return p or None

def _augp_light(opts, intensity=3, fixed=None, *, rng=None):
    rng = rng if rng is not None else random
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if opts.get("occlusion"):
        # Normalized coordinates work for both preview and full-size videos.
        if fixed and fixed.get("occlusion") is not None:
            x, y, width, height = map(float, fixed["occlusion"])
            width, height = min(1.0, max(0.01, width)), min(1.0, max(0.01, height))
            p["occlusion"] = [min(max(0.0, x), 1 - width),
                              min(max(0.0, y), 1 - height), width, height]
        else:
            width = rng.uniform(0.10 * s, 0.30 * s)
            height = rng.uniform(0.10 * s, 0.30 * s)
            p["occlusion"] = [rng.uniform(0, 1 - width),
                              rng.uniform(0, 1 - height), width, height]
    if fixed:
        if opts.get("brightness") and fixed.get("brightness") is not None:
            p["brightness"] = float(fixed["brightness"])
        if opts.get("contrast") and fixed.get("contrast") is not None:
            p["contrast"] = float(fixed["contrast"])
        if opts.get("saturation") and fixed.get("saturation") is not None:
            p["saturation"] = float(fixed["saturation"])
        if opts.get("color_jitter") and fixed.get("hue") is not None:
            p["hue"] = int(fixed["hue"])
            p["sat_off"] = int(fixed.get("sat_off") or 0)
            p["val_off"] = int(fixed.get("val_off") or 0)
        if opts.get("shadow") and fixed.get("shadow_a") is not None:
            p["shadow_dir"] = fixed.get("shadow_dir", "left")
            p["shadow_a"] = float(fixed["shadow_a"])
        if opts.get("noise") and fixed.get("noise_s") is not None:
            p["noise_s"] = float(fixed["noise_s"])
        if opts.get("blur") and fixed.get("blur_s") is not None:
            p["blur_k"] = int(fixed.get("blur_k", 3))
            p["blur_s"] = float(fixed["blur_s"])
        return p or None
    if opts.get("brightness"):
        p["brightness"] = rng.uniform(1 - 0.25 * s, 1 + 0.25 * s)
    if opts.get("contrast"):
        p["contrast"] = rng.uniform(1 - 0.25 * s, 1 + 0.25 * s)
    if opts.get("saturation"):
        p["saturation"] = rng.uniform(1 - 0.35 * s, 1 + 0.35 * s)
    if opts.get("color_jitter"):
        h = int(8 * s)
        sv = int(15 * s)
        p["hue"] = rng.randint(-h, h)
        p["sat_off"] = rng.randint(-sv, sv)
        p["val_off"] = rng.randint(-sv, sv)
    if opts.get("shadow"):
        p["shadow_dir"] = rng.choice(["left", "right", "top", "bottom"])
        p["shadow_a"] = rng.uniform(0.15 * s, 0.4 * s)
    if opts.get("noise"):
        p["noise_s"] = rng.uniform(4 * s, 12 * s)
    if opts.get("blur"):
        p["blur_k"] = rng.choice([3, 5] if s <= 1.0 else [3, 5, 7])
        p["blur_s"] = rng.uniform(0.4 * s, 1.2 * s)
    return p or None

def _augp_robot(opts, n_frames, fps, intensity=3, fixed=None, *, rng=None):
    rng = rng if rng is not None else random
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if fixed:
        if opts.get("random_start") and n_frames > 30 and fixed.get("trim") is not None:
            p["trim"] = max(0, int(fixed["trim"]))
        if opts.get("initial_noise") and fixed.get("offset_sigma") is not None:
            sigma = float(fixed["offset_sigma"])
            p["joint_offset"] = [sigma * (1 if i % 2 == 0 else -1) for i in range(5)] + [0.0]
        if opts.get("trajectory_jitter") and fixed.get("jitter_s") is not None:
            p["jitter_s"] = float(fixed["jitter_s"])
        p["fps"] = fps
        return p or None
    if opts.get("random_start") and n_frames > 30:
        p["trim"] = rng.randint(1, max(1, int(n_frames * 0.12 * s)))
    if opts.get("initial_noise"):
        p["joint_offset"] = [rng.gauss(0, 1.5 * s) for _ in range(5)] + [0.0]
    if opts.get("trajectory_jitter"):
        p["jitter_s"] = rng.uniform(0.3 * s, 1.0 * s)
    p["fps"] = fps
    return p or None

def _eval_challenge_params(config, episode, fps, duration):
    """Build one reproducible challenge without changing process-wide RNG state."""
    if not config or not config.get("enabled"):
        return None
    intensity = max(1, min(5, int(config.get("intensity", 3))))
    mode = config.get("mode", "random")
    fixed = (config.get("fixed_params") or {}) if mode == "fix" else {}
    rng = random.Random(episode)
    np_rng = np.random.RandomState(episode)
    camera = config.get("camera") or {}
    light = config.get("light") or {}
    robot = config.get("robot") or {}
    if not light.get("enabled", True):
        light = {}
    if (config.get("occlusion") or {}).get("enabled"):
        light = {**light, "occlusion": True}
    cam_p = (_augp_camera(camera, intensity, fixed.get("camera"), rng=rng, np_rng=np_rng)
             if camera.get("enabled", True) else None)
    light_p = _augp_light(light, intensity, fixed.get("light"), rng=rng)
    n_frames = max(1, int(fps * duration))
    robot_p = (_augp_robot(robot, n_frames, fps, intensity, fixed.get("robot"), rng=rng)
               if robot.get("enabled", False) else None)
    if robot_p and "trim" in robot_p:
        robot_p["trim"] = min(max(0, robot_p["trim"]), n_frames - 1)
    # Fixed trim is also meaningful for short live evaluation episodes.
    if robot_p and mode == "fix" and robot.get("random_start"):
        robot_p["trim"] = min(max(0, int((fixed.get("robot") or {}).get("trim") or 0)), n_frames - 1)
    elif robot_p and robot.get("random_start") and "trim" not in robot_p and n_frames > 1:
        robot_p["trim"] = rng.randint(1, min(n_frames - 1, max(1, int(n_frames * .12 * _INTENSITY_SCALE[intensity]))))
    params = {"seed": episode, "mode": mode}
    for group, values in [("camera", cam_p), ("light", light_p), ("robot", robot_p)]:
        if values:
            params[group] = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in values.items()}
    return params if len(params) > 2 else None


def _aug_frame(frame, cam_p, light_p, cv2=None, np=None, rng=None):
    """Apply camera shift and lighting augmentation to a BGR frame."""
    if cv2 is None:
        import cv2
    if np is None:
        import numpy as np
    h, w = frame.shape[:2]
    f = frame
    if cam_p:
        if "persp" in cam_p:
            persp = cam_p["persp"]
            if not isinstance(persp, np.ndarray):
                persp = np.array(persp)
            pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            pts2 = (pts1 + persp * [w, h]).astype(np.float32)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            f = cv2.warpPerspective(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if "tx" in cam_p:
            M = np.float32([[cam_p.get("scale", 1), cam_p.get("shear", 0), cam_p["tx"] * w],
                            [0, cam_p.get("scale", 1), cam_p.get("ty", 0) * h]])
            f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if "angle" in cam_p:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), cam_p["angle"], 1.0)
            f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if light_p:
        ff = f.astype(np.float32)
        if "brightness" in light_p:
            ff *= light_p["brightness"]
        if "contrast" in light_p:
            mean = ff.mean(axis=(0, 1), keepdims=True)
            ff = (ff - mean) * light_p["contrast"] + mean
        ff = np.clip(ff, 0, 255).astype(np.uint8)
        if "saturation" in light_p:
            hsv = cv2.cvtColor(ff, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= light_p["saturation"]
            ff = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if "hue" in light_p:
            hsv = cv2.cvtColor(ff, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + light_p["hue"]) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + light_p["sat_off"], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + light_p["val_off"], 0, 255)
            ff = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if "shadow_a" in light_p:
            hh, ww = ff.shape[:2]
            a = light_p["shadow_a"]
            d = light_p["shadow_dir"]
            if d in ("left", "right"):
                g = np.linspace(1 - a, 1, ww) if d == "left" else np.linspace(1, 1 - a, ww)
                shadow = np.tile(g, (hh, 1))
            else:
                g = np.linspace(1 - a, 1, hh) if d == "top" else np.linspace(1, 1 - a, hh)
                shadow = np.tile(g.reshape(-1, 1), (1, ww))
            ff = (ff.astype(np.float32) * shadow[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
        if "noise_s" in light_p:
            noise = (rng if rng is not None else np.random).normal(0, light_p["noise_s"], ff.shape)
            ff = np.clip(ff.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if "blur_k" in light_p:
            ff = cv2.GaussianBlur(ff, (light_p["blur_k"], light_p["blur_k"]), light_p["blur_s"])
        f = ff
    if light_p and "occlusion" in light_p:
        h, w = f.shape[:2]
        x, y, width, height = light_p["occlusion"]
        rw, rh = max(1, round(width * w)), max(1, round(height * h))
        x0, y0 = min(round(x * w), w - rw), min(round(y * h), h - rh)
        f = f.copy()
        f[y0:y0 + rh, x0:x0 + rw] = 0
    return f
