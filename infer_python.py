#!/usr/bin/env python3
"""
Standalone Python inference script for SmolVLA / ACT / pi0 policies.
Uses LeRobot's Python API directly instead of lerobot-rollout.

Usage:
    python infer_python.py \
        --model-path ./models/smolvla_V1 \
        --task "pick up the pink bow to the green bowl" \
        --robot-port /dev/tty.usbmodem5B141122411 \
        --robot-id my_awesome_follower_arm \
        --cameras '{"top": 0, "wrist": 1}' \
        --fps 30 \
        --width 320 --height 240
"""

import argparse
import json
import os
import signal
import select
import sys
import time
from pathlib import Path

from robot_handoff import disconnect_robot_hardware


WORKER_COMMANDS = {"run_task", "continue", "release_hardware", "shutdown"}


def prepare_runtime():
    """Load OpenCV before Torch to keep a single compatible OpenMP runtime on macOS."""
    import cv2
    import torch

    return cv2, torch


def cycle_boundary_reached(step: int, cycle_start: int, cycle_steps: int) -> bool:
    return cycle_steps > 0 and step - cycle_start >= cycle_steps


def normalize_cycle_command(value: str) -> str:
    command = str(value or "").strip().casefold()
    return command if command in {"continue", "stop"} else "stop"


def normalize_worker_command(value: str) -> dict | None:
    try:
        command = json.loads(str(value or ""))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(command, dict) or command.get("command") not in WORKER_COMMANDS:
        return None
    return command


def reset_policy_task_state(policy, language_cache: dict) -> None:
    queue = policy._queues.get("action")
    if hasattr(queue, "clear"):
        queue.clear()
    else:
        policy._queues["action"] = []
    language_cache.update(tokens=None, mask=None, embeddings=None)


def emit_worker_event(name: str, payload: dict) -> None:
    print(f"{name} {json.dumps(payload, separators=(',', ':'))}", flush=True)


def release_worker_hardware(runtime: dict, keep_torque: bool = True) -> None:
    robot = runtime.get("robot")
    if robot is not None and runtime.get("hardware_connected"):
        disconnect_robot_hardware(robot, keep_torque=keep_torque)
    runtime["hardware_connected"] = False


def run_worker_protocol_test(model_id: str) -> int:
    """Exercise the worker protocol without importing ML or hardware dependencies."""
    emit_worker_event("MODEL_READY", {"model_id": model_id})
    hardware_connected = False
    for line in sys.stdin:
        command = normalize_worker_command(line)
        if command is None:
            emit_worker_event("WORKER_ERROR", {"code": "INVALID_COMMAND", "error": "Invalid worker command"})
            continue
        name = command["command"]
        if name == "shutdown":
            return 0
        if name == "run_task":
            hardware_connected = True
            task = str(command.get("task", ""))
            emit_worker_event("HARDWARE_READY", {"task": task})
            emit_worker_event("CYCLE_READY", {"step": int(command.get("max_steps", 1)), "image": str(command.get("verification_image", ""))})
            continue
        if name == "release_hardware":
            hardware_connected = False
            emit_worker_event("HARDWARE_RELEASED", {"task": str(command.get("task", ""))})
            continue
        if name == "continue" and hardware_connected:
            emit_worker_event("CYCLE_READY", {"step": 1, "image": ""})
            continue
        emit_worker_event("WORKER_ERROR", {"code": "INVALID_STATE", "error": f"Cannot {name} in current state"})
    return 0


def save_verification_image(observation: dict, camera_names: list[str], target: Path) -> Path:
    import cv2
    import numpy as np

    frame = None
    for name in camera_names:
        for key in (name, f"observation.images.{name}"):
            if key in observation:
                frame = observation[key]
                break
        if frame is not None:
            break
    if frame is None:
        raise RuntimeError(f"No camera frame found for: {', '.join(camera_names)}")
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.dtype != np.uint8:
        scale = 255.0 if frame.size and float(frame.max()) <= 1.0 else 1.0
        frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp" + target.suffix)
    if not cv2.imwrite(str(temporary), frame):
        raise RuntimeError("Could not write verification image")
    temporary.replace(target)
    return target


def wait_for_cycle_command(timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while _running and time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            line = sys.stdin.readline()
            return normalize_cycle_command(line)
    return "stop"


def wait_for_worker_command(timeout: float | None = None) -> dict:
    deadline = None if timeout is None else time.monotonic() + timeout
    while _running and (deadline is None or time.monotonic() < deadline):
        wait = 0.1 if deadline is None else min(0.1, max(0.0, deadline - time.monotonic()))
        ready, _, _ = select.select([sys.stdin], [], [], wait)
        if not ready:
            continue
        line = sys.stdin.readline()
        if not line:
            return {"command": "shutdown"}
        command = normalize_worker_command(line)
        if command is not None:
            return command
        emit_worker_event("WORKER_ERROR", {"code": "INVALID_COMMAND", "error": "Invalid worker command"})
    return {"command": "shutdown"}


def poll_worker_command() -> dict | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:
        return {"command": "shutdown"}
    command = normalize_worker_command(line)
    if command is None:
        emit_worker_event("WORKER_ERROR", {"code": "INVALID_COMMAND", "error": "Invalid worker command"})
    return command


def apply_run_task_command(args, policy, language_cache: dict, command: dict) -> None:
    task = str(command.get("task", "")).strip()
    if not task:
        raise ValueError("run_task requires task")
    reset_policy_task_state(policy, language_cache)
    args.task = task
    args.max_steps = max(1, int(command.get("max_steps", 1)))
    args.verification_image = str(command.get("verification_image", ""))
    args.verification_timeout = float(command.get("verification_timeout", 120.0))


def parse_args():
    p = argparse.ArgumentParser(description="Python VLA inference")
    p.add_argument("--model-path", required=True, help="Path to model directory")
    p.add_argument("--task", default="", help="Task instruction text")
    p.add_argument("--robot-port", default="", help="Robot serial port")
    p.add_argument("--robot-id", default="my_awesome_follower_arm", help="Robot ID")
    p.add_argument("--cameras", default='{"top": 0, "wrist": 1}', help="JSON dict of camera_name: index")
    p.add_argument("--fps", type=int, default=30, help="Control loop FPS")
    p.add_argument("--width", type=int, default=320, help="Camera width")
    p.add_argument("--height", type=int, default=240, help="Camera height")
    p.add_argument("--fourcc", default="", help="Optional OpenCV camera FOURCC such as MJPG")
    p.add_argument("--chunk-size", type=int, default=0, help="Override chunk_size (0=use model default)")
    p.add_argument("--n-action-steps", type=int, default=0, help="Override n_action_steps (0=use model default)")
    p.add_argument("--device", default="", help="Device: cuda, mps, cpu (auto-detect if empty)")
    p.add_argument("--cache-language", action="store_true", help="Cache language tokens & embeddings (same task text every step)")
    p.add_argument("--max-steps", type=int, default=0, help="Actions per verification cycle (0=run until interrupted)")
    p.add_argument("--verification-image", default="", help="JPEG snapshot written at each action-cycle boundary")
    p.add_argument("--verification-timeout", type=float, default=120.0, help="Seconds to wait for continue/stop before disconnecting")
    p.add_argument("--persistent-worker", action="store_true", help="Keep the model loaded and accept JSON control commands")
    p.add_argument("--worker-protocol-test", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model-id", default="", help=argparse.SUPPRESS)
    return p.parse_args()


def detect_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Global flag for graceful shutdown
_running = True


def should_continue(running: bool, step: int, max_steps: int) -> bool:
    return bool(running) and (max_steps <= 0 or step < max_steps)


def _signal_handler(sig, frame):
    global _running
    print("\n[infer_python] Caught signal, stopping...")
    _running = False


def main():
    global _running
    cv2, torch = prepare_runtime()

    args = parse_args()

    if args.worker_protocol_test:
        return run_worker_protocol_test(args.model_id)
    if not args.robot_port:
        raise SystemExit("--robot-port is required")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    device = torch.device(args.device) if args.device else detect_device()
    print(f"[infer_python] Device: {device}")
    print(f"[infer_python] Model: {args.model_path}")
    print(f"[infer_python] Task: {args.task}")

    # --- Load model config to determine policy type ---
    import json as _json
    config_path = f"{args.model_path}/config.json"
    with open(config_path) as f:
        model_config = _json.load(f)
    policy_type = model_config.get("type", "unknown")
    print(f"[infer_python] Policy type: {policy_type}")

    # --- Load policy ---
    print("[infer_python] Loading policy...", flush=True)
    t0 = time.time()

    if policy_type == "smolvla":
        print("[infer_python] Importing SmolVLAPolicy...", flush=True)
        from lerobot.policies.smolvla import SmolVLAPolicy
        print("[infer_python] Calling from_pretrained...", flush=True)
        policy = SmolVLAPolicy.from_pretrained(args.model_path)
        print("[infer_python] from_pretrained done", flush=True)
    elif policy_type == "act":
        from lerobot.policies.act import ACTPolicy
        policy = ACTPolicy.from_pretrained(args.model_path)
    elif policy_type in ("pi0", "pi05"):
        from lerobot.policies.pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(args.model_path)
    else:
        print(f"[infer_python] ERROR: Unknown policy type '{policy_type}'")
        sys.exit(1)

    print("[infer_python] Calling policy.eval()...", flush=True)
    policy.eval()
    print(f"[infer_python] Policy loaded in {time.time() - t0:.1f}s", flush=True)

    # --- Build pre/post processors ---
    print("[infer_python] Building pre/post processors...", flush=True)
    from lerobot.policies import make_pre_post_processors
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("[infer_python] Pre/post processors ready", flush=True)

    language_cache = {"tokens": None, "mask": None, "embeddings": None}
    if args.persistent_worker:
        emit_worker_event("MODEL_READY", {"model_id": args.model_id})
        initial_command = wait_for_worker_command()
        if initial_command.get("command") != "run_task":
            return 0
        try:
            apply_run_task_command(args, policy, language_cache, initial_command)
        except (TypeError, ValueError) as exc:
            emit_worker_event("WORKER_ERROR", {"code": "INVALID_TASK", "error": str(exc)})
            return 1

    # --- Connect robot ---
    print("[infer_python] Connecting robot...", flush=True)
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cameras_dict = json.loads(args.cameras)

    # Build camera configs for LeRobot
    from lerobot.cameras.opencv import OpenCVCameraConfig
    cam_configs = {}
    for cam_name, cam_index in cameras_dict.items():
        cam_configs[cam_name] = OpenCVCameraConfig(
            index_or_path=cam_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc or None,
        )
        print(f"[infer_python] Camera '{cam_name}' index={cam_index}", flush=True)

    robot_config = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cam_configs,
    )
    print(f"[infer_python] Creating robot (port={args.robot_port})...", flush=True)
    robot = make_robot_from_config(robot_config)
    print("[infer_python] Calling robot.connect()...", flush=True)
    robot.connect()
    print("[infer_python] Robot connected", flush=True)
    hardware_connected = True
    if args.persistent_worker:
        emit_worker_event("HARDWARE_READY", {"task": args.task})

    # --- Build dataset features for observation frame ---
    print("[infer_python] Building dataset features...", flush=True)
    from lerobot.utils.feature_utils import hw_to_dataset_features
    print(f"[infer_python] robot.action_features = {robot.action_features}", flush=True)
    print(f"[infer_python] robot.observation_features = {robot.observation_features}", flush=True)
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    print(f"[infer_python] dataset_features keys: {list(dataset_features.keys())}", flush=True)

    # --- Override chunk_size / n_action_steps if requested ---
    if args.chunk_size > 0:
        policy.config.chunk_size = args.chunk_size
        print(f"[infer_python] Override chunk_size={args.chunk_size}", flush=True)
    if args.n_action_steps > 0:
        policy.config.n_action_steps = args.n_action_steps
        print(f"[infer_python] Override n_action_steps={args.n_action_steps}", flush=True)

    # --- Determine robot type ---
    robot_type = "so101_follower"

    # --- Monkey-patch sample_actions for detailed timing ---
    if policy_type == "smolvla":
        _orig_sample_actions = policy.model.sample_actions

        def _timed_sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
            import math
            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

            model = policy.model
            bsize = state.shape[0]
            dev = state.device

            if noise is None:
                actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
                noise = model.sample_noise(actions_shape, dev)

            t0 = time.time()
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            t_embed = time.time()

            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            _, past_key_values = model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=model.config.use_cache,
                fill_kv_cache=True,
            )
            t_kv = time.time()

            num_steps = model.config.num_steps
            dt = -1.0 / num_steps
            x_t = noise
            for step_i in range(num_steps):
                t_val = 1.0 + step_i * dt
                time_tensor = torch.tensor(t_val, dtype=torch.float32, device=dev).expand(bsize)
                v_t = model.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=time_tensor,
                )
                x_t = x_t + dt * v_t
            t_denoise = time.time()

            ms = lambda a, b: f"{(b-a)*1000:.1f}ms"
            print(
                f"[infer_python] PREDICT breakdown | "
                f"embed_prefix={ms(t0,t_embed)} "
                f"kv_cache={ms(t_embed,t_kv)} "
                f"denoise_x{num_steps}={ms(t_kv,t_denoise)} "
                f"TOTAL={ms(t0,t_denoise)}",
                flush=True,
            )
            return x_t

        policy.model.sample_actions = _timed_sample_actions
        print("[infer_python] Detailed PREDICT timing ENABLED", flush=True)

    # --- Setup language cache if requested ---
    # Exact key names from lerobot constants
    LANG_TOKENS_KEY = "observation.language.tokens"
    LANG_MASK_KEY = "observation.language.attention_mask"

    if args.cache_language and policy_type == "smolvla":
        print("[infer_python] Language caching ENABLED", flush=True)

        # Monkey-patch embed_language_tokens on vlm_with_expert to cache embeddings
        _vlm_expert = policy.model.vlm_with_expert
        _orig_embed_lang = _vlm_expert.embed_language_tokens

        def _cached_embed_language_tokens(tokens):
            if language_cache["embeddings"] is not None:
                return language_cache["embeddings"]
            result = _orig_embed_lang(tokens)
            language_cache["embeddings"] = result.detach()
            print(f"[infer_python] Cached language embeddings: {result.shape}", flush=True)
            return result

        _vlm_expert.embed_language_tokens = _cached_embed_language_tokens
    elif args.cache_language:
        print(f"[infer_python] Language caching not supported for policy type '{policy_type}', ignoring", flush=True)

    # --- Inference loop ---
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    period = 1.0 / args.fps
    step = 0
    cycle_start = 0

    # Video recording state (file-signal based)
    video_writer = None
    video_rec_meta = {}  # {width, height, cam_names}
    video_signal_path = Path("data/eval_videos/.recording.json")
    video_check_interval = 10  # check every N steps

    def _check_video_signal(obs, writer):
        """Check if recording should start/stop based on signal file."""
        import subprocess as _sp
        import numpy as np
        if video_signal_path.exists():
            if writer is None:
                try:
                    sig = json.loads(video_signal_path.read_text())
                    vpath = sig["video_path"]
                    vfps = sig.get("fps", args.fps)
                    vw = sig.get("width", args.width)
                    vh = sig.get("height", args.height)
                    cam_names = list(cameras_dict.keys())
                    out_w = vw * len(cam_names)
                    video_rec_meta["width"] = vw
                    video_rec_meta["height"] = vh
                    video_rec_meta["cam_names"] = cam_names
                    video_rec_meta["frame_count"] = 0
                    # Use ffmpeg for H.264 encoding (browser-compatible)
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-f", "rawvideo",
                        "-vcodec", "rawvideo",
                        "-pix_fmt", "bgr24",
                        "-s", f"{out_w}x{vh}",
                        "-r", str(vfps),
                        "-i", "-",
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-preset", "ultrafast",
                        "-crf", "23",
                        "-movflags", "+faststart",
                        vpath,
                    ]
                    writer = _sp.Popen(ffmpeg_cmd, stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                    print(f"[infer_python] Video recording started (ffmpeg H.264): {vpath}", flush=True)
                except Exception as e:
                    print(f"[infer_python] Video signal error: {e}", flush=True)
            if writer is not None:
                _write_video_frame(obs, writer)
        elif writer is not None:
            try:
                writer.stdin.close()
                writer.wait(timeout=10)
            except Exception:
                writer.kill()
            n = video_rec_meta.get("frame_count", 0)
            video_rec_meta.clear()
            print(f"[infer_python] Video recording stopped ({n} frames)", flush=True)
            writer = None
        return writer

    def _write_video_frame(obs, writer):
        """Write current observation frames to ffmpeg stdin."""
        import numpy as np
        frames = []
        vw = video_rec_meta["width"]
        vh = video_rec_meta["height"]
        for cam_name in video_rec_meta["cam_names"]:
            frame = None
            for key in (cam_name, f"observation.images.{cam_name}"):
                if key in obs:
                    frame = obs[key]
                    break
            if frame is not None:
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                frame = np.asarray(frame)
                if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                    frame = np.moveaxis(frame, 0, -1)
                if frame.dtype != np.uint8:
                    scale = 255.0 if frame.size and float(frame.max()) <= 1.0 else 1.0
                    frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (vw, vh))
            else:
                frame = np.zeros((vh, vw, 3), dtype=np.uint8)
            frames.append(frame)
        if frames:
            combined = np.hstack(frames)
            try:
                writer.stdin.write(combined.tobytes())
                video_rec_meta["frame_count"] = video_rec_meta.get("frame_count", 0) + 1
            except BrokenPipeError:
                pass

    print(f"[infer_python] Starting inference loop at {args.fps} FPS", flush=True)
    print("[infer_python] Press Ctrl+C to stop", flush=True)

    try:
        while _running:
            t0 = time.time()

            # Get observation from robot
            obs = robot.get_observation()
            t_obs = time.time()

            # Video recording (check signal every N steps)
            if step % video_check_interval == 0 or video_writer is not None:
                video_writer = _check_video_signal(obs, video_writer)

            # Build inference frame
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_features,
                device=device,
                task=args.task,
                robot_type=robot_type,
            )
            t_frame = time.time()

            # Preprocess
            obs_processed = preprocess(obs_frame)
            t_preprocess = time.time()

            # Cache language tokens after first preprocess, reuse on subsequent steps
            if args.cache_language and policy_type == "smolvla":
                if language_cache["tokens"] is None:
                    if LANG_TOKENS_KEY in obs_processed:
                        language_cache["tokens"] = obs_processed[LANG_TOKENS_KEY].clone()
                    if LANG_MASK_KEY in obs_processed:
                        language_cache["mask"] = obs_processed[LANG_MASK_KEY].clone()
                else:
                    if LANG_TOKENS_KEY in obs_processed:
                        obs_processed[LANG_TOKENS_KEY] = language_cache["tokens"]
                    if LANG_MASK_KEY in obs_processed:
                        obs_processed[LANG_MASK_KEY] = language_cache["mask"]

            # Predict action
            queue_empty = len(policy._queues.get("action", [])) == 0
            with torch.inference_mode():
                action = policy.select_action(obs_processed)
            t_action = time.time()

            # Postprocess
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            t_post = time.time()

            # Send to robot
            robot.send_action(action)
            t_send = time.time()

            step += 1

            # Detailed timing every second + on predict steps
            if step <= 3 or step % args.fps == 0 or queue_empty:
                ms = lambda a, b: f"{(b-a)*1000:.1f}ms"
                predict_tag = " ** PREDICT **" if queue_empty else " (reuse)"
                print(
                    f"[infer_python] Step {step}{predict_tag} | "
                    f"obs={ms(t0,t_obs)} frame={ms(t_obs,t_frame)} preprocess={ms(t_frame,t_preprocess)} "
                    f"action={ms(t_preprocess,t_action)} postprocess={ms(t_action,t_post)} send={ms(t_post,t_send)} "
                    f"TOTAL={ms(t0,t_send)}",
                    flush=True,
                )

            # Maintain target FPS
            elapsed = t_send - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            cycle_ready = cycle_boundary_reached(step, cycle_start, args.max_steps)
            worker_command = poll_worker_command() if args.persistent_worker and not cycle_ready else None
            if cycle_ready:
                verification_obs = robot.get_observation()
                target = Path(args.verification_image) if args.verification_image else Path(f"/tmp/hybridge-vla-{os.getpid()}.jpg")
                save_verification_image(verification_obs, list(cameras_dict), target)
                print(
                    "CYCLE_READY " + _json.dumps({"step": step, "image": str(target)}),
                    flush=True,
                )
                if not args.persistent_worker:
                    command = wait_for_cycle_command(args.verification_timeout)
                    print(f"[infer_python] Verification command: {command}", flush=True)
                    if command != "continue":
                        break
                    cycle_start = step
                    continue

                worker_command = wait_for_worker_command(args.verification_timeout)

            if args.persistent_worker and worker_command is not None:
                name = worker_command.get("command")
                print(f"[infer_python] Worker command: {name}", flush=True)
                if name == "continue":
                    cycle_start = step
                    continue
                if name == "shutdown":
                    break
                if name != "release_hardware":
                    emit_worker_event("WORKER_ERROR", {"code": "INVALID_STATE", "error": f"Cannot {name} at cycle boundary"})
                    continue

                handoff_runtime = {"robot": robot, "hardware_connected": hardware_connected}
                release_worker_hardware(handoff_runtime, keep_torque=True)
                hardware_connected = handoff_runtime["hardware_connected"]
                completed_task = args.task
                emit_worker_event("HARDWARE_RELEASED", {"task": completed_task})
                while _running:
                    next_command = wait_for_worker_command()
                    next_name = next_command.get("command")
                    if next_name == "shutdown":
                        break
                    if next_name != "run_task":
                        emit_worker_event("WORKER_ERROR", {"code": "INVALID_STATE", "error": f"Cannot {next_name} while hardware is released"})
                        continue
                    try:
                        apply_run_task_command(args, policy, language_cache, next_command)
                        robot.connect()
                        hardware_connected = True
                        step = 0
                        cycle_start = 0
                        emit_worker_event("HARDWARE_READY", {"task": args.task})
                    except Exception as exc:
                        emit_worker_event("WORKER_ERROR", {"code": "HARDWARE_CONNECT_FAILED", "error": str(exc)})
                        break
                    break
                if not hardware_connected:
                    break

    except Exception as e:
        print(f"[infer_python] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print(f"[infer_python] Stopping after {step} steps")
        if video_writer is not None:
            try:
                video_writer.stdin.close()
                video_writer.wait(timeout=10)
            except Exception:
                video_writer.kill()
            print(f"[infer_python] Video recording finalized ({video_rec_meta.get('frame_count', 0)} frames)")
        try:
            if hardware_connected:
                disconnect_robot_hardware(robot, keep_torque=False)
                print("[infer_python] Robot disconnected")
        except Exception as e:
            print(f"[infer_python] Disconnect error: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
