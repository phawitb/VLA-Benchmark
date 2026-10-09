"""
Custom Python data collection script for LeRobot SO101.

Provides full control between episodes: auto-move to start position,
random offset, structured JSON status output for the web UI.

Usage:
    python collect_python.py \
        --robot-port /dev/tty.usbmodem... \
        --robot-id my_awesome_follower_arm \
        --teleop-port /dev/tty.usbmodem... \
        --teleop-id my_awesome_leader_arm \
        --cameras '{"top": {"index": 0, "w": 640, "h": 480}, "wrist": {"index": 1, "w": 640, "h": 480}}' \
        --repo-id phawitbinabik/my-dataset \
        --root ./data/my-dataset \
        --task "pick up the pink bow" \
        --num-episodes 50 \
        --fps 30 \
        --start-position '{"shoulder_pan": 0, ...}' \
        --random-strength 20 \
        --episode-seconds 15
"""

import argparse
import json
import os
import random
import signal
import sys
import time

import numpy as np

# ── Status output helpers ──────────────────────────────────────────
def emit(msg_type, **kwargs):
    """Print a JSON status line to stdout for the backend to parse."""
    d = {"type": msg_type, **kwargs}
    print(json.dumps(d, ensure_ascii=False), flush=True)


def log(text):
    emit("log", text=text)


def announce(text):
    emit("announcement", text=text)


# ── Graceful shutdown ──────────────────────────────────────────────
_stop_requested = False

def _signal_handler(sig, frame):
    global _stop_requested
    _stop_requested = True
    log("Stop signal received, finishing current episode...")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Main ───────────────────────────────────────────────────────────
def main():
    global _stop_requested

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--teleop-port", required=True)
    parser.add_argument("--teleop-id", default="my_awesome_leader_arm")
    parser.add_argument("--cameras", required=True, help="JSON dict of cameras")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start-position", default=None, help="JSON dict of joint positions")
    parser.add_argument("--random-strength", type=float, default=0, help="0-100 random offset strength")
    parser.add_argument("--episode-seconds", type=float, default=30, help="Max seconds per episode")
    parser.add_argument("--encoder-threads", type=int, default=4)
    args = parser.parse_args()

    cameras = json.loads(args.cameras)
    start_pos = json.loads(args.start_position) if args.start_position else None
    max_offset_deg = (args.random_strength / 100.0) * 10.0  # 0-10 degrees max

    JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # ── Import LeRobot ─────────────────────────────────────────────
    log("Importing LeRobot...")
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
    from lerobot.utils.constants import ACTION, OBS_STR

    # ── Connect cameras ────────────────────────────────────────────
    cam_configs = {}
    for cam_name, cam_cfg in cameras.items():
        cam_configs[cam_name] = OpenCVCameraConfig(
            index_or_path=cam_cfg["index"],
            width=cam_cfg.get("w", 640),
            height=cam_cfg.get("h", 480),
            fps=args.fps,
        )

    # ── Connect robot ──────────────────────────────────────────────
    log("Connecting robot...")
    emit("phase", phase="connecting")
    robot_config = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cam_configs,
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()
    log(f"Robot connected on {args.robot_port}")

    # ── Connect teleop ─────────────────────────────────────────────
    log("Connecting teleoperator...")
    teleop_config = SOLeaderTeleopConfig(
        port=args.teleop_port,
        id=args.teleop_id,
    )
    teleop = make_teleoperator_from_config(teleop_config)
    teleop.connect()
    log(f"Teleop connected on {args.teleop_port}")

    # ── Build dataset features ─────────────────────────────────────
    log("Creating dataset...")
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**obs_features, **action_features}

    # ── Create dataset ─────────────────────────────────────────────
    import shutil
    root_path = os.path.abspath(args.root)
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
        log(f"Removed existing dataset: {root_path}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=dataset_features,
        root=root_path,
        robot_type="so101_follower",
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=args.encoder_threads,
    )
    log(f"Dataset created: {args.repo_id}")

    # ── Helper: move robot to position smoothly ────────────────────
    def move_to_position(target_pos, duration=1.5):
        """Smoothly interpolate robot to target position."""
        obs = robot.get_observation()
        current = {j: float(obs[f"{j}.pos"]) for j in JOINTS}
        steps = int(duration * 30)  # 30 Hz movement
        for step in range(1, steps + 1):
            t = step / steps
            # Smooth easing (ease in-out)
            t = t * t * (3 - 2 * t)
            interp = {}
            for j in JOINTS:
                interp[f"{j}.pos"] = current[j] + t * (target_pos[j] - current[j])
            robot.send_action(interp)
            time.sleep(1.0 / 30)

    # ── Helper: get randomized start position ──────────────────────
    def get_randomized_start():
        if start_pos is None:
            return None
        if max_offset_deg <= 0:
            return dict(start_pos)
        pos = {}
        for j in JOINTS:
            if j == "gripper":
                pos[j] = start_pos.get(j, 0)
            else:
                offset = random.uniform(-max_offset_deg, max_offset_deg)
                pos[j] = start_pos.get(j, 0) + offset
        return pos

    # ── Helper: wait for stdin command ───────────────────────────────
    def wait_for_command():
        """Block until 'next' or 'stop' received via stdin. Returns the command."""
        import select
        while not _stop_requested:
            if select.select([sys.stdin], [], [], 0.5)[0]:
                cmd = sys.stdin.readline().strip()
                if cmd in ("next", "stop", ""):
                    return cmd
        return "stop"

    # ── Recording loop ─────────────────────────────────────────────
    # Flow per episode:
    #   1. Voice "Reset environment" → robot moves to start → user places object
    #   2. Wait for right arrow (→ stdin "next")
    #   3. Voice "Start episode X" → record teleop frames
    #   4. Wait for right arrow (→ stdin "next") → save episode → loop
    control_interval = 1.0 / args.fps
    recorded = 0
    import select

    try:
        for ep in range(args.num_episodes):
            if _stop_requested:
                break

            ep_num = ep + 1
            emit("episode_start", episode=ep_num, total=args.num_episodes)

            # ── Phase 1: Reset & move to start ─────────────────────
            announce("Reset environment")
            emit("phase", phase="resetting")

            if start_pos is not None:
                target = get_randomized_start()
                offset_info = ""
                if max_offset_deg > 0:
                    offset_info = f" (random +/-{max_offset_deg:.1f} deg)"
                log(f"Episode {ep_num}/{args.num_episodes}: Moving to start position{offset_info}...")
                move_to_position(target, duration=1.5)
                log("Reached start position. Place objects, then press -> to start recording.")
            else:
                log(f"Episode {ep_num}/{args.num_episodes}: Place objects, then press -> to start recording.")

            emit("phase", phase="waiting_start")

            # ── Wait for right arrow to start recording ────────────
            cmd = wait_for_command()
            if cmd == "stop" or _stop_requested:
                break

            # ── Phase 2: Recording ─────────────────────────────────
            announce(f"Start episode {ep_num}")
            emit("phase", phase="recording")
            log(f"Recording episode {ep_num}... Press -> to stop.")

            start_t = time.perf_counter()
            frame_count = 0

            while not _stop_requested:
                loop_start = time.perf_counter()
                elapsed = loop_start - start_t

                # Check max episode duration
                if elapsed >= args.episode_seconds:
                    log(f"Episode {ep_num} reached max duration ({args.episode_seconds}s)")
                    break

                # Check stdin for 'next' command (non-blocking)
                if select.select([sys.stdin], [], [], 0)[0]:
                    cmd = sys.stdin.readline().strip()
                    if cmd == "next":
                        log(f"Episode {ep_num} ended by user ({frame_count} frames, {elapsed:.1f}s)")
                        break
                    elif cmd == "stop":
                        _stop_requested = True
                        break

                # Get observation
                obs = robot.get_observation()

                # Get teleop action
                act = teleop.get_action()

                # Send action to robot (teleoperation)
                robot.send_action(act)

                # Build and add frame
                obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                act_frame = build_dataset_frame(dataset.features, act, prefix=ACTION)
                frame = {**obs_frame, **act_frame, "task": args.task}
                dataset.add_frame(frame)
                frame_count += 1

                # Emit frame progress every 30 frames
                if frame_count % 30 == 0:
                    emit("frame_progress", episode=ep_num, frames=frame_count,
                         elapsed=round(elapsed, 1), max_seconds=args.episode_seconds)

                # Maintain FPS
                dt = time.perf_counter() - loop_start
                sleep_t = control_interval - dt
                if sleep_t > 0:
                    time.sleep(sleep_t)

            # ── Save episode ───────────────────────────────────────
            if frame_count > 0:
                emit("phase", phase="saving")
                log(f"Saving episode {ep_num} ({frame_count} frames)...")
                dataset.save_episode()
                recorded += 1
                log(f"Episode {ep_num} saved ({recorded}/{args.num_episodes})")
                emit("episode_saved", episode=ep_num, frames=frame_count, recorded=recorded,
                     total=args.num_episodes)
            else:
                log(f"Episode {ep_num} skipped (0 frames)")
                dataset.clear_episode_buffer()

    except Exception as e:
        log(f"Error: {e}")
        emit("error", message=str(e))
    finally:
        # ── Finalize ───────────────────────────────────────────────
        emit("phase", phase="finalizing")
        log("Finalizing dataset...")
        try:
            dataset.finalize()
            log("Dataset finalized")
        except Exception as e:
            log(f"Finalize error: {e}")

        log("Disconnecting hardware...")
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass
        try:
            if teleop.is_connected:
                teleop.disconnect()
        except Exception:
            pass

        emit("done", recorded=recorded, total=args.num_episodes)
        announce(f"Recording complete. {recorded} episodes saved.")
        log(f"Done. {recorded}/{args.num_episodes} episodes recorded.")


if __name__ == "__main__":
    main()
