"""VLA-Benchmark: Setup, Data Collection, Training, Inference & Evaluation server."""

import asyncio
import atexit
import base64
import datetime
import io
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model_registry import get_model_record, load_local_models, load_remote_models, merge_model_records
from robot_handoff import disconnect_robot_hardware
from vla_execution import VlaProcessManager, build_infer_command, build_worker_command

ROOT = Path(__file__).parent
DATA_DIR = (ROOT / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUGMENT_CONFIG_DIR = (ROOT / "augment_configs").resolve()
AUGMENT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")

robot_io_lock = threading.RLock()
robot_operation_lock = threading.Lock()
robot_operation_owner = None


class RobotBusyError(RuntimeError):
    pass


def acquire_robot_operation(owner: str) -> bool:
    global robot_operation_owner
    with robot_operation_lock:
        if robot_operation_owner is not None:
            return False
        robot_operation_owner = owner
        return True


def release_robot_operation(owner: str) -> None:
    global robot_operation_owner
    with robot_operation_lock:
        if robot_operation_owner == owner:
            robot_operation_owner = None


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _write_config(cfg: dict) -> None:
    target = ROOT / "config.yaml"
    temporary = ROOT / "config.yaml.tmp"
    temporary.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False))
    temporary.replace(target)


def _read_env():
    env = {}
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def _write_env(env: dict):
    env_path = ROOT / ".env"
    lines = [f"{k}={v}" for k, v in env.items()]
    env_path.write_text("\n".join(lines) + "\n")


def _load_model_registry(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    cameras = set(cfg.get("robot", {}).get("cameras", {}).keys())
    local = load_local_models(ROOT / "models", cameras)
    hf_user = str(cfg.get("hf_repo_name", "")).strip()
    if not hf_user:
        return local
    try:
        from huggingface_hub import HfApi, hf_hub_download

        token = _read_env().get("HF_TOKEN") or None
        api = HfApi(token=token)
        models = [
            {"id": m.id, "name": m.id.rsplit("/", 1)[-1]}
            for m in api.list_models(author=hf_user, sort="lastModified", limit=100)
            if any(m.id.rsplit("/", 1)[-1].lower().startswith(p) for p in ("smolvla", "act", "pi0", "pi05", "model"))
        ]
        datasets = [d.id for d in api.list_datasets(author=hf_user, limit=200)]

        def _hf_tasks(repo_id: str) -> list[str]:
            import tempfile
            import pyarrow.parquet as pq
            with tempfile.TemporaryDirectory() as tmp:
                path = hf_hub_download(repo_id, "meta/tasks.parquet", repo_type="dataset", local_dir=tmp, token=token)
                return pq.read_table(path).to_pydict().get("task", [])

        remote = load_remote_models(models, datasets, _hf_tasks)
        return merge_model_records(local, remote)
    except Exception as exc:
        print(f"[Model Registry] Hugging Face discovery failed: {exc}")
        return local


CONFIG = load_config()
RESULTS_DIR = ROOT / CONFIG["storage"]["results_dir"]
IMAGES_DIR = ROOT / CONFIG["storage"]["images_dir"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VLA-Benchmark")
vla_manager = VlaProcessManager(max_log_lines=500)
atexit.register(vla_manager.stop)

app.mount("/static", StaticFiles(directory="static"), name="static")
Path("data/eval_videos").mkdir(parents=True, exist_ok=True)
app.mount("/data/eval_videos", StaticFiles(directory="data/eval_videos"), name="eval_videos")

ROBOT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
robot_state = {"robot": None, "connected": False}
cam_state = {}
teleop_state = {"process": None, "running": False, "log_lines": [], "log_lock": threading.Lock()}

import_status = {"running": False, "progress": 0, "status": "", "error": None, "name": None}
augment_state = {"running": False, "progress": 0, "status": "", "error": None}


# ══════════ Robot Hardware ══════════

def connect_robot():
    import cv2
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    port = robot_cfg.get("port", "/dev/tty.usbmodem5B141122411")
    rcfg = SOFollowerRobotConfig(port=port, id=robot_cfg.get("id", "my_awesome_follower_arm"), cameras={})
    rob = make_robot_from_config(rcfg)
    rob.connect()
    robot_state["robot"] = rob
    robot_state["connected"] = True
    print(f"[Robot] Connected on {port}")

    cams = robot_cfg.get("cameras", {})
    for name, cam_cfg in cams.items():
        cap = cv2.VideoCapture(cam_cfg.get("index", 0))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("w", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("h", 480))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        state = {"cap": cap, "frame": None, "lock": threading.Lock(), "running": True, "stopped": threading.Event()}
        cam_state[name] = state
        ok = cap.isOpened()
        print(f"[Camera] '{name}' (idx {cam_cfg.get('index', 0)}): {'OK' if ok else 'FAILED'}")

        def loop(st=state, nm=name, cam_cfg=cam_cfg):
            try:
                while st["running"]:
                    try:
                        c = st["cap"]
                        if c is None or not c.isOpened():
                            c = cv2.VideoCapture(cam_cfg.get("index", 0))
                            c.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("w", 640))
                            c.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("h", 480))
                            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                            st["cap"] = c
                            time.sleep(1)
                            continue
                        c.grab()
                        ret, frame = c.retrieve()
                        if ret:
                            with st["lock"]:
                                st["frame"] = frame.copy()
                        else:
                            c.release()
                            st["cap"] = None
                            time.sleep(1)
                    except Exception as e:
                        print(f"[Camera] '{nm}' error: {e}")
                        time.sleep(1)
            finally:
                if st["cap"]:
                    st["cap"].release()
                    st["cap"] = None
                st["stopped"].set()

        threading.Thread(target=loop, daemon=True).start()


def disconnect_robot(*, keep_torque: bool = False):
    # Signal all camera threads to stop
    for name, st in list(cam_state.items()):
        st["running"] = False
    # Wait for threads to finish
    for name, st in list(cam_state.items()):
        st["stopped"].wait(timeout=5)
    # Force release any remaining captures
    for name, st in list(cam_state.items()):
        try:
            cap = st.get("cap")
            if cap is not None:
                cap.release()
                st["cap"] = None
                print(f"[Camera] '{name}' force-released")
        except Exception:
            pass
    cam_state.clear()

    with robot_io_lock:
        rob = robot_state["robot"]
        if rob:
            try:
                disconnect_robot_hardware(rob, keep_torque=keep_torque)
            except Exception:
                if keep_torque:
                    raise
        robot_state["robot"] = None
        robot_state["connected"] = False
    print("[Robot] Disconnected")


def robot_get_positions():
    with robot_io_lock:
        rob = robot_state["robot"]
        obs = rob.get_observation()
        return {j: round(float(obs[f"{j}.pos"]), 2) for j in ROBOT_JOINTS}


def robot_send_positions(positions, owner=None):
    with robot_operation_lock:
        active_owner = robot_operation_owner
    if active_owner is not None and active_owner != owner:
        raise RobotBusyError(f"Robot hardware is busy with {active_owner}")
    with robot_io_lock:
        rob = robot_state["robot"]
        obs = rob.get_observation()
        action = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in ROBOT_JOINTS}
        for j, v in positions.items():
            action[f"{j}.pos"] = float(v)
        rob.send_action(action)


# ══════════ Restart All ══════════

@app.post("/api/restart")
async def restart_all():
    """Kill this process and restart it — same as Ctrl+C then re-run."""
    import signal

    def _restart():
        time.sleep(0.5)  # let the HTTP response go out
        print("[Restart] Restarting server process...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_restart, daemon=True).start()
    return {"ok": True, "restarting": True}


# ══════════ Camera Check ══════════

@app.post("/api/cameras/check")
async def check_cameras(request: Request):
    """Scan all camera indices (0-9) and return hstacked preview of every available one."""
    import cv2

    results = []
    frames = []
    max_idx = 10

    for idx in range(max_idx):
        # Check if any active thread is using this index
        active_name = None
        for name, st in cam_state.items():
            cfg = load_config().get("robot", {}).get("cameras", {}).get(name, {})
            if cfg.get("index") == idx and st.get("running"):
                active_name = name
                break

        if active_name:
            st = cam_state[active_name]
            with st["lock"]:
                frame = st["frame"].copy() if st["frame"] is not None else None
            ok = frame is not None
        else:
            cap = cv2.VideoCapture(idx)
            ok = False
            frame = None
            if cap.isOpened():
                for _ in range(3):
                    cap.grab()
                ret, frame = cap.read()
                ok = ret and frame is not None
            cap.release()

        if not ok:
            continue

        label = f"idx:{idx}"
        if active_name:
            label += f" ({active_name})"
        results.append({"index": idx, "name": active_name, "ok": True})
        # Large index number
        cv2.putText(frame, str(idx), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        # Smaller name label below
        if active_name:
            cv2.putText(frame, active_name, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frames.append(frame)

    if not frames:
        return {"ok": True, "results": results, "preview": None, "message": "No cameras found"}

    # hstack all frames (resize to same height)
    target_h = max(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        if f.shape[0] != target_h:
            ratio = target_h / f.shape[0]
            f = cv2.resize(f, (int(f.shape[1] * ratio), target_h))
        resized.append(f)
    stacked = np.hstack(resized)
    _, buf = cv2.imencode(".jpg", stacked, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = base64.b64encode(buf.tobytes()).decode()

    return {"ok": True, "results": results, "preview": preview}


# ══════════ Setup & Config API ══════════

@app.get("/api/config")
async def get_config():
    cfg = load_config()
    return {"ok": True, "config": cfg}


@app.post("/api/config/save")
async def save_config(request: Request):
    body = await request.json()
    cfg = load_config()
    if "robot" in body:
        cfg["robot"] = body["robot"]
    if "teleop" in body:
        cfg["teleop"] = body["teleop"]
    if "data_collection" in body:
        cfg["data_collection"] = body["data_collection"]
    if "gpu_server" in body:
        cfg["gpu_server"] = body["gpu_server"]
    if "hf_repo_name" in body:
        cfg["hf_repo_name"] = body["hf_repo_name"]
    _write_config(cfg)
    return {"ok": True}


@app.get("/api/env")
async def get_env():
    env = _read_env()
    masked = {}
    for k, v in env.items():
        if v and len(v) > 8:
            masked[k] = v[:4] + "..." + v[-4:]
        else:
            masked[k] = "***" if v else ""
    return {"ok": True, "env": masked}


@app.post("/api/env/save")
async def save_env(request: Request):
    body = await request.json()
    env = _read_env()
    for k, v in body.items():
        if isinstance(v, str):
            env[k] = v
    _write_env(env)
    for k, v in env.items():
        os.environ[k] = v
    return {"ok": True}


@app.get("/api/models/registry")
async def model_registry():
    records = _load_model_registry()
    return {"ok": True, "models": records}


@app.get("/api/robot/status")
async def robot_status():
    return {
        "connected": robot_state["connected"],
        "operation_owner": robot_operation_owner,
    }


@app.post("/api/robot/start")
async def robot_start():
    if robot_state["connected"]:
        return {"ok": True, "message": "Already connected"}
    try:
        connect_robot()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/robot/stop")
async def robot_stop():
    if not robot_state["connected"]:
        return {"ok": True, "message": "Not connected"}
    disconnect_robot()
    return {"ok": True}


@app.get("/api/robot/positions")
async def robot_positions():
    if not robot_state["connected"]:
        return {"ok": False, "error": "Robot not connected"}
    try:
        pos = robot_get_positions()
        return {"ok": True, "positions": pos}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/robot/move")
async def robot_move(request: Request):
    if not robot_state["connected"]:
        return {"ok": False, "error": "Robot not connected"}
    try:
        body = await request.json()
        positions = body.get("positions", {})
        robot_send_positions(positions, owner="control_panel")
        return {"ok": True}
    except RobotBusyError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════ Datasets API ══════════

@app.get("/api/datasets")
async def list_datasets():
    data_dir = ROOT / "data"
    datasets = []
    if data_dir.exists():
        for d in sorted(data_dir.iterdir()):
            info_path = d / "meta" / "info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text())
                    task_list = []
                    tasks_path = d / "meta" / "tasks.parquet"
                    if tasks_path.exists():
                        try:
                            import pyarrow.parquet as pq
                            tbl = pq.read_table(str(tasks_path))
                            task_list = tbl.to_pydict().get("task", [])
                        except Exception:
                            pass
                    datasets.append({
                        "name": d.name,
                        "total_episodes": info.get("total_episodes", 0),
                        "total_frames": info.get("total_frames", 0),
                        "fps": info.get("fps", 0),
                        "robot_type": info.get("robot_type", ""),
                        "tasks": task_list,
                        "cameras": sorted(
                            key for key, feature in info.get("features", {}).items()
                            if key.startswith("observation.images.")
                            and isinstance(feature, dict)
                            and feature.get("dtype") == "video"
                        ),
                    })
                except Exception:
                    pass
    return {"ok": True, "datasets": datasets}


@app.get("/api/datasets/{name}/episodes")
async def list_dataset_episodes(name: str):
    import pyarrow.parquet as pq

    ds_dir = ROOT / "data" / name
    info_path = ds_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}

    info = json.loads(info_path.read_text())
    fps = info.get("fps", 30)

    episodes = []
    ep_dir = ds_dir / "meta" / "episodes"
    if ep_dir.exists():
        for chunk_dir in sorted(ep_dir.iterdir()):
            if chunk_dir.is_dir():
                for pf in sorted(chunk_dir.glob("*.parquet")):
                    try:
                        tbl = pq.read_table(str(pf))
                        d = tbl.to_pydict()
                        for i in range(len(d.get("episode_index", []))):
                            row = {}
                            for k, v in d.items():
                                val = v[i]
                                if hasattr(val, "tolist"):
                                    val = val.tolist()
                                row[k] = val
                            row.setdefault("index", row.get("episode_index", len(episodes)))
                            row.setdefault("length", row.get("num_frames", 0))
                            episodes.append(row)
                    except Exception:
                        pass
    episodes.sort(key=lambda r: r.get("episode_index", r.get("index", 0)))

    return {"ok": True, "fps": fps, "total_episodes": info.get("total_episodes", 0), "episodes": episodes}


@app.get("/api/datasets/{name}/episode/{ep_idx}")
async def get_episode_data(name: str, ep_idx: int):
    import pyarrow.parquet as pq

    ds_dir = ROOT / "data" / name
    data_dir = ds_dir / "data"
    if not data_dir.exists():
        return {"ok": False, "error": "Dataset not found"}

    info = json.loads((ds_dir / "meta" / "info.json").read_text())
    fps = info.get("fps", 30)

    frames = []
    for chunk_dir in sorted(data_dir.iterdir()):
        if chunk_dir.is_dir():
            for pf in sorted(chunk_dir.glob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                    d = tbl.to_pydict()
                    for i in range(len(d.get("episode_index", []))):
                        if d["episode_index"][i] == ep_idx:
                            frames.append({
                                "state": d["observation.state"][i],
                                "action": d["action"][i],
                                "frame_index": d["frame_index"][i],
                                "timestamp": d.get("timestamp", [0])[i],
                            })
                except Exception:
                    pass

    if not frames:
        return {"ok": False, "error": f"Episode {ep_idx} not found"}

    frames.sort(key=lambda f: f["frame_index"])

    videos = {}
    ep_dir = ds_dir / "meta" / "episodes"
    if ep_dir.exists():
        for chunk_dir in sorted(ep_dir.iterdir()):
            if chunk_dir.is_dir():
                for pf in sorted(chunk_dir.glob("*.parquet")):
                    try:
                        tbl = pq.read_table(str(pf))
                        d = tbl.to_pydict()
                        for i in range(len(d.get("episode_index", []))):
                            if d["episode_index"][i] == ep_idx:
                                for cam in _get_camera_keys(info):
                                    ck = f"videos/{cam}/chunk_index"
                                    fk = f"videos/{cam}/file_index"
                                    ft = f"videos/{cam}/from_timestamp"
                                    tt = f"videos/{cam}/to_timestamp"
                                    if ck in d and fk in d:
                                        ci = d[ck][i]
                                        fi = d[fk][i]
                                        videos[cam] = {
                                            "path": f"/data/{name}/videos/{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4",
                                            "from_timestamp": d[ft][i] if ft in d else 0,
                                            "to_timestamp": d[tt][i] if tt in d else 0,
                                        }
                                break
                    except Exception:
                        pass

    return {"ok": True, "fps": fps, "total_frames": len(frames), "frames": frames, "videos": videos}


@app.get("/api/datasets/{name}/info")
async def dataset_info(name: str):
    ds_dir = ROOT / "data" / name
    info_path = ds_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}
    info = json.loads(info_path.read_text())
    tasks = []
    tp = ds_dir / "meta" / "tasks.parquet"
    if tp.exists():
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(str(tp))
            tasks = tbl.to_pydict().get("task", [])
        except Exception:
            pass
    return {"ok": True, "info": info, "tasks": tasks}


def _compute_stats_json(data_rows, new_dir):
    stat_keys = ["action", "observation.state", "timestamp", "frame_index",
                 "episode_index", "index", "task_index"]
    stats = {}
    for key in stat_keys:
        vals = []
        for row in data_rows:
            v = row.get(key)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                vals.append(v)
            else:
                vals.append([v])
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        stats[key] = {
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "count": [len(arr)],
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    # Detect cameras from info.json
    info_file = new_dir / "meta" / "info.json"
    if info_file.exists():
        cam_keys = _get_camera_keys(json.loads(info_file.read_text()))
    else:
        cam_keys = ["observation.images.top", "observation.images.wrist"]
    for cam in cam_keys:
        stats[cam] = {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
            "count": [len(data_rows)],
            "q01": [[[0.01]], [[0.01]], [[0.01]]],
            "q10": [[[0.1]], [[0.1]], [[0.1]]],
            "q50": [[[0.5]], [[0.5]], [[0.5]]],
            "q90": [[[0.9]], [[0.9]], [[0.9]]],
            "q99": [[[0.99]], [[0.99]], [[0.99]]],
        }
    (new_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))


@app.post("/api/datasets/{name}/remove-camera")
async def remove_dataset_camera(name: str, request: Request):
    import shutil
    import pyarrow.parquet as pq

    body = await request.json()
    camera = str(body.get("camera", "")).strip()
    output_name = str(body.get("output_name", "")).strip().replace(" ", "-").replace("/", "-")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        return {"ok": False, "error": "Invalid source dataset name"}
    if not output_name or not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):
        return {"ok": False, "error": "Invalid output name."}
    if not re.fullmatch(r"observation\.images\.[A-Za-z0-9_-]+", camera):
        return {"ok": False, "error": "Invalid camera key"}

    source_dir = ROOT / "data" / name
    output_dir = ROOT / "data" / output_name
    info_path = source_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}
    if output_dir.exists():
        return {"ok": False, "error": f"Dataset '{output_name}' already exists"}

    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    cameras = sorted(k for k, v in features.items() if k.startswith("observation.images.") and isinstance(v, dict) and v.get("dtype") == "video")
    if camera not in cameras:
        return {"ok": False, "error": f"Camera '{camera}' is not present in this dataset"}
    if len(cameras) < 2:
        return {"ok": False, "error": "Cannot remove the dataset's only camera"}

    try:
        shutil.copytree(source_dir, output_dir)
        new_info = json.loads((output_dir / "meta" / "info.json").read_text())
        new_info.get("features", {}).pop(camera, None)
        (output_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

        stats_path = output_dir / "meta" / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            stats.pop(camera, None)
            stats_path.write_text(json.dumps(stats, indent=4))

        video_prefix = f"videos/{camera}/"
        stats_prefix = f"stats/{camera}/"
        for parquet_path in (output_dir / "meta" / "episodes").rglob("*.parquet"):
            table = pq.read_table(parquet_path)
            keep_columns = [
                col for col in table.column_names
                if not col.startswith(video_prefix) and not col.startswith(stats_prefix)
            ]
            pq.write_table(table.select(keep_columns), parquet_path)

        shutil.rmtree(output_dir / "videos" / camera, ignore_errors=True)

        return {
            "ok": True,
            "name": output_name,
            "removed_camera": camera,
            "remaining_cameras": [cam for cam in cameras if cam != camera],
            "episodes": new_info.get("total_episodes", 0),
            "frames": new_info.get("total_frames", 0),
        }
    except Exception as e:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.post("/api/datasets/{name}/remove-beginning")
async def remove_dataset_beginning(name: str, request: Request):
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq

    body = await request.json()
    try:
        seconds = float(body.get("seconds", 2))
    except (TypeError, ValueError):
        return {"ok": False, "error": "Seconds must be a number"}
    output_name = str(body.get("output_name", "")).strip().replace(" ", "-").replace("/", "-")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        return {"ok": False, "error": "Invalid source dataset name"}
    if not output_name or not re.fullmatch(r"[A-Za-z0-9_.-]+", output_name):
        return {"ok": False, "error": "Invalid output name."}
    if not math.isfinite(seconds) or seconds <= 0:
        return {"ok": False, "error": "Seconds must be greater than zero"}

    source_dir = ROOT / "data" / name
    output_dir = ROOT / "data" / output_name
    info_path = source_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}
    if output_dir.exists():
        return {"ok": False, "error": f"Dataset '{output_name}' already exists"}

    try:
        info = json.loads(info_path.read_text())
        fps = float(info.get("fps", 0))
        if not math.isfinite(fps) or fps <= 0:
            return {"ok": False, "error": "Dataset FPS must be greater than zero"}
        remove_count = max(1, int(round(seconds * fps)))

        data_rows_by_episode = {}
        for parquet_path in sorted((source_dir / "data").rglob("*.parquet")):
            for row in pq.read_table(parquet_path).to_pylist():
                data_rows_by_episode.setdefault(int(row["episode_index"]), []).append(row)
        for rows in data_rows_by_episode.values():
            rows.sort(key=lambda row: row.get("frame_index", 0))
        if not data_rows_by_episode:
            return {"ok": False, "error": "No frame data found"}
        for episode_index, rows in sorted(data_rows_by_episode.items()):
            if len(rows) <= remove_count:
                return {"ok": False, "error": f"Removing {seconds:g}s would empty episode {episode_index}"}

        episode_meta = {}
        for parquet_path in sorted((source_dir / "meta" / "episodes").rglob("*.parquet")):
            for row in pq.read_table(parquet_path).to_pylist():
                episode_meta[int(row["episode_index"])] = row

        new_rows = []
        new_episodes = []
        timestamp_shift = remove_count / fps
        for episode_index, rows in sorted(data_rows_by_episode.items()):
            retained = rows[remove_count:]
            dataset_from_index = len(new_rows)
            for frame_index, row in enumerate(retained):
                new_row = dict(row)
                new_row["frame_index"] = frame_index
                new_row["index"] = len(new_rows)
                if new_row.get("timestamp") is not None:
                    new_row["timestamp"] = max(0.0, float(new_row["timestamp"]) - timestamp_shift)
                new_rows.append(new_row)

            meta = dict(episode_meta.get(episode_index, {"episode_index": episode_index}))
            meta["length"] = len(retained)
            meta["dataset_from_index"] = dataset_from_index
            meta["dataset_to_index"] = len(new_rows)
            for key, value in list(meta.items()):
                if key.startswith("videos/") and key.endswith("/from_timestamp") and value is not None:
                    meta[key] = float(value) + timestamp_shift
            new_episodes.append(meta)

        shutil.copytree(source_dir, output_dir)
        shutil.rmtree(output_dir / "data", ignore_errors=True)
        shutil.rmtree(output_dir / "meta" / "episodes", ignore_errors=True)
        data_output = output_dir / "data" / "chunk-000"
        episode_output = output_dir / "meta" / "episodes" / "chunk-000"
        data_output.mkdir(parents=True)
        episode_output.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(new_rows), data_output / "file-000.parquet")
        pq.write_table(pa.Table.from_pylist(new_episodes), episode_output / "file-000.parquet")

        new_info = dict(info)
        new_info["total_frames"] = len(new_rows)
        (output_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))
        _compute_stats_json(new_rows, output_dir)

        return {
            "ok": True, "name": output_name, "episodes": len(new_episodes),
            "frames": len(new_rows), "removed_frames": remove_count * len(new_episodes), "seconds": seconds,
        }
    except Exception as e:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.post("/api/datasets/combine")
async def combine_datasets(request: Request):
    import shutil

    body = await request.json()
    source_names = body.get("datasets", [])
    new_name = body.get("name", "").strip()
    if not source_names or len(source_names) < 2:
        return {"ok": False, "error": "Select at least 2 datasets"}
    if not new_name:
        return {"ok": False, "error": "Enter a name for the new dataset"}
    new_name = new_name.replace(" ", "-").replace("/", "-")

    new_dir = ROOT / "data" / new_name
    if new_dir.exists():
        return {"ok": False, "error": f"Dataset '{new_name}' already exists"}

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        sources = []
        for sname in source_names:
            sdir = ROOT / "data" / sname
            info_path = sdir / "meta" / "info.json"
            if not info_path.exists():
                return {"ok": False, "error": f"Dataset '{sname}' not found"}
            info = json.loads(info_path.read_text())
            sources.append({"name": sname, "dir": sdir, "info": info})

        base_info = dict(sources[0]["info"])

        all_tasks = []
        task_set = set()
        for src in sources:
            tp = src["dir"] / "meta" / "tasks.parquet"
            if tp.exists():
                try:
                    tbl = pq.read_table(str(tp))
                    for t in tbl.to_pydict().get("task", []):
                        if t not in task_set:
                            task_set.add(t)
                            all_tasks.append(t)
                except Exception:
                    pass
        task_to_idx = {t: i for i, t in enumerate(all_tasks)}

        combined_data = []
        combined_ep = []
        new_ep_idx = 0
        global_frame_idx = 0

        for src in sources:
            sdir = src["dir"]
            ep_meta_list = []
            ep_dir = sdir / "meta" / "episodes"
            if ep_dir.exists():
                for pf in sorted(ep_dir.rglob("*.parquet")):
                    try:
                        tbl = pq.read_table(str(pf))
                    except Exception:
                        continue
                    d = tbl.to_pydict()
                    for i in range(len(d.get("episode_index", []))):
                        ep_meta_list.append({k: v[i] for k, v in d.items()})
            ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

            data_dir = sdir / "data"
            src_frames = {}
            if data_dir.exists():
                for pf in sorted(data_dir.rglob("*.parquet")):
                    try:
                        tbl = pq.read_table(str(pf))
                    except Exception:
                        continue
                    d = tbl.to_pydict()
                    for i in range(len(d.get("episode_index", []))):
                        ep = d["episode_index"][i]
                        row = {}
                        for k, v in d.items():
                            val = v[i]
                            if hasattr(val, "tolist"):
                                val = val.tolist()
                            row[k] = val
                        src_frames.setdefault(ep, []).append(row)

            for ep in src_frames:
                src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

            src_tasks = []
            tp = sdir / "meta" / "tasks.parquet"
            if tp.exists():
                try:
                    tbl = pq.read_table(str(tp))
                    src_tasks = tbl.to_pydict().get("task", [])
                except Exception:
                    pass

            for old_ep in sorted(src_frames.keys()):
                frames = src_frames[old_ep]
                ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)
                ep_global_start = global_frame_idx

                for fi, row in enumerate(frames):
                    new_row = dict(row)
                    new_row["episode_index"] = new_ep_idx
                    new_row["frame_index"] = fi
                    new_row["index"] = global_frame_idx
                    old_ti = row.get("task_index", 0)
                    if old_ti < len(src_tasks) and src_tasks[old_ti] in task_to_idx:
                        new_row["task_index"] = task_to_idx[src_tasks[old_ti]]
                    else:
                        new_row["task_index"] = 0
                    combined_data.append(new_row)
                    global_frame_idx += 1

                new_meta = {}
                new_meta["episode_index"] = new_ep_idx
                if ep_meta and "tasks" in ep_meta:
                    new_meta["tasks"] = ep_meta["tasks"]
                elif src_tasks:
                    new_meta["tasks"] = [src_tasks[0]]
                else:
                    new_meta["tasks"] = all_tasks[:1] if all_tasks else [""]
                new_meta["length"] = len(frames)
                new_meta["data/chunk_index"] = 0
                new_meta["data/file_index"] = 0
                new_meta["dataset_from_index"] = ep_global_start
                new_meta["dataset_to_index"] = global_frame_idx

                for cam in _get_camera_keys(base_info):
                    cam_key = f"videos/{cam}"
                    if ep_meta:
                        orig_chunk = ep_meta.get(f"{cam_key}/chunk_index", 0)
                        orig_file = ep_meta.get(f"{cam_key}/file_index", 0)
                        src_vid = sdir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                    else:
                        src_vid = None
                    new_meta[f"{cam_key}/from_timestamp"] = ep_meta.get(f"{cam_key}/from_timestamp", 0.0) if ep_meta else 0.0
                    new_meta[f"{cam_key}/to_timestamp"] = ep_meta.get(f"{cam_key}/to_timestamp", 0.0) if ep_meta else 0.0
                    new_meta[f"{cam_key}/chunk_index"] = 0
                    new_meta[f"{cam_key}/file_index"] = 0
                    new_meta[f"_src_vid_{cam}"] = str(src_vid) if src_vid and src_vid.exists() else None

                if ep_meta:
                    for k, v in ep_meta.items():
                        if k.startswith("stats/"):
                            new_meta[k] = v

                combined_ep.append(new_meta)
                new_ep_idx += 1

        if not combined_data:
            return {"ok": False, "error": "No data found in source datasets"}

        new_dir.mkdir(parents=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True)

        for cam in _get_camera_keys(base_info):
            vid_dir = new_dir / "videos" / cam / "chunk-000"
            vid_dir.mkdir(parents=True)
            src_vids = []
            src_vid_set = set()
            for ep_meta in combined_ep:
                sv = ep_meta.get(f"_src_vid_{cam}")
                if sv and sv not in src_vid_set:
                    src_vid_set.add(sv)
                    src_vids.append(sv)
            vid_map = {}
            for fi, sv in enumerate(src_vids):
                dst = vid_dir / f"file-{fi:03d}.mp4"
                shutil.copy2(sv, str(dst))
                vid_map[sv] = fi
            cam_key = f"videos/{cam}"
            for ep_meta in combined_ep:
                sv = ep_meta.pop(f"_src_vid_{cam}", None)
                if sv and sv in vid_map:
                    ep_meta[f"{cam_key}/file_index"] = vid_map[sv]

        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))

        for ep in combined_ep:
            for k in list(ep.keys()):
                if k.startswith("_src_vid_"):
                    del ep[k]
        cols = {k: [r[k] for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))

        import pandas as pd
        tasks_df = pd.DataFrame({"task_index": list(range(len(all_tasks)))}, index=pd.Index(all_tasks, name="task"))
        tasks_df.to_parquet(str(new_dir / "meta" / "tasks.parquet"))

        new_info = dict(base_info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["total_tasks"] = len(all_tasks)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))
        _compute_stats_json(combined_data, new_dir)

        return {
            "ok": True,
            "name": new_name,
            "episodes": new_ep_idx,
            "frames": len(combined_data),
            "tasks": all_tasks,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.post("/api/datasets/{name}/random-subset")
async def random_subset_dataset(name: str, request: Request):
    import shutil
    import random

    body = await request.json()
    num_episodes = int(body.get("num_episodes", 10))
    new_name = body.get("new_name", "").strip()
    if not new_name:
        return {"ok": False, "error": "Enter a name for the new dataset"}
    new_name = new_name.replace(" ", "-").replace("/", "-")
    if not re.match(r'^[a-zA-Z0-9_\-]+$', new_name):
        return {"ok": False, "error": "Invalid name."}

    src_dir = ROOT / "data" / name
    new_dir = ROOT / "data" / new_name
    info_path = src_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": f"Dataset '{name}' not found"}
    if new_dir.exists():
        return {"ok": False, "error": f"Dataset '{new_name}' already exists"}

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        src_info = json.loads(info_path.read_text())
        total_eps = src_info.get("total_episodes", 0)
        if num_episodes > total_eps:
            return {"ok": False, "error": f"Requested {num_episodes} episodes but dataset only has {total_eps}"}
        if num_episodes < 1:
            return {"ok": False, "error": "Need at least 1 episode"}

        selected_eps = sorted(random.sample(range(total_eps), num_episodes))

        all_tasks = []
        tp = src_dir / "meta" / "tasks.parquet"
        if tp.exists():
            try:
                tbl = pq.read_table(str(tp))
                all_tasks = tbl.to_pydict().get("task", [])
            except Exception:
                pass

        ep_meta_list = []
        ep_dir = src_dir / "meta" / "episodes"
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep_meta_list.append({k: v[i] for k, v in d.items()})
        ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

        data_dir = src_dir / "data"
        src_frames = {}
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep = d["episode_index"][i]
                    if ep not in selected_eps:
                        continue
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    src_frames.setdefault(ep, []).append(row)

        for ep in src_frames:
            src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

        combined_data = []
        combined_ep = []
        new_ep_idx = 0
        global_frame_idx = 0

        for old_ep in selected_eps:
            frames = src_frames.get(old_ep, [])
            if not frames:
                continue
            ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)
            ep_global_start = global_frame_idx

            for fi, row in enumerate(frames):
                new_row = dict(row)
                new_row["episode_index"] = new_ep_idx
                new_row["frame_index"] = fi
                new_row["index"] = global_frame_idx
                combined_data.append(new_row)
                global_frame_idx += 1

            new_meta = {}
            new_meta["episode_index"] = new_ep_idx
            if ep_meta and "tasks" in ep_meta:
                new_meta["tasks"] = ep_meta["tasks"]
            elif all_tasks:
                new_meta["tasks"] = [all_tasks[0]]
            else:
                new_meta["tasks"] = [""]
            new_meta["length"] = len(frames)
            new_meta["data/chunk_index"] = 0
            new_meta["data/file_index"] = 0
            new_meta["dataset_from_index"] = ep_global_start
            new_meta["dataset_to_index"] = global_frame_idx

            for cam in _get_camera_keys(src_info):
                cam_key = f"videos/{cam}"
                if ep_meta:
                    orig_chunk = ep_meta.get(f"{cam_key}/chunk_index", 0)
                    orig_file = ep_meta.get(f"{cam_key}/file_index", 0)
                    src_vid = src_dir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                else:
                    src_vid = None
                new_meta[f"{cam_key}/from_timestamp"] = ep_meta.get(f"{cam_key}/from_timestamp", 0.0) if ep_meta else 0.0
                new_meta[f"{cam_key}/to_timestamp"] = ep_meta.get(f"{cam_key}/to_timestamp", 0.0) if ep_meta else 0.0
                new_meta[f"{cam_key}/chunk_index"] = 0
                new_meta[f"{cam_key}/file_index"] = 0
                new_meta[f"_src_vid_{cam}"] = str(src_vid) if src_vid and src_vid.exists() else None

            if ep_meta:
                for k, v in ep_meta.items():
                    if k.startswith("stats/"):
                        new_meta[k] = v

            combined_ep.append(new_meta)
            new_ep_idx += 1

        if not combined_data:
            return {"ok": False, "error": "No data found for selected episodes"}

        new_dir.mkdir(parents=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True)

        for cam in _get_camera_keys(src_info):
            vid_dir = new_dir / "videos" / cam / "chunk-000"
            vid_dir.mkdir(parents=True)
            src_vids = []
            src_vid_set = set()
            for ep_meta in combined_ep:
                sv = ep_meta.get(f"_src_vid_{cam}")
                if sv and sv not in src_vid_set:
                    src_vid_set.add(sv)
                    src_vids.append(sv)
            vid_map = {}
            for fi, sv in enumerate(src_vids):
                dst = vid_dir / f"file-{fi:03d}.mp4"
                shutil.copy2(sv, str(dst))
                vid_map[sv] = fi
            cam_key = f"videos/{cam}"
            for ep_meta in combined_ep:
                sv = ep_meta.pop(f"_src_vid_{cam}", None)
                if sv and sv in vid_map:
                    ep_meta[f"{cam_key}/file_index"] = vid_map[sv]

        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))

        for ep in combined_ep:
            for k in list(ep.keys()):
                if k.startswith("_src_vid_"):
                    del ep[k]
        cols = {k: [r[k] for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))

        import pandas as pd
        if all_tasks:
            tasks_df = pd.DataFrame({"task_index": list(range(len(all_tasks)))}, index=pd.Index(all_tasks, name="task"))
            tasks_df.to_parquet(str(new_dir / "meta" / "tasks.parquet"))

        new_info = dict(src_info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))
        _compute_stats_json(combined_data, new_dir)

        return {
            "ok": True,
            "name": new_name,
            "episodes": new_ep_idx,
            "frames": len(combined_data),
            "selected": selected_eps,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.delete("/api/datasets/{name}")
async def delete_dataset(name: str):
    import shutil
    dataset_dir = ROOT / "data" / name
    if not dataset_dir.exists() or not (dataset_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}
    try:
        shutil.rmtree(dataset_dir)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/datasets/{name}/update-task")
async def update_dataset_task(name: str, request: Request):
    data = await request.json()
    new_task = data.get("task", "").strip()
    if not new_task:
        return {"ok": False, "error": "Task instruction is required"}
    dataset_dir = ROOT / "data" / name
    if not dataset_dir.exists():
        return {"ok": False, "error": "Dataset not found"}
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {"ok": False, "error": "tasks.parquet not found"}
    selected_episodes = data.get("episodes", None)
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        tbl = pq.read_table(str(tasks_path))
        td = tbl.to_pydict()
        old_tasks = td.get("task", [])

        def _update_ep_file(pf, ep_filter=None):
            ep_tbl = pq.read_table(str(pf))
            ep_d = ep_tbl.to_pydict()
            if "tasks" not in ep_d or "episode_index" not in ep_d:
                return 0
            count = 0
            indices = ep_d["episode_index"]
            new_tasks_col = []
            for j, task_list in enumerate(ep_d["tasks"]):
                should_update = ep_filter is None or indices[j] in ep_filter
                if should_update:
                    if isinstance(task_list, list):
                        new_tasks_col.append([new_task] * len(task_list))
                    else:
                        new_tasks_col.append(new_task)
                    count += 1
                else:
                    new_tasks_col.append(task_list)
            if count > 0:
                ep_d["tasks"] = new_tasks_col
                pq.write_table(pa.table(ep_d), str(pf))
            return count

        ep_filter = set(selected_episodes) if selected_episodes is not None else None
        updated_eps = 0

        ep_dir = dataset_dir / "meta" / "episodes"
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                updated_eps += _update_ep_file(pf, ep_filter)

        all_tasks_set = set(old_tasks)
        all_tasks_set.add(new_task)
        unique_tasks = set()
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                ep_tbl = pq.read_table(str(pf))
                ep_d = ep_tbl.to_pydict()
                for task_list in ep_d.get("tasks", []):
                    if isinstance(task_list, list):
                        unique_tasks.update(task_list)
                    else:
                        unique_tasks.add(task_list)
        if not unique_tasks:
            unique_tasks = all_tasks_set
        unique_tasks_list = sorted(unique_tasks)
        td["task"] = unique_tasks_list
        td["task_index"] = list(range(len(unique_tasks_list)))
        pq.write_table(pa.table(td), str(tasks_path))

        return {"ok": True, "task": new_task, "updated_episodes": updated_eps}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/datasets/{name}/rename")
async def rename_dataset(name: str, request: Request):
    data = await request.json()
    new_name = data.get("new_name", "").strip()
    if not new_name:
        return {"ok": False, "error": "New name is required"}
    if new_name == name:
        return {"ok": False, "error": "New name is the same as current name"}
    if not re.match(r'^[a-zA-Z0-9_\-]+$', new_name):
        return {"ok": False, "error": "Invalid name."}
    dataset_dir = ROOT / "data" / name
    new_dir = ROOT / "data" / new_name
    if not dataset_dir.exists():
        return {"ok": False, "error": "Dataset not found"}
    if new_dir.exists():
        return {"ok": False, "error": f"Dataset '{new_name}' already exists"}
    try:
        dataset_dir.rename(new_dir)
        return {"ok": True, "new_name": new_name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Push to Hugging Face ──

hf_push_state = {
    "running": False,
    "progress": 0,
    "status": "",
    "error": None,
    "logs": [],
}


def _run_hf_push(dataset_name: str, repo_id: str):
    state = hf_push_state
    state["running"] = True
    state["progress"] = 10
    state["status"] = "Preparing upload..."
    state["error"] = None
    state["logs"] = []

    dataset_dir = ROOT / "data" / dataset_name

    try:
        upload_files = []
        for f in sorted(dataset_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(dataset_dir)
            if rel.name == "augmentation.json":
                continue
            parts = rel.parts
            if parts[0].startswith("tmp"):
                continue
            upload_files.append((str(f), str(rel)))

        if not upload_files:
            state["error"] = "No files to upload"
            state["status"] = "Error: no files"
            return

        state["status"] = f"Uploading {len(upload_files)} files to {repo_id}..."
        state["progress"] = 20

        script = f"""
import sys, os
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi()
repo_id = {repr(repo_id)}
print(f"Creating/checking repo: {{repo_id}}", flush=True)
api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
print("Repo ready. Preparing commit...", flush=True)
ops = []
files = {upload_files!r}
for local, remote in files:
    ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local))
    print(f"ADD {{remote}}", flush=True)
print(f"Committing {{len(ops)}} files...", flush=True)
api.create_commit(
    repo_id=repo_id,
    repo_type="dataset",
    operations=ops,
    commit_message="Upload dataset via VLA-Benchmark",
)
# Tag with codebase version (required by lerobot-train)
import json
info_path = [local for local, remote in files if remote == "meta/info.json"]
if info_path:
    with open(info_path[0]) as f:
        info = json.load(f)
    version = info.get("codebase_version", "v2.1")
    try:
        try:
            api.delete_tag(repo_id, tag=version, repo_type="dataset")
        except Exception:
            pass
        api.create_tag(repo_id, tag=version, repo_type="dataset")
        print(f"TAG {{version}}", flush=True)
    except Exception as e:
        print(f"TAG_WARN: {{e}}", flush=True)
print("DONE", flush=True)
"""
        lerobot_python = "/opt/miniconda3/envs/lerobot/bin/python"
        if not Path(lerobot_python).exists():
            lerobot_python = "python3"

        sub_env = dict(os.environ)
        hf_token = _read_env().get("HF_TOKEN", "")
        if hf_token:
            sub_env["HF_TOKEN"] = hf_token

        proc = subprocess.Popen(
            [lerobot_python, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=sub_env,
        )

        total = len(upload_files)
        uploaded = 0
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            state["logs"].append(line)
            if len(state["logs"]) > 500:
                state["logs"] = state["logs"][-300:]
            if line.startswith("ADD "):
                uploaded += 1
                pct = 20 + int(uploaded / total * 70)
                state["progress"] = min(pct, 90)
                state["status"] = f"Uploading ({uploaded}/{total}): {line[4:]}"
            elif line == "DONE":
                state["progress"] = 100
                state["status"] = f"Pushed {total} files to {repo_id}"

        proc.wait()
        if proc.returncode != 0:
            state["error"] = f"Process exited with code {proc.returncode}"
            state["status"] = f"Error: exit code {proc.returncode}"
        else:
            state["progress"] = 100
            state["status"] = f"Done! Pushed to {repo_id}"

    except Exception as e:
        state["error"] = str(e)
        state["status"] = f"Error: {e}"
    finally:
        state["running"] = False


@app.get("/api/hf-dataset-tags")
async def hf_dataset_tags(repo_id: str):
    """Check if a HF dataset has a codebase version tag."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=_read_env().get("HF_TOKEN", "") or None)
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
        tags = [t.name for t in (refs.tags or [])]
        # Look for version tags like v2.1, v2.0, etc.
        version_tags = [t for t in tags if t.startswith("v")]
        if version_tags:
            return {"ok": True, "tagged": True, "version": version_tags[-1], "tags": tags}
        return {"ok": True, "tagged": False, "tags": tags}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/hf-dataset-tag")
async def hf_dataset_tag(request: Request):
    """Create a codebase version tag on a HF dataset."""
    body = await request.json()
    repo_id = body.get("repo_id", "")
    if not repo_id:
        return {"ok": False, "error": "No repo_id"}
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=_read_env().get("HF_TOKEN", "") or None)
        # Try to get version from the dataset's info.json
        version = "v2.1"
        try:
            import tempfile
            info_path = api.hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
            info = json.loads(Path(info_path).read_text())
            version = info.get("codebase_version", "v2.1")
        except Exception:
            pass
        try:
            api.delete_tag(repo_id, tag=version, repo_type="dataset")
        except Exception:
            pass
        api.create_tag(repo_id, tag=version, repo_type="dataset")
        return {"ok": True, "version": version}
    except Exception as e:
        err = str(e)
        if "already exists" in err.lower() or "Reference already exists" in err:
            # Tag already exists, that's fine
            return {"ok": True, "version": version, "note": "already exists"}
        return {"ok": False, "error": err}


@app.post("/api/datasets/{name}/push-hf")
async def push_to_hf(name: str, request: Request):
    if hf_push_state["running"]:
        return {"ok": False, "error": "Push already in progress"}
    dataset_dir = ROOT / "data" / name
    if not dataset_dir.exists() or not (dataset_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}
    body = await request.json()
    repo_id = body.get("repo_id", "").strip()
    if not repo_id or "/" not in repo_id:
        return {"ok": False, "error": "Invalid repo_id (format: username/dataset-name)"}
    threading.Thread(target=_run_hf_push, args=(name, repo_id), daemon=True).start()
    return {"ok": True, "message": f"Pushing to {repo_id}..."}


@app.get("/api/hf-push/status")
async def hf_push_status():
    return dict(hf_push_state)


# ── Download from Hugging Face ──

hf_download_state = {
    "running": False,
    "progress": 0,
    "status": "",
    "error": None,
    "logs": [],
}


def _run_hf_download(repo_id: str, local_name: str):
    state = hf_download_state
    state["running"] = True
    state["progress"] = 10
    state["status"] = f"Downloading {repo_id}..."
    state["error"] = None
    state["logs"] = []

    dataset_dir = ROOT / "data" / local_name

    try:
        script = f"""
import sys, os
from huggingface_hub import snapshot_download
repo_id = {repr(repo_id)}
local_dir = {repr(str(dataset_dir))}
print(f"Downloading {{repo_id}} to {{local_dir}}", flush=True)
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
)
print("DONE", flush=True)
"""
        lerobot_python = "/opt/miniconda3/envs/lerobot/bin/python"
        if not Path(lerobot_python).exists():
            lerobot_python = "python3"

        sub_env = dict(os.environ)
        hf_token = _read_env().get("HF_TOKEN", "")
        if hf_token:
            sub_env["HF_TOKEN"] = hf_token

        proc = subprocess.Popen(
            [lerobot_python, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=sub_env,
        )

        state["progress"] = 20
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            state["logs"].append(line)
            if len(state["logs"]) > 500:
                state["logs"] = state["logs"][-300:]
            if "Downloading" in line or "Fetching" in line:
                state["status"] = line[:120]
                state["progress"] = min(state["progress"] + 2, 85)
            elif line == "DONE":
                state["progress"] = 100
                state["status"] = f"Downloaded {repo_id} to data/{local_name}"

        proc.wait()
        if proc.returncode != 0:
            state["error"] = f"Process exited with code {proc.returncode}"
            state["status"] = f"Error: exit code {proc.returncode}"
        else:
            state["progress"] = 100
            state["status"] = f"Done! Downloaded to data/{local_name}"

    except Exception as e:
        state["error"] = str(e)
        state["status"] = f"Error: {e}"
    finally:
        state["running"] = False


@app.post("/api/datasets/download-hf")
async def download_from_hf(request: Request):
    if hf_download_state["running"]:
        return {"ok": False, "error": "Download already in progress"}
    body = await request.json()
    repo_id = body.get("repo_id", "").strip()
    if not repo_id or "/" not in repo_id:
        return {"ok": False, "error": "Invalid repo_id (format: username/dataset-name)"}

    local_name = repo_id.split("/", 1)[1].replace(" ", "-")

    dataset_dir = ROOT / "data" / local_name
    if dataset_dir.exists() and (dataset_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": f"Dataset '{local_name}' already exists locally."}

    threading.Thread(target=_run_hf_download, args=(repo_id, local_name), daemon=True).start()
    return {"ok": True, "message": f"Downloading {repo_id}...", "local_name": local_name}


@app.get("/api/hf-download/status")
async def hf_download_status():
    return dict(hf_download_state)


# ══════════ Additional Dataset API (from LeRobot Dataset Manager) ══════════

@app.get("/data/{path:path}")
async def serve_data_file(path: str):
    """Serve any file under DATA_DIR (videos, parquet, etc.)."""
    file_path = DATA_DIR / path
    if not file_path.exists() or not file_path.is_file():
        return {"ok": False, "error": "File not found"}
    return FileResponse(str(file_path))


@app.get("/api/data-dir")
async def get_data_dir():
    """Return the configured data directory path."""
    return {"ok": True, "data_dir": str(DATA_DIR)}


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------
@app.get("/api/datasets/{name}/segments/{ep}")
async def get_segments(name: str, ep: int):
    """Get saved segments for an episode."""
    seg_path = DATA_DIR / name / "meta" / "segments.json"
    if seg_path.exists():
        try:
            data = json.loads(seg_path.read_text())
            segs = data.get(str(ep))
            if segs:
                return {"ok": True, "segments": segs}
        except Exception:
            pass
    return {"ok": True, "segments": None}


@app.post("/api/datasets/{name}/segments/{ep}")
async def save_segments(name: str, ep: int, request: Request):
    """Save segments for an episode."""
    body = await request.json()
    segments = body.get("segments", [])
    seg_path = DATA_DIR / name / "meta" / "segments.json"
    data = {}
    if seg_path.exists():
        try:
            data = json.loads(seg_path.read_text())
        except Exception:
            pass
    data[str(ep)] = segments
    seg_path.write_text(json.dumps(data, indent=2))
    return {"ok": True}


# ---------------------------------------------------------------------------
# Delete episodes
# ---------------------------------------------------------------------------
@app.post("/api/datasets/{name}/delete-episodes")
async def delete_episodes(name: str, request: Request):
    """Delete selected episodes and re-index."""
    body = await request.json()
    episodes_to_delete = set(body.get("episodes", []))
    if not episodes_to_delete:
        return {"ok": False, "error": "No episodes specified"}

    dataset_dir = DATA_DIR / name
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}

    try:
        info = json.loads(info_path.read_text())
        original_total = info.get("total_episodes", 0)

        # 1. Filter episode metadata
        ep_dir = dataset_dir / "meta" / "episodes"
        all_ep_rows = []
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    if d["episode_index"][i] not in episodes_to_delete:
                        all_ep_rows.append({k: v[i] for k, v in d.items()})

        remaining_old = sorted(set(r["episode_index"] for r in all_ep_rows))
        ep_remap = {old: new for new, old in enumerate(remaining_old)}
        for row in all_ep_rows:
            row["episode_index"] = ep_remap[row["episode_index"]]

        # 2. Filter data
        data_dir = dataset_dir / "data"
        all_data_rows = []
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    if d["episode_index"][i] not in episodes_to_delete:
                        row = {}
                        for k, v in d.items():
                            val = v[i]
                            if hasattr(val, "tolist"):
                                val = val.tolist()
                            row[k] = val
                        all_data_rows.append(row)

        for idx, row in enumerate(all_data_rows):
            row["episode_index"] = ep_remap[row["episode_index"]]
            row["index"] = idx

        # 3. Write filtered episode metadata
        if ep_dir.exists():
            shutil.rmtree(ep_dir)
        ep_out = ep_dir / "chunk-000"
        ep_out.mkdir(parents=True, exist_ok=True)
        if all_ep_rows:
            cols = {k: [r[k] for r in all_ep_rows] for k in all_ep_rows[0]}
            pq.write_table(pa.table(cols), str(ep_out / "file-000.parquet"))

        # 4. Write filtered data
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_out = data_dir / "chunk-000"
        data_out.mkdir(parents=True, exist_ok=True)
        if all_data_rows:
            cols = {k: [r[k] for r in all_data_rows] for k in all_data_rows[0]}
            pq.write_table(pa.table(cols), str(data_out / "file-000.parquet"))

        # 5. Update info.json
        new_total_episodes = len(remaining_old)
        info["total_episodes"] = new_total_episodes
        info["total_frames"] = len(all_data_rows)
        info["splits"] = {"train": f"0:{new_total_episodes}"}
        info_path.write_text(json.dumps(info, indent=4))

        # 6. Remove stale stats
        stats_path = dataset_dir / "meta" / "stats.json"
        if stats_path.exists():
            stats_path.unlink()

        deleted_count = original_total - new_total_episodes
        return {
            "ok": True,
            "deleted": deleted_count,
            "remaining_episodes": new_total_episodes,
            "remaining_frames": len(all_data_rows),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Import dataset (copy from external path into ./data)
# ---------------------------------------------------------------------------
@app.post("/api/datasets/import")
async def import_dataset(request: Request):
    """Import a dataset by copying from an external folder into ./data."""
    body = await request.json()
    source_path = body.get("path", "").strip()
    custom_name = body.get("name", "").strip()

    if not source_path:
        return {"ok": False, "error": "Path is required"}

    src = Path(source_path).resolve()
    if not src.exists():
        return {"ok": False, "error": f"Path does not exist: {src}"}
    if not src.is_dir():
        return {"ok": False, "error": "Path must be a directory"}

    # Check if it's a single dataset (has meta/info.json) or a folder of datasets
    datasets_to_import = []
    if (src / "meta" / "info.json").exists():
        # Single dataset
        name = custom_name or src.name
        datasets_to_import.append({"src": src, "name": name})
    else:
        # Folder containing multiple datasets
        for child in sorted(src.iterdir()):
            if child.is_dir() and (child / "meta" / "info.json").exists():
                datasets_to_import.append({"src": child, "name": child.name})

    if not datasets_to_import:
        return {"ok": False, "error": "No valid LeRobot datasets found (requires meta/info.json)"}

    if import_status["running"]:
        return {"ok": False, "error": "An import is already in progress"}

    # Run copy in background thread
    import_status.update({"running": True, "progress": 0, "status": "Starting...", "error": None, "name": None})

    def do_import():
        try:
            total = len(datasets_to_import)
            imported = []
            skipped = []
            for i, ds in enumerate(datasets_to_import):
                name = ds["name"]
                # Sanitize name
                name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
                dst = DATA_DIR / name
                import_status["status"] = f"Copying {name} ({i+1}/{total})..."
                import_status["progress"] = int((i / total) * 100)

                if dst.exists():
                    skipped.append(name)
                    continue

                shutil.copytree(str(ds["src"]), str(dst))
                imported.append(name)

            import_status["progress"] = 100
            parts = []
            if imported:
                parts.append(f"Imported {len(imported)} dataset(s): {', '.join(imported)}")
            if skipped:
                parts.append(f"Skipped {len(skipped)} (already exist): {', '.join(skipped)}")
            import_status["status"] = ". ".join(parts) if parts else "Nothing to import"
            import_status["name"] = imported[0] if len(imported) == 1 else None
        except Exception as e:
            import_status["error"] = str(e)
            import_status["status"] = f"Error: {e}"
        finally:
            import_status["running"] = False

    threading.Thread(target=do_import, daemon=True).start()
    return {"ok": True, "count": len(datasets_to_import)}


@app.get("/api/import/status")
async def get_import_status():
    """Get current import progress."""
    return import_status


# ---------------------------------------------------------------------------
# Browse filesystem (for import dialog)
# ---------------------------------------------------------------------------
@app.get("/api/browse")
async def browse_filesystem(path: str = "~"):
    """List directories for the import file browser."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {"ok": False, "error": "Path does not exist"}
    if not target.is_dir():
        target = target.parent

    entries = []
    try:
        for child in sorted(target.iterdir()):
            if child.name.startswith('.'):
                continue
            if child.is_dir():
                is_dataset = (child / "meta" / "info.json").exists()
                has_datasets = False
                if not is_dataset:
                    try:
                        has_datasets = any(
                            (gc / "meta" / "info.json").exists()
                            for gc in child.iterdir() if gc.is_dir()
                        )
                    except PermissionError:
                        pass
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "is_dataset": is_dataset,
                    "has_datasets": has_datasets,
                })
    except PermissionError:
        return {"ok": False, "error": "Permission denied"}

    # Check if current directory itself is a dataset
    current_is_dataset = (target / "meta" / "info.json").exists()

    return {
        "ok": True,
        "current": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "is_dataset": current_is_dataset,
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Upload dataset (drag & drop from browser)
# ---------------------------------------------------------------------------
@app.post("/api/datasets/upload")
async def upload_dataset(
    folder_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Receive files from browser drag & drop and save as a dataset."""
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', folder_name)
    dst = DATA_DIR / name
    if dst.exists():
        return {"ok": False, "error": f"'{name}' already exists"}

    try:
        for f in files:
            # f.filename contains the relative path (e.g. "meta/info.json")
            rel = f.filename or ""
            file_dst = dst / rel
            file_dst.parent.mkdir(parents=True, exist_ok=True)
            content = await f.read()
            file_dst.write_bytes(content)

        # Verify it's a valid dataset
        if (dst / "meta" / "info.json").exists():
            return {"ok": True, "name": name}

        # Check if sub-datasets were uploaded
        sub = [d.name for d in dst.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists()]
        if sub:
            return {"ok": True, "name": name, "sub_datasets": sub}

        # Not valid — clean up
        shutil.rmtree(dst)
        return {"ok": False, "error": "No valid LeRobot dataset found (requires meta/info.json)"}
    except Exception as e:
        if dst.exists():
            shutil.rmtree(dst)
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Remove idle frames
# ---------------------------------------------------------------------------
def _auto_detect_segments(frames):
    """Auto-detect idle/movement segments using velocity-based method (mirrors frontend)."""
    n = len(frames)
    if n < 5:
        return [{"start": 0, "end": n - 1, "phase": "movement"}]

    # Compute velocity (sum of absolute action diffs)
    action_key = "action"
    action_len = len(frames[0].get(action_key, [])) if frames[0].get(action_key) else 6
    vel = [0.0]
    for i in range(1, n):
        v = 0.0
        a_cur = frames[i].get(action_key, [])
        a_prev = frames[i - 1].get(action_key, [])
        for j in range(min(action_len, len(a_cur), len(a_prev))):
            v += abs((a_cur[j] if j < len(a_cur) else 0) - (a_prev[j] if j < len(a_prev) else 0))
        vel.append(v)

    # Smooth
    hw = 5
    sv = []
    for i in range(n):
        lo, hi = max(0, i - hw), min(n - 1, i + hw)
        sv.append(sum(vel[lo:hi + 1]) / (hi - lo + 1))

    sorted_sv = sorted(sv)
    vel_thresh = max(sorted_sv[int(n * 0.85)] * 0.12, 0.3)
    labels = ["movement" if v > vel_thresh else "idle" for v in sv]

    # Build segments
    segs = []
    seg_start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[seg_start]:
            segs.append({"start": seg_start, "end": i - 1, "phase": labels[seg_start]})
            seg_start = i

    # Merge tiny segments
    for _ in range(3):
        i = len(segs) - 1
        while i >= 0:
            if segs[i]["end"] - segs[i]["start"] < 4 and len(segs) > 1:
                if i > 0:
                    segs[i - 1]["end"] = segs[i]["end"]
                else:
                    segs[1]["start"] = segs[0]["start"]
                segs.pop(i)
            i -= 1

    # Idle only at start/end — convert middle idle segments to movement
    for i in range(len(segs)):
        if segs[i]["phase"] == "idle" and i > 0 and i < len(segs) - 1:
            segs[i]["phase"] = "movement"
            segs[i]["label"] = "movement"
    # Merge consecutive movement segments
    merged = [segs[0]]
    for s in segs[1:]:
        if s["phase"] == merged[-1]["phase"]:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)

    return merged


@app.post("/api/datasets/{name}/remove-idle")
async def remove_idle_dataset(name: str, request: Request):
    """Create new dataset with idle frames removed from start/end of each episode."""
    body = await request.json()
    custom_name = body.get("new_name", "").strip().replace(" ", "-").replace("/", "-")
    mode = body.get("mode", "start")  # "start", "end", or "both"

    src_dir = DATA_DIR / name
    new_name = custom_name if custom_name else name + "_removeidle"
    new_dir = DATA_DIR / new_name

    if not (src_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}
    if new_dir.exists():
        return {"ok": False, "error": f"'{new_name}' already exists"}

    # Load saved segments
    seg_path = src_dir / "meta" / "segments.json"
    saved_segments = {}
    if seg_path.exists():
        try:
            saved_segments = json.loads(seg_path.read_text())
        except Exception:
            pass

    try:
        src_info = json.loads((src_dir / "meta" / "info.json").read_text())
        all_tasks = []
        tp = src_dir / "meta" / "tasks.parquet"
        if tp.exists():
            try:
                all_tasks = pq.read_table(str(tp)).to_pydict().get("task", [])
            except Exception:
                pass

        # Read episode metadata
        ep_meta_list = []
        ep_dir = src_dir / "meta" / "episodes"
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep_meta_list.append({k: v[i] for k, v in d.items()})
        ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

        # Read data grouped by episode
        src_frames = {}
        data_dir = src_dir / "data"
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep = d["episode_index"][i]
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    src_frames.setdefault(ep, []).append(row)
        for ep in src_frames:
            src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

        combined_data, combined_ep = [], []
        new_ep_idx, global_frame_idx, total_removed = 0, 0, 0

        for old_ep in range(src_info.get("total_episodes", len(src_frames))):
            frames = src_frames.get(old_ep, [])
            if not frames:
                continue

            # Use saved segments or auto-detect to find movement frames
            ep_segs = saved_segments.get(str(old_ep))
            if not ep_segs:
                # Auto-detect segments (same algorithm as frontend)
                ep_segs = _auto_detect_segments(frames)

            # Determine trim range based on mode
            # Find first and last movement segment indices
            first_move = None
            last_move = None
            for si, seg in enumerate(ep_segs):
                if seg.get("phase") == "movement":
                    if first_move is None:
                        first_move = si
                    last_move = si

            if first_move is None:
                # No movement at all — skip episode
                total_removed += len(frames)
                continue

            # Build keep ranges: always keep everything between first and last movement
            # (middle idle segments are never removed)
            if mode == "start":
                # Remove idle before first movement only
                trim_start = ep_segs[first_move]["start"]
                trim_end = len(frames) - 1
            elif mode == "end":
                # Remove idle after last movement only
                trim_start = 0
                trim_end = ep_segs[last_move]["end"]
            else:  # "both"
                trim_start = ep_segs[first_move]["start"]
                trim_end = ep_segs[last_move]["end"]

            trimmed = frames[trim_start:trim_end + 1]
            total_removed += len(frames) - len(trimmed)
            if not trimmed:
                continue

            ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)
            ep_start = global_frame_idx
            t0 = trimmed[0].get("timestamp", 0.0)

            for fi, row in enumerate(trimmed):
                new_row = dict(row)
                new_row["episode_index"] = new_ep_idx
                new_row["frame_index"] = fi
                new_row["index"] = global_frame_idx
                new_row["timestamp"] = row.get("timestamp", 0.0) - t0
                combined_data.append(new_row)
                global_frame_idx += 1

            new_meta = {"episode_index": new_ep_idx, "length": len(trimmed),
                        "data/chunk_index": 0, "data/file_index": 0,
                        "dataset_from_index": ep_start, "dataset_to_index": global_frame_idx}
            if ep_meta and "tasks" in ep_meta:
                new_meta["tasks"] = ep_meta["tasks"]
            elif all_tasks:
                new_meta["tasks"] = [all_tasks[0]]

            # Copy video references with adjusted timestamps
            if ep_meta:
                cam_keys = set()
                for k in ep_meta:
                    if k.startswith("videos/"):
                        parts = k.split("/")
                        cam_name = "/".join(parts[1:-1])
                        if cam_name:
                            cam_keys.add(cam_name)
                for cam in cam_keys:
                    ck = f"videos/{cam}"
                    orig_chunk = ep_meta.get(f"{ck}/chunk_index", 0)
                    orig_file = ep_meta.get(f"{ck}/file_index", 0)
                    src_vid = src_dir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                    new_meta[f"{ck}/from_timestamp"] = ep_meta.get(f"{ck}/from_timestamp", 0.0) + t0
                    new_meta[f"{ck}/to_timestamp"] = ep_meta.get(f"{ck}/to_timestamp", 0.0)
                    new_meta[f"{ck}/chunk_index"] = 0
                    new_meta[f"{ck}/file_index"] = 0
                    new_meta[f"_src_vid_{cam}"] = str(src_vid) if src_vid.exists() else None

            combined_ep.append(new_meta)
            new_ep_idx += 1

        if not combined_data:
            return {"ok": False, "error": "No data after removing idle"}

        # Write output
        new_dir.mkdir(parents=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True)

        # Copy videos
        all_cams = set()
        for em in combined_ep:
            for k in list(em.keys()):
                if k.startswith("_src_vid_"):
                    all_cams.add(k[len("_src_vid_"):])
        for cam in all_cams:
            vid_dir = new_dir / "videos" / cam / "chunk-000"
            vid_dir.mkdir(parents=True)
            src_vids, src_set = [], set()
            for em in combined_ep:
                sv = em.get(f"_src_vid_{cam}")
                if sv and sv not in src_set:
                    src_set.add(sv)
                    src_vids.append(sv)
            vid_map = {}
            for fi, sv in enumerate(src_vids):
                shutil.copy2(sv, str(vid_dir / f"file-{fi:03d}.mp4"))
                vid_map[sv] = fi
            for em in combined_ep:
                sv = em.pop(f"_src_vid_{cam}", None)
                if sv and sv in vid_map:
                    em[f"videos/{cam}/file_index"] = vid_map[sv]

        for em in combined_ep:
            for k in [k for k in em if k.startswith("_")]:
                del em[k]

        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))
        cols = {k: [r.get(k) for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))
        if all_tasks:
            import pandas as pd
            tasks_df = pd.DataFrame({"task_index": list(range(len(all_tasks)))}, index=pd.Index(all_tasks, name="task"))
            tasks_df.to_parquet(str(new_dir / "meta" / "tasks.parquet"))

        new_info = dict(src_info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

        return {"ok": True, "name": new_name, "episodes": new_ep_idx,
                "frames": len(combined_data), "removed_frames": total_removed}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

_INTENSITY_SCALE = {1: 0.25, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0}


def _augp_camera(opts, intensity=3, fixed=None):
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
        p["persp"] = np.random.uniform(-0.05 * s, 0.05 * s, (4, 2))
    if opts.get("affine"):
        p["tx"] = random.uniform(-0.04 * s, 0.04 * s)
        p["ty"] = random.uniform(-0.04 * s, 0.04 * s)
        p["shear"] = random.uniform(-0.03 * s, 0.03 * s)
        p["scale"] = random.uniform(1 - 0.05 * s, 1 + 0.05 * s)
    if opts.get("rotation"):
        p["angle"] = random.uniform(-4 * s, 4 * s)
    return p or None


def _augp_light(opts, intensity=3, fixed=None):
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if fixed:
        if opts.get("brightness") and fixed.get("brightness") is not None:
            p["brightness"] = float(fixed["brightness"])
        if opts.get("contrast") and fixed.get("contrast") is not None:
            p["contrast"] = float(fixed["contrast"])
        if opts.get("saturation") and fixed.get("saturation") is not None:
            p["saturation"] = float(fixed["saturation"])
        if opts.get("color_jitter") and fixed.get("hue") is not None:
            p["hue"] = int(fixed["hue"])
            p["sat_off"] = int(fixed.get("sat_off", 0))
            p["val_off"] = int(fixed.get("val_off", 0))
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
        p["brightness"] = random.uniform(1 - 0.25 * s, 1 + 0.25 * s)
    if opts.get("contrast"):
        p["contrast"] = random.uniform(1 - 0.25 * s, 1 + 0.25 * s)
    if opts.get("saturation"):
        p["saturation"] = random.uniform(1 - 0.35 * s, 1 + 0.35 * s)
    if opts.get("color_jitter"):
        h = int(8 * s)
        sv = int(15 * s)
        p["hue"] = random.randint(-h, h)
        p["sat_off"] = random.randint(-sv, sv)
        p["val_off"] = random.randint(-sv, sv)
    if opts.get("shadow"):
        p["shadow_dir"] = random.choice(["left", "right", "top", "bottom"])
        p["shadow_a"] = random.uniform(0.15 * s, 0.4 * s)
    if opts.get("noise"):
        p["noise_s"] = random.uniform(4 * s, 12 * s)
    if opts.get("blur"):
        p["blur_k"] = random.choice([3, 5] if s <= 1.0 else [3, 5, 7])
        p["blur_s"] = random.uniform(0.4 * s, 1.2 * s)
    return p or None


def _augp_robot(opts, n_frames, fps, intensity=3, fixed=None):
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
        p["trim"] = random.randint(1, max(1, int(n_frames * 0.12 * s)))
    if opts.get("initial_noise"):
        p["joint_offset"] = [random.gauss(0, 1.5 * s) for _ in range(5)] + [0.0]
    if opts.get("trajectory_jitter"):
        p["jitter_s"] = random.uniform(0.3 * s, 1.0 * s)
    p["fps"] = fps
    return p or None


_PICK_SYNS = ["pick up", "grab", "lift", "take", "grasp"]
_PLACE_SYNS = ["place", "put", "set down", "drop", "lay"]
_PREP_SYNS = {"to": ["to", "into", "onto", "in", "on"],
              "from": ["from", "out of", "off"]}


def _aug_task_text(task, opts):
    if not opts or not opts.get("enabled"):
        return task
    result = task
    do_syn = opts.get("synonym") or opts.get("paraphrase")
    if do_syn:
        for verb, syns in [("pick up", _PICK_SYNS), ("place", _PLACE_SYNS)]:
            if verb in result.lower():
                rep = random.choice(syns)
                result = re.sub(re.escape(verb), rep, result, count=1, flags=re.I)
        for prep, syns in _PREP_SYNS.items():
            pat = rf"\b{prep}\b"
            if re.search(pat, result, re.I):
                result = re.sub(pat, random.choice(syns), result, count=1, flags=re.I)
    if opts.get("typo") and random.random() < 0.35:
        words = result.split()
        if len(words) > 2:
            idx = random.randint(1, len(words) - 1)
            w = list(words[idx])
            if len(w) > 2:
                pos = random.randint(0, len(w) - 2)
                op = random.choice(["swap", "del", "dup"])
                if op == "swap":
                    w[pos], w[pos + 1] = w[pos + 1], w[pos]
                elif op == "del":
                    w.pop(pos)
                else:
                    w.insert(pos, w[pos])
                words[idx] = "".join(w)
            result = " ".join(words)
    return result


def _aug_frame(frame, cam_p, light_p):
    import cv2
    h, w = frame.shape[:2]
    f = frame
    if cam_p:
        if "persp" in cam_p:
            pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            pts2 = (pts1 + cam_p["persp"] * [w, h]).astype(np.float32)
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
            noise = np.random.normal(0, light_p["noise_s"], ff.shape)
            ff = np.clip(ff.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if "blur_k" in light_p:
            ff = cv2.GaussianBlur(ff, (light_p["blur_k"], light_p["blur_k"]), light_p["blur_s"])
        f = ff
    return f


def _aug_episode_data(rows, robot_p):
    if not robot_p:
        return rows
    rows = [dict(r) for r in rows]
    trim = robot_p.get("trim", 0)
    if trim > 0 and trim < len(rows):
        rows = rows[trim:]
    fps = robot_p.get("fps", 30)
    offset = robot_p.get("joint_offset", None)
    jitter = robot_p.get("jitter_s", 0)
    for i, r in enumerate(rows):
        r["frame_index"] = i
        r["timestamp"] = i / fps
        action = list(r.get("action", []))
        state = list(r.get("observation.state", []))
        for j in range(min(len(action), 5)):
            if offset:
                action[j] += offset[j]
                state[j] += offset[j]
            if jitter > 0:
                jit = random.gauss(0, jitter)
                action[j] += jit
                state[j] += jit
        r["action"] = action
        r["observation.state"] = state
    return rows


def _aug_process_video(src_path, dst_path, from_ts, to_ts, fps, cam_p, light_p, vid_w, vid_h, expected_frames=None):
    duration = to_ts - from_ts
    if duration <= 0 or not Path(src_path).exists():
        return 0.0, 0.0, 0
    read_cmd = [
        "ffmpeg", "-v", "quiet",
        "-ss", str(from_ts), "-i", str(src_path),
        "-t", str(duration + 1.0 / fps),  # read slightly more to avoid missing last frame
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{vid_w}x{vid_h}", "pipe:1",
    ]
    write_cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{vid_w}x{vid_h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23", str(dst_path),
    ]
    try:
        reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE)
        writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE)
        frame_size = vid_w * vid_h * 3
        count = 0
        while True:
            if expected_frames and count >= expected_frames:
                break
            raw = reader.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(vid_h, vid_w, 3)
            if cam_p or light_p:
                frame = _aug_frame(frame, cam_p, light_p)
            writer.stdin.write(frame.tobytes())
            count += 1
        reader.stdout.close()
        writer.stdin.close()
        reader.wait()
        writer.wait()
        return 0.0, count / fps if fps > 0 else 0.0, count
    except Exception as e:
        print(f"[Augment] Video processing error: {e}")
        return 0.0, 0.0, 0


def _get_camera_keys(info):
    """Extract camera feature keys from dataset info."""
    cams = []
    for k, v in info.get("features", {}).items():
        if k.startswith("observation.images.") and isinstance(v, dict) and v.get("dtype") == "video":
            cams.append(k)
    return cams


def _run_augmentation(src_name, new_name, target_episodes, techniques, intensity=3, mode="random", fixed_params=None):
    state = augment_state
    state["running"] = True
    state["progress"] = 0
    state["status"] = "Loading source dataset..."
    state["error"] = None

    try:
        src_dir = DATA_DIR / src_name
        new_dir = DATA_DIR / new_name
        info = json.loads((src_dir / "meta" / "info.json").read_text())
        fps = info.get("fps", 30)
        cam_keys = _get_camera_keys(info)

        # Get video dimensions from first camera feature
        vid_h, vid_w = 240, 320
        if cam_keys:
            shape = info.get("features", {}).get(cam_keys[0], {}).get("shape", [240, 320, 3])
            vid_h, vid_w = shape[0], shape[1]

        # Read tasks
        src_tasks = []
        tp = src_dir / "meta" / "tasks.parquet"
        if tp.exists():
            try:
                src_tasks = pq.read_table(str(tp)).to_pydict().get("task", [])
            except Exception:
                pass

        # Read episode metadata
        ep_meta_list = []
        ep_dir = src_dir / "meta" / "episodes"
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep_meta_list.append({k: v[i] for k, v in d.items()})
        ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

        # Read all data frames by episode
        src_frames = {}
        data_dir = src_dir / "data"
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep = d["episode_index"][i]
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    src_frames.setdefault(ep, []).append(row)
        for ep in src_frames:
            src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

        n_src_eps = len(src_frames)
        if n_src_eps == 0:
            raise ValueError("Source dataset has no episodes")

        sorted_eps = sorted(src_frames.keys())
        if mode == "random":
            # Random: pick random source episode for each augmented copy
            ep_schedule = [random.choice(sorted_eps) for _ in range(target_episodes)]
        else:
            # Fix: evenly distribute copies across episodes
            ep_schedule = []
            base = target_episodes // n_src_eps
            remainder = target_episodes % n_src_eps
            for i, ep in enumerate(sorted_eps):
                n = base + (1 if i < remainder else 0)
                ep_schedule.extend([ep] * n)

        total_steps = target_episodes
        current_step = 0

        new_dir.mkdir(parents=True, exist_ok=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in cam_keys:
            (new_dir / "videos" / cam / "chunk-000").mkdir(parents=True, exist_ok=True)

        combined_data = []
        combined_ep = []
        aug_ep_info = []  # per-episode augmentation info
        all_new_tasks = list(src_tasks)
        task_set = set(src_tasks)
        new_ep_idx = 0
        global_frame_idx = 0

        cam_opts = techniques.get("camera_shift", {})
        light_opts = techniques.get("light_quality", {})
        robot_opts = techniques.get("robot_noise", {})
        lang_opts = techniques.get("language", {})
        cam_fixed = fixed_params.get("camera") if fixed_params else None
        light_fixed = fixed_params.get("light") if fixed_params else None
        robot_fixed = fixed_params.get("robot") if fixed_params else None

        seen_eps = set()
        for sched_idx, old_ep in enumerate(ep_schedule):
                is_original = old_ep not in seen_eps
                seen_eps.add(old_ep)
                frames = src_frames[old_ep]
                ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)

                state["status"] = f"Episode {new_ep_idx + 1}/{target_episodes} (src {old_ep})"
                current_step += 1
                state["progress"] = int(current_step / total_steps * 100)

                cam_p = None if is_original else (_augp_camera(cam_opts, intensity, cam_fixed) if cam_opts.get("enabled") else None)
                light_p = None if is_original else (_augp_light(light_opts, intensity, light_fixed) if light_opts.get("enabled") else None)
                robot_p = None if is_original else (
                    _augp_robot(robot_opts, len(frames), fps, intensity, robot_fixed) if robot_opts.get("enabled") else None
                )

                # Record per-episode augmentation info
                def _serialize_params(p):
                    if p is None: return None
                    out = {}
                    for k, v in p.items():
                        if hasattr(v, 'tolist'): out[k] = v.tolist()
                        else: out[k] = v
                    return out
                aug_ep_info.append({
                    "new_ep": new_ep_idx,
                    "source_ep": old_ep,
                    "is_original": is_original,
                    "camera": _serialize_params(cam_p),
                    "light": _serialize_params(light_p),
                    "robot": _serialize_params(robot_p),
                })

                aug_rows = _aug_episode_data(frames, robot_p) if not is_original else [dict(r) for r in frames]

                if is_original:
                    ep_task_list = ep_meta["tasks"] if ep_meta and "tasks" in ep_meta else src_tasks[:1]
                else:
                    orig_tasks = ep_meta["tasks"] if ep_meta and "tasks" in ep_meta else src_tasks[:1]
                    ep_task_list = [_aug_task_text(t, lang_opts) for t in orig_tasks]
                for t in ep_task_list:
                    if t not in task_set:
                        task_set.add(t)
                        all_new_tasks.append(t)
                task_to_idx = {t: i for i, t in enumerate(all_new_tasks)}

                ep_global_start = global_frame_idx
                for fi, row in enumerate(aug_rows):
                    nr = dict(row)
                    nr["episode_index"] = new_ep_idx
                    nr["frame_index"] = fi
                    nr["index"] = global_frame_idx
                    nr["timestamp"] = fi / fps
                    old_ti = row.get("task_index", 0)
                    if old_ti < len(ep_task_list):
                        nr["task_index"] = task_to_idx.get(ep_task_list[old_ti], 0)
                    elif ep_task_list:
                        nr["task_index"] = task_to_idx.get(ep_task_list[0], 0)
                    combined_data.append(nr)
                    global_frame_idx += 1

                new_meta = {
                    "episode_index": new_ep_idx,
                    "tasks": ep_task_list,
                    "length": len(aug_rows),
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": ep_global_start,
                    "dataset_to_index": global_frame_idx,
                }

                for cam in cam_keys:
                    cam_key = f"videos/{cam}"
                    orig_chunk = ep_meta.get(f"{cam_key}/chunk_index", 0) if ep_meta else 0
                    orig_file = ep_meta.get(f"{cam_key}/file_index", 0) if ep_meta else 0
                    orig_from = ep_meta.get(f"{cam_key}/from_timestamp", 0.0) if ep_meta else 0.0
                    orig_to = ep_meta.get(f"{cam_key}/to_timestamp", 0.0) if ep_meta else 0.0
                    src_vid = src_dir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                    dst_vid = new_dir / "videos" / cam / "chunk-000" / f"file-{new_ep_idx:03d}.mp4"
                    new_meta[f"{cam_key}/chunk_index"] = 0
                    new_meta[f"{cam_key}/file_index"] = new_ep_idx

                    if src_vid.exists():
                        new_from, new_to, _ = _aug_process_video(
                            src_vid, dst_vid, orig_from, orig_to, fps,
                            cam_p if not is_original else None,
                            light_p if not is_original else None,
                            vid_w, vid_h,
                            expected_frames=len(frames),
                        )
                    else:
                        new_from, new_to = 0.0, 0.0

                    trim = robot_p.get("trim", 0) if robot_p else 0
                    if trim > 0:
                        new_from += trim / fps
                    new_meta[f"{cam_key}/from_timestamp"] = new_from
                    new_meta[f"{cam_key}/to_timestamp"] = new_to

                if ep_meta:
                    for k, v in ep_meta.items():
                        if k.startswith("stats/"):
                            new_meta[k] = v
                new_meta["meta/episodes/chunk_index"] = 0
                new_meta["meta/episodes/file_index"] = 0

                combined_ep.append(new_meta)
                new_ep_idx += 1

        state["status"] = "Writing dataset..."
        state["progress"] = 95

        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))
        cols = {k: [r.get(k) for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))
        if all_new_tasks:
            import pandas as pd
            tasks_df = pd.DataFrame({"task_index": list(range(len(all_new_tasks)))}, index=pd.Index(all_new_tasks, name="task"))
            tasks_df.to_parquet(str(new_dir / "meta" / "tasks.parquet"))

        new_info = dict(info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["total_tasks"] = len(all_new_tasks)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        for feat_key in cam_keys:
            feat = new_info.get("features", {}).get(feat_key, {})
            if "info" in feat:
                feat["info"]["video.codec"] = "h264"
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

        # Compute stats (including camera keys)
        state["status"] = "Computing stats..."
        _compute_stats_json(combined_data, new_dir)

        # Save augmentation config to separate folder (keep dataset as pure LeRobot format)
        aug_meta = {
            "source_dataset": src_name,
            "target_episodes": target_episodes,
            "intensity": intensity,
            "mode": mode,
            "techniques": techniques,
            "fixed_params": fixed_params,
            "episodes": aug_ep_info,
        }
        (AUGMENT_CONFIG_DIR / f"{new_name}.json").write_text(json.dumps(aug_meta, indent=2))

        state["progress"] = 100
        state["status"] = f"Done! {new_ep_idx} episodes, {len(combined_data)} frames"
        print(f"[Augment] Created {new_name}: {new_ep_idx} eps, {len(combined_data)} frames")

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["error"] = str(e)
        state["status"] = f"Error: {e}"
    finally:
        state["running"] = False


@app.get("/api/datasets/{name}/augment-config")
async def get_augment_config(name: str):
    """Get augmentation config if this dataset was created by augmentation."""
    cfg_path = AUGMENT_CONFIG_DIR / f"{name}.json"
    if not cfg_path.exists():
        return {"ok": True, "is_augmented": False}
    try:
        cfg = json.loads(cfg_path.read_text())
        return {"ok": True, "is_augmented": True, "config": cfg}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/datasets/{name}/augment-preview")
async def augment_preview(name: str, request: Request):
    """Generate preview of augmentation: original vs augmented first frames."""
    import cv2
    import base64

    body = await request.json()
    intensity = body.get("intensity", 3)
    techniques = body.get("techniques", {})
    num_previews = body.get("count", 5)

    src_dir = DATA_DIR / name
    if not (src_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}

    info = json.loads((src_dir / "meta" / "info.json").read_text())
    cam_keys = _get_camera_keys(info)
    if not cam_keys:
        return {"ok": False, "error": "No camera keys found"}

    total_eps = info.get("total_episodes", 0)
    if total_eps == 0:
        return {"ok": False, "error": "No episodes"}

    # Read episode metadata to find video files
    ep_meta_list = []
    ep_dir = src_dir / "meta" / "episodes"
    if ep_dir.exists():
        for pf in sorted(ep_dir.rglob("*.parquet")):
            try:
                tbl = pq.read_table(str(pf))
            except Exception:
                continue
            d = tbl.to_pydict()
            for i in range(len(d.get("episode_index", []))):
                ep_meta_list.append({k: v[i] for k, v in d.items()})
    ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

    # Pick random episodes
    indices = random.sample(range(len(ep_meta_list)), min(num_previews, len(ep_meta_list)))

    cam_opts = techniques.get("camera", {})
    light_opts = techniques.get("light", {})
    mode = body.get("mode", "random")
    fixed_params = body.get("fixed_params") if mode == "fix" else None
    cam_fixed = fixed_params.get("camera") if fixed_params else None
    light_fixed = fixed_params.get("light") if fixed_params else None
    first_cam = cam_keys[0]

    previews = []
    for idx in indices:
        em = ep_meta_list[idx]
        ck = f"videos/{first_cam}"
        chunk = em.get(f"{ck}/chunk_index", 0)
        fi = em.get(f"{ck}/file_index", 0)
        from_ts = em.get(f"{ck}/from_timestamp", 0.0)
        vid_path = src_dir / "videos" / first_cam / f"chunk-{chunk:03d}" / f"file-{fi:03d}.mp4"
        if not vid_path.exists():
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if from_ts > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, from_ts * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            continue

        # Resize for preview (max 160px wide)
        h, w = frame.shape[:2]
        scale = min(160 / w, 120 / h)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Encode original
        _, buf_orig = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_orig = base64.b64encode(buf_orig).decode()

        # Apply augmentation
        cam_p = _augp_camera(cam_opts, intensity, cam_fixed) if cam_opts.get("enabled") else None
        light_p = _augp_light(light_opts, intensity, light_fixed) if light_opts.get("enabled") else None
        aug_frame = _aug_frame(small, cam_p, light_p)

        _, buf_aug = cv2.imencode(".jpg", aug_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_aug = base64.b64encode(buf_aug).decode()

        previews.append({
            "ep": em.get("episode_index", idx),
            "original": b64_orig,
            "augmented": b64_aug,
        })

    return {"ok": True, "previews": previews}


@app.post("/api/datasets/{name}/augment")
async def start_augment(name: str, request: Request):
    if augment_state["running"]:
        return {"ok": False, "error": "Augmentation already running"}
    body = await request.json()
    new_name = body.get("name", "").strip().replace(" ", "-").replace("/", "-")
    target_episodes = body.get("target_episodes")
    intensity = body.get("intensity", 3)
    techniques = body.get("techniques", {})
    if not new_name:
        return {"ok": False, "error": "Enter a dataset name"}
    src_dir = DATA_DIR / name
    if not (src_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Source dataset not found"}
    if (DATA_DIR / new_name).exists():
        return {"ok": False, "error": f"Dataset '{new_name}' already exists"}
    if not target_episodes or target_episodes < 1:
        return {"ok": False, "error": "Invalid target episodes"}
    if intensity not in (1, 2, 3, 4, 5):
        intensity = 3
    mode = body.get("mode", "random")
    fixed_params = body.get("fixed_params") if mode == "fix" else None
    threading.Thread(
        target=_run_augmentation,
        args=(name, new_name, target_episodes, techniques, intensity, mode, fixed_params),
        daemon=True,
    ).start()
    return {"ok": True, "message": "Augmentation started"}


@app.get("/api/augment/status")
async def augment_status():
    return {
        "running": augment_state["running"],
        "progress": augment_state["progress"],
        "status": augment_state["status"],
        "error": augment_state["error"],
    }


# ══════════ Data Collection API ══════════

datacollect_state = {
    "process": None,
    "running": False,
    "started_at": None,
    "log_lines": [],
    "log_lock": threading.Lock(),
    "completed": False,
}


def _datacollect_reader(proc, state):
    buf = ""
    try:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            c = ch.decode("utf-8", errors="replace")
            if c == "\n":
                if buf.strip():
                    with state["log_lock"]:
                        state["log_lines"].append(buf)
                        if len(state["log_lines"]) > 500:
                            state["log_lines"] = state["log_lines"][-500:]
                    print(f"[DataCollect] {buf}")
                buf = ""
            else:
                buf += c
                if "press enter" in buf.lower() and buf.rstrip().endswith(":"):
                    with state["log_lock"]:
                        state["log_lines"].append(buf)
                    try:
                        proc.stdin.write(b"\n")
                        proc.stdin.flush()
                    except Exception:
                        pass
                    buf = ""
    except Exception:
        pass
    if buf.strip():
        with state["log_lock"]:
            state["log_lines"].append(buf)


@app.post("/api/datacollect/verify")
async def datacollect_verify():
    import shutil

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    teleop_cfg = cfg.get("teleop", {})
    checks = []

    robot_port = robot_cfg.get("port", "")
    robot_ok = os.path.exists(robot_port)
    checks.append({"name": "Robot port", "detail": robot_port, "ok": robot_ok,
                    "error": "" if robot_ok else f"Port not found: {robot_port}"})

    teleop_port = teleop_cfg.get("port", "")
    teleop_ok = os.path.exists(teleop_port)
    checks.append({"name": "Teleop port", "detail": teleop_port, "ok": teleop_ok,
                    "error": "" if teleop_ok else f"Port not found: {teleop_port}"})

    import cv2
    cams = robot_cfg.get("cameras", {})
    cam_frames = []
    for cam_name, cam_cfg in cams.items():
        idx = cam_cfg.get("index", 0)
        # Try to use already-running camera thread first
        st = cam_state.get(cam_name)
        if st and st.get("running"):
            with st["lock"]:
                frame = st["frame"].copy() if st["frame"] is not None else None
            cam_ok = frame is not None
            checks.append({"name": f"Camera: {cam_name}", "detail": f"index {idx}", "ok": cam_ok,
                            "error": "" if cam_ok else f"Cannot read from camera index {idx}"})
            if cam_ok:
                cv2.putText(frame, f"{idx} ({cam_name})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                cam_frames.append(frame)
        else:
            # No active thread — open camera directly to test
            cap = cv2.VideoCapture(idx)
            cam_ok = False
            frame = None
            if cap.isOpened():
                ret, frame = cap.read()
                cam_ok = ret
            cap.release()
            checks.append({"name": f"Camera: {cam_name}", "detail": f"index {idx}", "ok": cam_ok,
                            "error": "" if cam_ok else f"Cannot read from camera index {idx}"})
            if cam_ok and frame is not None:
                cv2.putText(frame, f"{idx} ({cam_name})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                cam_frames.append(frame)

    camera_preview = None
    if cam_frames:
        target_h = max(f.shape[0] for f in cam_frames)
        resized = []
        for f in cam_frames:
            if f.shape[0] != target_h:
                ratio = target_h / f.shape[0]
                f = cv2.resize(f, (int(f.shape[1] * ratio), target_h))
            resized.append(f)
        stacked = np.hstack(resized)
        _, buf = cv2.imencode(".jpg", stacked, [cv2.IMWRITE_JPEG_QUALITY, 85])
        camera_preview = base64.b64encode(buf.tobytes()).decode()

    stat = shutil.disk_usage(str(ROOT))
    free_gb = stat.free / (1024 ** 3)
    disk_ok = free_gb > 1.0
    checks.append({"name": "Disk space", "detail": f"{free_gb:.1f} GB free", "ok": disk_ok,
                    "error": "" if disk_ok else "Less than 1 GB free"})

    lerobot_bin = "/opt/miniconda3/envs/lerobot/bin/lerobot-record"
    bin_ok = os.path.exists(lerobot_bin)
    checks.append({"name": "lerobot-record", "detail": lerobot_bin, "ok": bin_ok,
                    "error": "" if bin_ok else f"Binary not found: {lerobot_bin}"})

    all_ok = all(c["ok"] for c in checks)
    result = {"ok": all_ok, "checks": checks}
    if camera_preview:
        result["camera_preview"] = camera_preview
    return result


def _build_datacollect_cmd() -> list[str]:
    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    teleop_cfg = cfg.get("teleop", {})
    dc_cfg = cfg.get("data_collection", {})
    cams = robot_cfg.get("cameras", {})
    dc_cams = dc_cfg.get("cameras", {})

    cam_parts = []
    for cam_name, cam_c in cams.items():
        dc_cam = dc_cams.get(cam_name, {})
        cam_fps = dc_cam.get("fps", 30)
        cam_w = dc_cam.get("width", 320)
        cam_h = dc_cam.get("height", 240)
        part = (f"{cam_name}: {{type: opencv, index_or_path: {cam_c.get('index', 0)}, "
                f"width: {cam_w}, height: {cam_h}, fps: {cam_fps}}}")
        cam_parts.append(part)
    cameras_str = "{" + ", ".join(cam_parts) + "}"
    fps = dc_cams.get("top", {}).get("fps", 30)

    return [
        "/opt/miniconda3/envs/lerobot/bin/lerobot-record",
        f"--robot.type=so101_follower",
        f"--robot.port={robot_cfg.get('port', '')}",
        f"--robot.id={robot_cfg.get('id', 'my_awesome_follower_arm')}",
        f"--robot.cameras={cameras_str}",
        f"--teleop.type={teleop_cfg.get('type', 'so101_leader')}",
        f"--teleop.port={teleop_cfg.get('port', '')}",
        f"--teleop.id={teleop_cfg.get('id', 'my_awesome_leader_arm')}",
        f"--display_data=false",
        f"--dataset.repo_id={dc_cfg.get('repo_id', 'user/my-dataset')}",
        f"--dataset.root={dc_cfg.get('dataset_root', './data/dataset')}",
        f"--dataset.push_to_hub={'true' if dc_cfg.get('push_to_hub', False) else 'false'}",
        f"--dataset.num_episodes={dc_cfg.get('num_episodes', 50)}",
        f"--dataset.single_task={dc_cfg.get('task', 'default task')}",
        f"--dataset.streaming_encoding=true",
        f"--dataset.fps={fps}",
        f"--dataset.encoder_threads={dc_cfg.get('encoder_threads', 4)}",
    ]


@app.get("/api/datacollect/command")
async def datacollect_command():
    cmd = _build_datacollect_cmd()
    cfg = load_config()
    dc_cfg = cfg.get("data_collection", {})
    dataset_root = Path(dc_cfg.get("dataset_root", "./data/dataset"))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset_exists = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()
    if dc_cfg.get("resume", False) and dataset_exists:
        cmd.append("--resume=true")
    import shlex
    quoted = [shlex.quote(arg) if any(c in arg for c in ' {}') else arg for arg in cmd]
    return {"ok": True, "command": " \\\n  ".join(quoted)}


@app.post("/api/datacollect/start")
async def datacollect_start():
    if datacollect_state["running"]:
        return {"ok": False, "error": "Already recording"}
    if teleop_state["running"]:
        await teleop_stop()
    if robot_state["connected"]:
        disconnect_robot()
    time.sleep(1)

    cmd = _build_datacollect_cmd()

    import shutil
    cfg = load_config()
    dc_cfg = cfg.get("data_collection", {})
    dataset_root = Path(dc_cfg.get("dataset_root", "./data/dataset"))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset_exists = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()

    if dc_cfg.get("resume", False) and dataset_exists:
        cmd.append("--resume=true")
    elif not dc_cfg.get("resume", False):
        if dataset_root.exists():
            shutil.rmtree(dataset_root)

    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        datacollect_state["process"] = p
        datacollect_state["running"] = True
        datacollect_state["started_at"] = time.time()
        datacollect_state["log_lines"] = []
        datacollect_state["completed"] = False
        threading.Thread(target=_datacollect_reader, args=(p, datacollect_state), daemon=True).start()
        return {"ok": True, "message": "Recording started"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/datacollect/stop")
async def datacollect_stop():
    if not datacollect_state["running"]:
        return {"ok": True, "message": "Not recording"}
    p = datacollect_state["process"]
    if p:
        import signal as sig
        p.send_signal(sig.SIGINT)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    datacollect_state["process"] = None
    datacollect_state["running"] = False
    datacollect_state["started_at"] = None
    return {"ok": True, "message": "Recording stopped"}


@app.get("/api/datacollect/status")
async def datacollect_status():
    elapsed = 0
    if datacollect_state["started_at"] and datacollect_state["running"]:
        elapsed = time.time() - datacollect_state["started_at"]
    p = datacollect_state["process"]
    if p and p.poll() is not None:
        datacollect_state["running"] = False
        datacollect_state["process"] = None
        datacollect_state["started_at"] = None
        datacollect_state["completed"] = True
    with datacollect_state["log_lock"]:
        recent_logs = list(datacollect_state["log_lines"][-50:])
    return {
        "running": datacollect_state["running"],
        "elapsed": round(elapsed, 1),
        "logs": recent_logs,
        "completed": datacollect_state["completed"],
    }


# ── Teleop (for data collection) ──

@app.post("/api/teleop/start")
async def teleop_start():
    if teleop_state["running"]:
        return {"ok": False, "error": "Teleop already running"}
    if robot_state["connected"]:
        disconnect_robot()
    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    teleop_cfg = cfg.get("teleop", {})
    cmd = [
        "/opt/miniconda3/envs/lerobot/bin/lerobot-teleoperate",
        f"--robot.type=so101_follower",
        f"--robot.port={robot_cfg.get('port', '')}",
        f"--robot.id={robot_cfg.get('id', 'my_awesome_follower_arm')}",
        f"--teleop.type={teleop_cfg.get('type', 'so101_leader')}",
        f"--teleop.port={teleop_cfg.get('port', '')}",
        f"--teleop.id={teleop_cfg.get('id', 'my_awesome_leader_arm')}",
        "--display_data=false",
    ]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        teleop_state["process"] = p
        teleop_state["running"] = True
        teleop_state["log_lines"] = []

        def reader(proc, state):
            for line in iter(proc.stdout.readline, b''):
                text = line.decode("utf-8", errors="replace").rstrip()
                with state["log_lock"]:
                    state["log_lines"].append(text)
                    if len(state["log_lines"]) > 500:
                        state["log_lines"] = state["log_lines"][-500:]
            proc.wait()
            state["running"] = False

        threading.Thread(target=reader, args=(p, teleop_state), daemon=True).start()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/teleop/stop")
async def teleop_stop():
    if not teleop_state["running"]:
        return {"ok": True, "message": "Not running"}
    p = teleop_state["process"]
    if p:
        import signal as sig
        p.send_signal(sig.SIGINT)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    teleop_state["process"] = None
    teleop_state["running"] = False
    return {"ok": True}


@app.get("/api/teleop/status")
async def teleop_status():
    with teleop_state["log_lock"]:
        logs = list(teleop_state["log_lines"][-20:])
    return {"running": teleop_state["running"], "logs": logs}


# ══════════ Training via SSH ══════════

TRAIN_LOG_DIR = ROOT / "data" / "training_logs"
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _gpu_ssh_config():
    cfg = load_config()
    gpu = cfg.get("gpu_server", {})
    return {
        "host": gpu.get("host", ""),
        "user": gpu.get("user", ""),
        "port": gpu.get("port", 22),
        "password": gpu.get("password", ""),
        "conda_env": gpu.get("conda_env", "lerobot"),
        "wandb_key": gpu.get("wandb_key", ""),
        "work_dir": gpu.get("work_dir", ""),
    }


def _conda_init_cmd() -> str:
    """Build conda init command. Uses conda_init_path from config or auto-detect common locations."""
    gpu = _gpu_ssh_config()
    custom = gpu.get("conda_init_path", "")
    if custom:
        return f". {custom}"
    # Try common conda locations
    return (
        "for p in /opt/tljh/user/etc/profile.d/conda.sh "
        "$HOME/miniconda3/etc/profile.d/conda.sh "
        "$HOME/anaconda3/etc/profile.d/conda.sh "
        "$HOME/miniforge3/etc/profile.d/conda.sh "
        "/opt/conda/etc/profile.d/conda.sh; do "
        "[ -f \"$p\" ] && . \"$p\" && break; done"
    )


def _build_ssh_cmd(remote_command: str, gpu: dict = None):
    if gpu is None:
        gpu = _gpu_ssh_config()
    # Source conda init script directly (bashrc guard blocks non-interactive shells)
    conda_init = _conda_init_cmd()
    wrapped = f"bash -c {shlex.quote(conda_init + ' && ' + remote_command)}"
    ssh_cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
        "-p", str(gpu["port"]), f"{gpu['user']}@{gpu['host']}",
        wrapped,
    ]
    if gpu.get("password"):
        return ["sshpass", "-p", gpu["password"]] + ssh_cmd
    return ssh_cmd


def _ssh_run_sync(remote_command: str, timeout: int = 30) -> dict:
    gpu = _gpu_ssh_config()
    if not gpu["host"] or not gpu["user"]:
        return {"ok": False, "error": "SSH not configured. Set GPU Server in Setup > Config."}
    cmd = _build_ssh_cmd(remote_command, gpu)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return {"ok": True, "stdout": result.stdout, "stderr": result.stderr}
        return {"ok": False, "error": result.stderr.strip() or f"SSH exit code {result.returncode}",
                "stdout": result.stdout, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"SSH timed out ({timeout}s)"}
    except FileNotFoundError:
        return {"ok": False, "error": "sshpass not found. Install it or use SSH key auth."}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _write_remote_file(remote_path: str, content: str, gpu: dict = None):
    """Write content to a file on the remote server via SSH stdin."""
    if gpu is None:
        gpu = _gpu_ssh_config()
    if not gpu["host"] or not gpu["user"]:
        return {"ok": False, "error": "SSH not configured."}
    # Use ssh with stdin to write file - avoids all quoting issues
    dir_path = remote_path.rsplit("/", 1)[0]
    ssh_cmd = [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
        "-p", str(gpu["port"]), f"{gpu['user']}@{gpu['host']}",
        f"mkdir -p {dir_path} && cat > {remote_path}",
    ]
    if gpu.get("password"):
        ssh_cmd = ["sshpass", "-p", gpu["password"]] + ssh_cmd
    try:
        result = subprocess.run(ssh_cmd, input=content, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return {"ok": True}
        return {"ok": False, "error": result.stderr.strip() or f"Exit code {result.returncode}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _save_job_local(job_id: str, status_data: dict = None, log_lines: list = None):
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    if job_path.exists():
        existing = json.loads(job_path.read_text())
    else:
        existing = {"job_id": job_id, "logs": []}
    if status_data:
        existing["status"] = status_data.get("status", existing.get("status"))
        existing["returncode"] = status_data.get("returncode", existing.get("returncode"))
        for k in ("cmd", "config", "job_name", "pid", "started_at", "finished_at", "log_file", "wb_url"):
            if k in status_data:
                existing[k] = status_data[k]
    if log_lines and isinstance(log_lines, list):
        if len(log_lines) > len(existing.get("logs", [])):
            existing["logs"] = log_lines
        # Extract W&B URL from logs if not already set
        if not existing.get("wb_url"):
            import re
            for line in log_lines:
                m = re.search(r'https://wandb\.ai/\S+', line)
                if m:
                    existing["wb_url"] = m.group(0).rstrip(')')
                    break
    existing["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    job_path.write_text(json.dumps(existing, ensure_ascii=False, indent=1))


def _build_train_command(config: dict, gpu: dict) -> str:
    """Build the lerobot-train command string for remote execution."""
    def val(raw):
        text = str(raw)
        if all(c.isalnum() or c in '_./:@+-' for c in text):
            return text
        return f"'{text}'"

    policy_type = config.get("policy_type", "act")
    lerobot_policy = "pi0" if policy_type == "pi05" else policy_type
    dataset_repo_id = config.get("dataset_repo_id", "")
    ds_name = dataset_repo_id.split("/")[-1] if "/" in dataset_repo_id else dataset_repo_id
    job_name = config.get("job_name", f"{policy_type}_{ds_name}")

    args = [
        f"--policy.type={val(lerobot_policy)}",
        f"--dataset.repo_id={val(dataset_repo_id)}",
        f"--policy.use_amp={'true' if config.get('use_amp', True) else 'false'}",
        f"--steps={val(config.get('steps', 100000))}",
        f"--batch_size={val(config.get('batch_size', 32))}",
        f"--policy.optimizer_lr={val(config.get('optimizer_lr', 0.00001))}",
    ]

    if policy_type == "smolvla":
        args.append(f"--policy.load_vlm_weights={'true' if config.get('load_vlm_weights') else 'false'}")
        if config.get("vlm_model_name"):
            args.append(f"--policy.vlm_model_name={val(config['vlm_model_name'])}")
        args.append(f"--policy.train_expert_only={'true' if config.get('train_expert_only') else 'false'}")
        args.append(f"--policy.freeze_vision_encoder={'true' if config.get('freeze_vision_encoder') else 'false'}")
    elif policy_type == "act":
        if config.get("vision_backbone"):
            args.append(f"--policy.vision_backbone={val(config['vision_backbone'])}")
        if config.get("use_vae"):
            args.append("--policy.use_vae=true")
    elif policy_type == "pi05":
        args.append("--policy.pretrained_path=lerobot/pi0.5")
        if config.get("paligemma_variant"):
            args.append(f"--policy.paligemma_variant={val(config['paligemma_variant'])}")
        if config.get("action_expert_variant"):
            args.append(f"--policy.action_expert_variant={val(config['action_expert_variant'])}")

    if config.get("chunk_size"):
        args.append(f"--policy.chunk_size={val(config['chunk_size'])}")
    if config.get("n_action_steps"):
        args.append(f"--policy.n_action_steps={val(config['n_action_steps'])}")
    if config.get("gradient_checkpointing"):
        args.append("--policy.gradient_checkpointing=true")

    wandb_enable = config.get("wandb_enable", False)
    args.append(f"--wandb.enable={'true' if wandb_enable else 'false'}")
    if wandb_enable:
        args.append(f"--wandb.project={val(config.get('wandb_project', 'SO101-Experiments'))}")
        if config.get("wandb_entity"):
            args.append(f"--wandb.entity={val(config['wandb_entity'])}")

    args.append(f"--output_dir={val(f'outputs/train/{job_name}')}")
    args.append(f"--job_name={val(job_name)}")
    if config.get("policy_repo_id"):
        args.append(f"--policy.repo_id={val(config['policy_repo_id'])}")
    if config.get("resume"):
        args.append("--resume=true")

    return "lerobot-train " + " ".join(args)


def _get_worker_script(gpu: dict) -> str:
    """Generate the persistent worker bash script that processes the queue."""
    conda_init = _conda_init_cmd()
    cfg_all = load_config()
    hf_token = cfg_all.get("hf_token", "")
    hf_export = f"export HF_TOKEN={shlex.quote(hf_token)}" if hf_token else ""
    wandb_export = f"export WANDB_API_KEY={shlex.quote(gpu['wandb_key'])}" if gpu.get("wandb_key") else ""
    work_dir = gpu["work_dir"]
    vram_threshold = 8000

    return f"""#!/bin/bash
{conda_init}
conda activate {gpu['conda_env']}
{hf_export}
{wandb_export}
cd {work_dir}

QUEUE_FILE="{work_dir}/train_queue.json"
VRAM_THRESHOLD={vram_threshold}

echo "=== Train worker started ==="

while true; do
    # Read first pending job from queue
    JOB_JSON=$(python3 -c "
import json, fcntl, sys, os
qf = '$QUEUE_FILE'
if not os.path.exists(qf):
    sys.exit(0)
with open(qf) as f:
    fcntl.flock(f, fcntl.LOCK_SH)
    q = json.load(f)
for j in q.get('jobs', []):
    if j['status'] == 'pending':
        print(json.dumps(j))
        break
" 2>/dev/null)

    if [ -z "$JOB_JSON" ]; then
        sleep 10
        continue
    fi

    # Extract fields
    JOB_ID=$(echo "$JOB_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
    TRAIN_CMD=$(echo "$JOB_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['train_cmd'])")
    LOG_FILE=$(echo "$JOB_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['log_file'])")
    DS_NAME=$(echo "$JOB_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ds_name',''))")
    OUTPUT_DIR=$(echo "$JOB_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('output_dir',''))")

    echo "Worker: Found pending job $JOB_ID"

    # Wait for VRAM to be free
    echo "=== Waiting for GPU ===" > "$LOG_FILE"
    while true; do
        VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        VRAM_USED=${{VRAM_USED:-0}}
        if [ "$VRAM_USED" -lt $VRAM_THRESHOLD ]; then
            echo "GPU is free (VRAM: ${{VRAM_USED}}MB). Starting training..." >> "$LOG_FILE"
            break
        fi
        echo "Waiting for GPU... (VRAM: ${{VRAM_USED}}MB/${{VRAM_THRESHOLD}}MB threshold)" >> "$LOG_FILE"
        sleep 30
    done

    # Mark job as running in queue
    python3 -c "
import json, fcntl
with open('$QUEUE_FILE', 'r+') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    q = json.load(f)
    for j in q['jobs']:
        if j['job_id'] == '$JOB_ID':
            j['status'] = 'running'
            break
    f.seek(0); f.truncate()
    json.dump(q, f, indent=1)
"

    # Clear HF cache for this dataset
    if [ -n "$DS_NAME" ]; then
        echo "Clearing HF cache for $DS_NAME..." >> "$LOG_FILE"
        find ~/.cache/huggingface -maxdepth 4 -type d -name "*${{DS_NAME}}*" -exec rm -rf '{{}}' + 2>/dev/null || true
    fi

    # Clean previous output
    [ -n "$OUTPUT_DIR" ] && rm -rf "$OUTPUT_DIR"

    # Run training
    echo "=== Starting training ===" >> "$LOG_FILE"
    eval "$TRAIN_CMD" >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    # Mark completed or failed in queue
    if [ $EXIT_CODE -eq 0 ]; then
        FINAL_STATUS="completed"
    else
        FINAL_STATUS="failed"
    fi

    python3 -c "
import json, fcntl
with open('$QUEUE_FILE', 'r+') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    q = json.load(f)
    for j in q['jobs']:
        if j['job_id'] == '$JOB_ID':
            j['status'] = '$FINAL_STATUS'
            j['exit_code'] = $EXIT_CODE
            break
    f.seek(0); f.truncate()
    json.dump(q, f, indent=1)
"
    echo "Worker: Job $JOB_ID finished with status $FINAL_STATUS (exit=$EXIT_CODE)"
done
"""


def _ensure_worker_running(gpu: dict) -> dict:
    """Ensure the train_worker tmux session is running on the GPU server."""
    r = _ssh_run_sync("tmux has-session -t train_worker 2>/dev/null && echo ALIVE || echo DEAD", timeout=10)
    if r["ok"] and "ALIVE" in r.get("stdout", ""):
        return {"ok": True, "started": False}

    # Write worker script and start it
    script_path = f"{gpu['work_dir']}/train_worker.sh"
    script_content = _get_worker_script(gpu)
    wr = _write_remote_file(script_path, script_content, gpu)
    if not wr["ok"]:
        return {"ok": False, "error": f"Failed to write worker script: {wr['error']}"}

    r = _ssh_run_sync(
        f"chmod +x {script_path} && "
        f"tmux new-session -d -s train_worker 'bash {script_path}'",
        timeout=15
    )
    if r["ok"]:
        return {"ok": True, "started": True}
    return {"ok": False, "error": r.get("error", "Failed to start worker")}


def _append_to_queue(job_entry: dict, gpu: dict) -> dict:
    """Append a job to the remote queue file atomically.
    Writes job JSON to a temp file first, then uses python to merge it into the queue.
    """
    queue_path = f"{gpu['work_dir']}/train_queue.json"
    tmp_path = f"{gpu['work_dir']}/._job_tmp.json"
    # Write job entry to temp file (avoids all quoting issues)
    wr = _write_remote_file(tmp_path, json.dumps(job_entry), gpu)
    if not wr["ok"]:
        return wr
    py_script = (
        "import json, fcntl, os\n"
        f"qf = '{queue_path}'\n"
        f"tf = '{tmp_path}'\n"
        "if not os.path.exists(qf):\n"
        "    with open(qf, 'w') as f: json.dump({'jobs': []}, f)\n"
        "job = json.load(open(tf))\n"
        "with open(qf, 'r+') as f:\n"
        "    fcntl.flock(f, fcntl.LOCK_EX)\n"
        "    q = json.load(f)\n"
        "    q['jobs'].append(job)\n"
        "    f.seek(0); f.truncate()\n"
        "    json.dump(q, f, indent=1)\n"
        "os.remove(tf)\n"
        "print('OK')\n"
    )
    script_path = f"{gpu['work_dir']}/._queue_append.py"
    wr2 = _write_remote_file(script_path, py_script, gpu)
    if not wr2["ok"]:
        return wr2
    return _ssh_run_sync(f"python3 {script_path}", timeout=15)


def _remove_from_queue(job_id: str, gpu: dict, new_status: str = "cancelled") -> dict:
    """Mark a job in the remote queue as cancelled/stopped."""
    queue_path = f"{gpu['work_dir']}/train_queue.json"
    py_script = (
        "import json, fcntl, os\n"
        f"qf = '{queue_path}'\n"
        "if not os.path.exists(qf): exit(0)\n"
        "with open(qf, 'r+') as f:\n"
        "    fcntl.flock(f, fcntl.LOCK_EX)\n"
        "    q = json.load(f)\n"
        "    for j in q['jobs']:\n"
        f"        if j['job_id'] == '{job_id}':\n"
        f"            j['status'] = '{new_status}'\n"
        "            break\n"
        "    f.seek(0); f.truncate()\n"
        "    json.dump(q, f, indent=1)\n"
        "print('OK')\n"
    )
    script_path = f"{gpu['work_dir']}/._queue_update.py"
    wr = _write_remote_file(script_path, py_script, gpu)
    if not wr["ok"]:
        return wr
    return _ssh_run_sync(f"python3 {script_path}", timeout=15)


def _read_remote_queue(gpu: dict) -> dict:
    """Read the remote queue file and return {job_id: status} map."""
    queue_path = f"{gpu['work_dir']}/train_queue.json"
    r = _ssh_run_sync(f"cat {queue_path} 2>/dev/null || echo '{{}}'", timeout=10)
    statuses = {}
    if r["ok"] and r.get("stdout", "").strip():
        try:
            qdata = json.loads(r["stdout"])
            for j in qdata.get("jobs", []):
                statuses[j["job_id"]] = j.get("status", "unknown")
        except (json.JSONDecodeError, KeyError):
            pass
    return statuses


def _prepare_job(body: dict, gpu: dict) -> dict:
    """Prepare job metadata for the queue-based worker system."""
    policy_type = body.get("policy_type", "act")
    ds_name = body.get("dataset_repo_id", "").split("/")[-1]
    job_name = body.get("job_name", f"{policy_type}_{ds_name}")
    job_id = f"{job_name}_{int(time.time())}"
    log_file = f"{gpu['work_dir']}/logs/{job_id}.log"
    train_cmd = _build_train_command(body, gpu)
    output_dir = f"{gpu['work_dir']}/outputs/train/{job_name}"

    return {
        "ok": True, "job_id": job_id, "job_name": job_name,
        "log_file": log_file, "train_cmd": train_cmd, "config": body,
        "queue_entry": {
            "job_id": job_id, "status": "pending", "job_name": job_name,
            "train_cmd": train_cmd, "log_file": log_file,
            "output_dir": output_dir, "ds_name": ds_name,
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }


def _start_job_on_server(job_info: dict) -> dict:
    """Add a job to the remote queue and ensure the worker is running."""
    gpu = _gpu_ssh_config()

    # Ensure log directory exists
    _ssh_run_sync(f"mkdir -p {gpu['work_dir']}/logs", timeout=10)

    # Append job to queue
    result = _append_to_queue(job_info["queue_entry"], gpu)
    if not result["ok"]:
        return {"ok": False, "error": f"Failed to append to queue: {result.get('error')}"}

    # Ensure worker is running
    wr = _ensure_worker_running(gpu)
    if not wr["ok"]:
        return {"ok": False, "error": f"Failed to start worker: {wr.get('error')}"}

    # Save local tracking
    _save_job_local(job_info["job_id"], {
        "status": "queued",
        "config": job_info["config"],
        "job_name": job_info["job_name"],
        "cmd": job_info["train_cmd"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "log_file": job_info["log_file"],
        "tmux_session": "train_worker",
    })
    return {"ok": True}


@app.post("/api/training/train")
async def training_start(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        return {"ok": False, "error": f"Invalid request body: {e}"}

    gpu = _gpu_ssh_config()
    if not gpu["host"]:
        return {"ok": False, "error": "SSH not configured. Set GPU Server in Setup > Config."}

    job_info = _prepare_job(body, gpu)
    if not job_info["ok"]:
        return {"ok": False, "error": job_info["error"]}

    result = _start_job_on_server(job_info)
    if result["ok"]:
        return {"ok": True, "job_id": job_info["job_id"], "status": "queued"}
    return {"ok": False, "error": result.get("error", "SSH failed")}


@app.get("/api/training/jobs")
async def training_jobs():
    gpu = _gpu_ssh_config()
    ssh_ok = bool(gpu["host"] and gpu["user"])

    # Check worker status and remote queue
    worker_alive = False
    remote_statuses = {}
    if ssh_ok:
        r = _ssh_run_sync("tmux has-session -t train_worker 2>/dev/null && echo ALIVE || echo DEAD", timeout=10)
        worker_alive = r["ok"] and "ALIVE" in r.get("stdout", "")
        remote_statuses = _read_remote_queue(gpu)

    merged = {}
    for f in sorted(TRAIN_LOG_DIR.glob("*.json"), reverse=True):
        try:
            local = json.loads(f.read_text())
            jid = local.get("job_id", f.stem)
            config = local.get("config") or {}
            status = local.get("status", "unknown")

            # Update status from server-side logs and remote queue
            if status in ("queued", "running", "downloading"):
                log_file = local.get("log_file", "")
                log_tail = ""
                if ssh_ok and log_file:
                    lr = _ssh_run_sync(f"tail -n 30 {log_file} 2>/dev/null || true", timeout=10)
                    if lr["ok"]:
                        log_tail = lr["stdout"] or ""
                if not log_tail:
                    log_tail = "\n".join(local.get("logs", [])[-30:])

                if "End of training" in log_tail:
                    status = "completed"
                    _save_job_local(jid, {"status": "completed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
                elif "Starting training" in log_tail and status == "queued":
                    status = "running"
                    _save_job_local(jid, {"status": "running"})
                elif ssh_ok and jid in remote_statuses:
                    rs = remote_statuses[jid]
                    if rs == "completed":
                        status = "completed"
                        _save_job_local(jid, {"status": "completed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
                    elif rs == "failed":
                        status = "failed"
                        _save_job_local(jid, {"status": "failed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
                    elif rs == "running":
                        status = "running"
                        _save_job_local(jid, {"status": "running"})
                    elif rs in ("pending",) and not worker_alive:
                        status = "failed"
                        _save_job_local(jid, {"status": "failed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

            merged[jid] = {
                "status": status,
                "returncode": local.get("returncode"),
                "job_name": local.get("job_name"),
                "config": config,
                "dataset_repo_id": config.get("dataset_repo_id") or config.get("repo_id", ""),
                "last_updated": local.get("last_updated"),
                "wb_url": local.get("wb_url"),
                "tmux_session": "train_worker",
            }
        except Exception:
            pass
    return {"server_online": ssh_ok, "jobs": merged}


@app.get("/api/training/jobs/{job_id}/status")
async def training_job_status(job_id: str):
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    if not job_path.exists():
        return {"error": "Job not found"}
    local = json.loads(job_path.read_text())
    status = local.get("status", "unknown")

    # Update status from server-side logs and remote queue
    if status in ("queued", "running", "downloading"):
        log_file = local.get("log_file", "")
        log_tail = ""
        if log_file:
            lr = _ssh_run_sync(f"tail -n 30 {log_file} 2>/dev/null || true", timeout=10)
            if lr["ok"]:
                log_tail = lr["stdout"] or ""
        if not log_tail:
            log_tail = "\n".join(local.get("logs", [])[-30:])

        if "End of training" in log_tail:
            status = "completed"
            _save_job_local(job_id, {"status": "completed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        elif "Starting training" in log_tail and status == "queued":
            status = "running"
            _save_job_local(job_id, {"status": "running"})
        else:
            gpu = _gpu_ssh_config()
            remote_statuses = _read_remote_queue(gpu)
            rs = remote_statuses.get(job_id)
            if rs == "completed":
                status = "completed"
                _save_job_local(job_id, {"status": "completed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            elif rs == "failed":
                status = "failed"
                _save_job_local(job_id, {"status": "failed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            elif rs == "running":
                status = "running"
                _save_job_local(job_id, {"status": "running"})

    return {
        "job_id": job_id,
        "status": status,
        "returncode": local.get("returncode"),
        "config": local.get("config") or {},
        "job_name": local.get("job_name"),
        "cmd": local.get("cmd"),
        "started_at": local.get("started_at"),
        "finished_at": local.get("finished_at"),
        "wb_url": local.get("wb_url"),
    }


@app.get("/api/training/jobs/{job_id}/logs")
async def training_job_logs(job_id: str, tail: int = 100):
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    if not job_path.exists():
        return {"error": "Log not found"}
    local = json.loads(job_path.read_text())
    log_file = local.get("log_file", "")

    # Try to extract W&B URL from early log lines if not yet found
    if log_file and not local.get("wb_url"):
        wr = _ssh_run_sync(f"grep -m1 'https://wandb.ai/' {log_file} 2>/dev/null || true", timeout=10)
        if wr["ok"] and "https://wandb.ai/" in (wr["stdout"] or ""):
            import re
            m = re.search(r'https://wandb\.ai/\S+', wr["stdout"])
            if m:
                _save_job_local(job_id, {"wb_url": m.group(0).rstrip(')')})

    # Try to fetch logs from remote via SSH
    if log_file:
        r = _ssh_run_sync(f"tail -n {tail} {log_file} 2>/dev/null || echo '(log file not found)'", timeout=10)
        if r["ok"]:
            lines = r["stdout"].strip().split("\n") if r["stdout"].strip() else []
            if lines:
                _save_job_local(job_id, log_lines=lines)
            return {"job_id": job_id, "lines": lines}

    # Fallback to locally cached logs
    lines = local.get("logs", [])
    if tail:
        lines = lines[-tail:]
    return {"job_id": job_id, "lines": lines, "local_only": True}


@app.delete("/api/training/jobs/{job_id}")
async def training_stop_job(job_id: str):
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    if not job_path.exists():
        return {"error": "Job not found"}
    local = json.loads(job_path.read_text())
    gpu = _gpu_ssh_config()
    status = local.get("status", "")

    # Check remote queue status
    remote_statuses = _read_remote_queue(gpu)
    rs = remote_statuses.get(job_id, "")

    if rs == "running":
        # Kill worker (stops current training), mark job cancelled, restart worker for remaining jobs
        _ssh_run_sync("tmux kill-session -t train_worker 2>/dev/null || true", timeout=10)
        _remove_from_queue(job_id, gpu, "cancelled")
        _ensure_worker_running(gpu)
    elif rs == "pending":
        # Just remove from queue
        _remove_from_queue(job_id, gpu, "cancelled")
    else:
        # Legacy: try killing old-style tmux session
        tmux = local.get("tmux_session", "")
        if tmux and tmux != "train_worker":
            _ssh_run_sync(f"tmux kill-session -t {tmux} 2>/dev/null || true", timeout=10)

    _save_job_local(job_id, {"status": "stopped", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return {"ok": True, "status": "stopped"}


@app.post("/api/training/kill-all")
async def training_kill_all():
    results = {"killed": [], "cleaned": [], "errors": []}
    gpu = _gpu_ssh_config()

    # Kill worker and clear remote queue
    _ssh_run_sync("tmux kill-session -t train_worker 2>/dev/null || true", timeout=10)
    queue_path = f"{gpu['work_dir']}/train_queue.json"
    _write_remote_file(queue_path, '{"jobs": []}', gpu)

    # Also kill any leftover old-style train_ tmux sessions
    _ssh_run_sync("tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^train_' | xargs -I{} tmux kill-session -t {} 2>/dev/null || true", timeout=10)

    # Clean local job files
    for f in TRAIN_LOG_DIR.glob("*.json"):
        try:
            local = json.loads(f.read_text())
            jid = local.get("job_id", f.stem)
            status = local.get("status", "")
            if status in ("running", "downloading", "queued"):
                results["killed"].append(jid)
            f.unlink()
            if status in ("failed", "stopped", "cancelled"):
                results["cleaned"].append(jid)
        except Exception as e:
            results["errors"].append(str(e))

    # Clear HF cache on GPU server
    _ssh_run_sync("find ~/.cache/huggingface -maxdepth 4 -type d -name 'datasets--*' -exec rm -rf {} + 2>/dev/null || true", timeout=30)
    results["cache_cleared"] = True
    return {"ok": True, **results}


@app.patch("/api/training/history/{job_id}")
async def update_job_history(job_id: str, request: Request):
    body = await request.json()
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    if not job_path.exists():
        return {"ok": False, "error": "Not found"}
    new_status = body.get("status")
    if not new_status:
        return {"ok": False, "error": "No status provided"}
    _save_job_local(job_id, {"status": new_status})
    return {"ok": True, "status": new_status}


@app.delete("/api/training/history/{job_id}")
async def delete_job_history(job_id: str):
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    if job_path.exists():
        job_path.unlink()
        return {"ok": True}
    return {"ok": False, "error": "Not found"}


# ══════════ VLA Inference API ══════════

infer_py_state = {"running": False, "process": None, "log_lines": []}


def _infer_py_reader(proc, state):
    for line in iter(proc.stdout.readline, b''):
        text = line.decode("utf-8", errors="replace").rstrip()
        state["log_lines"].append(text)
        if len(state["log_lines"]) > 2000:
            state["log_lines"] = state["log_lines"][-1500:]
    proc.wait()
    state["running"] = False


@app.post("/api/infer-python/start")
async def infer_python_start(request: Request):
    if infer_py_state["running"]:
        return {"ok": False, "error": "Already running"}
    body = await request.json()
    model_name = body.get("model_name", "")
    task = body.get("task", "")
    fps = int(body.get("fps", 30))
    cam_width = int(body.get("width", 320))
    cam_height = int(body.get("height", 240))
    chunk_size = int(body.get("chunk_size", 0))
    n_action_steps = int(body.get("n_action_steps", 0))
    cache_language = body.get("cache_language", False)

    if not model_name:
        return {"ok": False, "error": "No model selected"}

    if teleop_state["running"]:
        await teleop_stop()
    if robot_state["connected"]:
        disconnect_robot()
        time.sleep(0.2)

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    cams = robot_cfg.get("cameras", {})

    cameras_json = {}
    for cam_name, cam_c in cams.items():
        cameras_json[cam_name] = cam_c.get("index", 0)

    model_dir = f"./models/{model_name}"

    cmd = [
        sys.executable, "infer_python.py",
        f"--model-path={model_dir}",
        f"--robot-port={robot_cfg.get('port', '')}",
        f"--robot-id={robot_cfg.get('id', 'my_awesome_follower_arm')}",
        f"--cameras={json.dumps(cameras_json)}",
        f"--fps={fps}",
        f"--width={cam_width}",
        f"--height={cam_height}",
    ]
    if task:
        cmd.append(f"--task={task}")
    if chunk_size > 0:
        cmd.append(f"--chunk-size={chunk_size}")
    if n_action_steps > 0:
        cmd.append(f"--n-action-steps={n_action_steps}")
    if cache_language:
        cmd.append("--cache-language")

    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        infer_py_state["process"] = p
        infer_py_state["running"] = True
        infer_py_state["log_lines"] = []
        threading.Thread(target=_infer_py_reader, args=(p, infer_py_state), daemon=True).start()
        return {"ok": True, "pid": p.pid, "cmd": " ".join(cmd)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/infer-python/stop")
async def infer_python_stop():
    if not infer_py_state["running"] and not infer_py_state["process"]:
        return {"ok": True, "message": "Not running"}

    p = infer_py_state["process"]
    if p:
        import signal as sig
        try:
            os.killpg(os.getpgid(p.pid), sig.SIGINT)
        except (ProcessLookupError, OSError):
            try:
                p.send_signal(sig.SIGINT)
            except (ProcessLookupError, OSError):
                pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), sig.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    p.kill()
                except (ProcessLookupError, OSError):
                    pass
    infer_py_state["running"] = False
    return {"ok": True}


@app.get("/api/infer-python/status")
async def infer_python_status():
    return {
        "running": infer_py_state["running"],
        "lines": infer_py_state["log_lines"][-100:],
    }


# ══════════ Evaluation API ══════════

EVAL_DIR = ROOT / "data" / "eval_sessions"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_CONFIG_FILE = EVAL_DIR / "eval_config.json"
EVAL_VIDEOS_DIR = ROOT / "data" / "eval_videos"
EVAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESET_POS_FILE = EVAL_DIR / "reset_position.json"

eval_state = {
    "running": False,
    "process": None,
    "reader_thread": None,
    "run_token": None,
    "log_lines": [],
    "session_id": None,
    "worker_pid": None,
    "phase": "idle",
}
eval_lifecycle_lock = threading.Lock()
eval_video_state = {"recording": False, "stop_event": None, "video_path": None}


def _eval_load_reset_position():
    if EVAL_RESET_POS_FILE.exists():
        try:
            return json.loads(EVAL_RESET_POS_FILE.read_text())
        except Exception:
            pass
    return None


def _eval_move_to_reset(owner="eval_reset", acquire=True):
    pos = _eval_load_reset_position()
    if not pos:
        return True, None

    if acquire and not acquire_robot_operation(owner):
        return False, f"HARDWARE_BUSY: Robot hardware is busy with {robot_operation_owner}"

    try:
        connect_robot()
        for _ in range(30):
            robot_send_positions(pos, owner=owner)
            time.sleep(0.033)
        disconnect_robot()
        return True, None
    except Exception as e:
        try:
            disconnect_robot()
        except Exception:
            pass
        return False, str(e)
    finally:
        if acquire:
            release_robot_operation(owner)


def _eval_filter_cameras(all_cameras: dict, cam_select: str) -> dict:
    if cam_select == '1-top':
        return {k: v for k, v in all_cameras.items() if k == 'top'}
    elif cam_select == '1-wrist':
        return {k: v for k, v in all_cameras.items() if k == 'wrist'}
    elif cam_select.startswith('1'):
        return dict(list(all_cameras.items())[:1])
    else:
        return all_cameras


@app.get("/api/eval/config")
async def eval_get_config():
    if EVAL_CONFIG_FILE.exists():
        try:
            return {"ok": True, "config": json.loads(EVAL_CONFIG_FILE.read_text())}
        except Exception:
            pass
    return {"ok": True, "config": {}}


@app.post("/api/eval/config")
async def eval_save_config(request: Request):
    body = await request.json()
    EVAL_CONFIG_FILE.write_text(json.dumps(body, indent=2))
    return {"ok": True}


@app.get("/api/eval/reset-position")
async def eval_get_reset_position():
    pos = _eval_load_reset_position()
    return {"ok": True, "position": pos}


@app.post("/api/eval/reset-position")
async def eval_save_reset_position(request: Request):
    body = await request.json()
    pos = body.get("position")
    if pos:
        EVAL_RESET_POS_FILE.write_text(json.dumps(pos, indent=2))
        return {"ok": True, "position": pos}
    auto_connected = False
    try:
        if not robot_state["connected"]:
            connect_robot()
            auto_connected = True
        pos = robot_get_positions()
        EVAL_RESET_POS_FILE.write_text(json.dumps(pos, indent=2))
        if auto_connected:
            disconnect_robot()
        return {"ok": True, "position": pos}
    except Exception as e:
        if auto_connected:
            try:
                disconnect_robot()
            except Exception:
                pass
        return {"ok": False, "error": str(e)}


@app.delete("/api/eval/reset-position")
async def eval_delete_reset_position():
    if EVAL_RESET_POS_FILE.exists():
        EVAL_RESET_POS_FILE.unlink()
    return {"ok": True}


@app.post("/api/eval/move-to-reset")
async def eval_move_to_reset():
    pos = _eval_load_reset_position()
    if not pos:
        return {"ok": False, "error": "No reset position saved"}
    ok, err = _eval_move_to_reset()
    if ok:
        return {"ok": True}
    if str(err).startswith("HARDWARE_BUSY:"):
        return JSONResponse(status_code=409, content={"ok": False, "code": "HARDWARE_BUSY", "error": str(err).split(":", 1)[1].strip()})
    return {"ok": False, "error": err}


@app.post("/api/eval/start")
async def eval_start(request: Request):
    body = await request.json()
    if eval_state["running"]:
        return {"ok": False, "error": "Already running"}
    if not body.get("model_name", ""):
        return {"ok": False, "error": "No model selected"}
    if not acquire_robot_operation("eval"):
        return JSONResponse(status_code=409, content={
            "ok": False,
            "code": "HARDWARE_BUSY",
            "error": f"Robot hardware is busy with {robot_operation_owner}",
        })
    start_token = uuid.uuid4().hex
    with eval_lifecycle_lock:
        eval_state["running"] = True
        eval_state["phase"] = "starting"
        eval_state["run_token"] = start_token
        eval_state["worker_pid"] = None
    body["_start_token"] = start_token
    try:
        result = await _eval_start_owned(body)
    except Exception:
        with eval_lifecycle_lock:
            if eval_state.get("run_token") == start_token:
                eval_state["running"] = False
                eval_state["phase"] = "idle"
                eval_state["run_token"] = None
        release_robot_operation("eval")
        raise
    if not result.get("ok"):
        with eval_lifecycle_lock:
            if eval_state.get("run_token") == start_token:
                eval_state["running"] = False
                eval_state["phase"] = "idle"
                eval_state["run_token"] = None
        release_robot_operation("eval")
    return result


async def _eval_start_owned(body: dict):
    start_token = body.get("_start_token")
    model_id = body.get("model_id", "")
    task = body.get("task", "")
    fps = int(body.get("fps", 30))
    cam_width = int(body.get("width", 320))
    cam_height = int(body.get("height", 240))
    cam_select = str(body.get("cam_count", "2"))
    chunk_size = int(body.get("chunk_size", 0))
    n_action_steps = int(body.get("n_action_steps", 0))
    duration = max(1, int(body.get("duration", 12)))

    if teleop_state["running"]:
        await teleop_stop()
    if robot_state["connected"]:
        disconnect_robot()
        time.sleep(1.0)

    # Challenge environment: generate augmentation params for this episode
    # Seed is based on episode number so same episode = same augmentation across models
    challenge_env = body.get("challenge_env")
    augment_params = None
    augment_seed = None
    if challenge_env and challenge_env.get("enabled"):
        episode = int(body.get("episode", 1))
        augment_seed = episode
        random.seed(augment_seed)
        np.random.seed(augment_seed)
        ch_intensity = int(challenge_env.get("intensity", 3))
        ch_camera = challenge_env.get("camera", {})
        ch_light = challenge_env.get("light", {})
        cam_p = _augp_camera(ch_camera, intensity=ch_intensity)
        light_p = _augp_light(ch_light, intensity=ch_intensity)
        # Restore RNG to non-deterministic state
        random.seed()
        np.random.seed()
        if cam_p or light_p:
            augment_params = {}
            if cam_p:
                # Convert numpy arrays to lists for JSON serialization
                serializable_cam = {}
                for k, v in cam_p.items():
                    serializable_cam[k] = v.tolist() if hasattr(v, 'tolist') else v
                augment_params["camera"] = serializable_cam
            if light_p:
                augment_params["light"] = light_p

    reset_pos = _eval_load_reset_position()
    if reset_pos:
        ok, err = _eval_move_to_reset(owner="eval", acquire=False)
        if not ok:
            return {"ok": False, "error": f"Failed to move to reset position: {err}"}

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    all_cams = robot_cfg.get("cameras", {})

    selected_cams = _eval_filter_cameras(all_cams, cam_select)
    cams = selected_cams

    record = get_model_record(_load_model_registry(cfg), str(model_id))
    if not record or not record.get("selectable"):
        return {"ok": False, "error": f"Selected model is not executable: {model_id}"}
    model_path = (ROOT / str(record.get("local_path", ""))).resolve()
    models_root = (ROOT / "models").resolve()
    if models_root not in model_path.parents or not model_path.exists():
        return {"ok": False, "error": "Model checkpoint is not available locally"}
    resolved = {**record, "local_path": str(model_path)}

    try:
        lerobot_python = Path("/opt/miniconda3/envs/lerobot/bin/python")
        runtime_python = str(lerobot_python if lerobot_python.is_file() else Path(sys.executable))
        cmd = build_worker_command(
            python=runtime_python,
            script=ROOT / "infer_python.py",
            model=resolved,
            robot=robot_cfg,
            cameras=cams,
            fps=fps,
            width=cam_width,
            height=cam_height,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
        )
        ready = await asyncio.to_thread(vla_manager.ensure_model, cmd, str(model_id), 120.0)
        if not ready.get("ok"):
            await asyncio.to_thread(vla_manager.stop)
            return ready
        with eval_lifecycle_lock:
            cancelled = start_token is not None and (
                eval_state.get("run_token") != start_token or not eval_state.get("running")
            )
            if not cancelled:
                eval_state["worker_pid"] = ready.get("pid")
                eval_state["phase"] = "connecting"
        if cancelled:
            await asyncio.to_thread(vla_manager.stop_if_pid, ready.get("pid"))
            return {"ok": False, "code": "EVAL_START_CANCELLED", "error": "Evaluation start was cancelled"}
        verification_image = ROOT / "data" / "eval_verification" / "latest.jpg"
        verification_image.parent.mkdir(parents=True, exist_ok=True)
        started = await asyncio.to_thread(
            vla_manager.run_task,
            task,
            duration * fps,
            str(verification_image),
            max(120.0, duration + 30.0),
            30.0,
            augment_params,
        )
        with eval_lifecycle_lock:
            cancelled = start_token is not None and (
                eval_state.get("run_token") != start_token or not eval_state.get("running")
            )
        if cancelled:
            active_pid = vla_manager.status().get("pid")
            if active_pid is not None:
                await asyncio.to_thread(vla_manager.stop_if_pid, active_pid)
            return {"ok": False, "code": "EVAL_START_CANCELLED", "error": "Evaluation start was cancelled"}
        if not started.get("ok"):
            await asyncio.to_thread(vla_manager.stop)
            return started
        with eval_lifecycle_lock:
            eval_state["running"] = True
            eval_state["phase"] = "running"
            eval_state["process"] = None
            eval_state["run_token"] = start_token or uuid.uuid4().hex
            eval_state["reader_thread"] = None
            eval_state["log_lines"] = []
            eval_state["worker_pid"] = started.get("pid")
        result = {
            "ok": True,
            "pid": started.get("pid"),
            "reused_model": bool(ready.get("reused")),
            "cmd": " ".join(cmd),
        }
        if augment_params:
            result["augment_params"] = augment_params
            result["augment_seed"] = augment_seed
        return result
    except Exception as e:
        await asyncio.to_thread(vla_manager.stop)
        return {"ok": False, "error": str(e)}


@app.post("/api/eval/stop")
async def eval_stop():
    if not eval_state.get("running"):
        worker = vla_manager.status()
        return {
            "ok": True,
            "message": "No evaluation episode is running",
            "model_ready": bool(worker.get("model_ready")),
            "pid": worker.get("pid"),
        }
    # Stop video recording
    video_filename = None
    if eval_video_state["recording"]:
        signal_file = EVAL_VIDEOS_DIR / ".recording.json"
        signal_file.unlink(missing_ok=True)
        video_filename = os.path.basename(eval_video_state["video_path"]) if eval_video_state["video_path"] else None
        eval_video_state["recording"] = False
    eval_video_state["stop_event"] = None

    phase = eval_state.get("phase", "running")
    start_in_flight = phase in {"starting", "connecting"}
    if start_in_flight:
        stopped = await asyncio.to_thread(vla_manager.stop)
        released = {"ok": bool(stopped.get("ok", True)), "worker_stopped": True}
    else:
        # Stop inference loop but keep hardware connected for fast restart
        released = await asyncio.to_thread(vla_manager.stop_cycle, 5.0)
        if not released.get("ok") and not released.get("already_stopped"):
            # Fallback: full release if stop_cycle fails
            released = await asyncio.to_thread(vla_manager.release_hardware, 10.0)
            if not released.get("ok"):
                await asyncio.to_thread(vla_manager.stop)
    with eval_lifecycle_lock:
        eval_state["running"] = False
        eval_state["process"] = None
        eval_state["reader_thread"] = None
        eval_state["run_token"] = None
        eval_state["phase"] = "idle"
    if not start_in_flight:
        release_robot_operation("eval")

    worker = vla_manager.status()
    return {
        "ok": True,
        "video_filename": video_filename,
        "model_ready": bool(worker.get("model_ready")),
        "pid": worker.get("pid"),
    }


@app.post("/api/eval/start-recording")
async def eval_start_recording(request: Request):
    """Signal the VLA worker to start recording video via a file-based trigger."""
    body = await request.json()
    fps = int(body.get("fps", 30))
    width = int(body.get("width", 320))
    height = int(body.get("height", 240))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"eval_{ts}.mp4"
    video_path = EVAL_VIDEOS_DIR / video_filename

    # Write signal file that infer_python.py watches
    signal_file = EVAL_VIDEOS_DIR / ".recording.json"
    signal_file.write_text(json.dumps({
        "video_path": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
    }))

    eval_video_state["recording"] = True
    eval_video_state["stop_event"] = threading.Event()
    eval_video_state["video_path"] = str(video_path)

    return {"ok": True, "video_filename": video_filename}


@app.post("/api/eval/unload")
async def eval_unload():
    if eval_state.get("running"):
        stopped = await eval_stop()
        if not stopped.get("ok") and not stopped.get("worker_stopped"):
            return stopped
    worker = vla_manager.status()
    eval_pid = eval_state.get("worker_pid")
    if eval_pid is None or worker.get("pid") != eval_pid:
        return {"ok": True, "message": "Evaluation does not own the active worker"}
    with robot_operation_lock:
        if robot_operation_owner not in (None, "eval"):
            return JSONResponse(status_code=409, content={
                "ok": False,
                "code": "HARDWARE_BUSY",
                "error": f"VLA worker is in use by {robot_operation_owner}",
            })
        result = vla_manager.stop_if_pid(eval_pid)
    if result.get("code") == "VLA_WORKER_CHANGED":
        return {"ok": True, "message": "Evaluation worker already changed"}
    eval_state["worker_pid"] = None
    return {"ok": bool(result.get("ok", True))}


@app.get("/api/eval/status")
async def eval_status():
    worker = vla_manager.status()
    episode_running = bool(eval_state["running"] and worker.get("running") and worker.get("hardware_connected"))
    if eval_state["running"] and not worker.get("running"):
        eval_state["running"] = False
        release_robot_operation("eval")
    return {
        "running": episode_running,
        "worker_running": bool(worker.get("running")),
        "model_ready": bool(worker.get("model_ready")),
        "pid": worker.get("pid"),
        "state": worker.get("state"),
        "lines": worker.get("lines", [])[-100:],
    }


@app.post("/api/eval/sessions")
async def eval_save_session(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        return {"ok": False, "error": "No session_id"}
    path = EVAL_DIR / f"{session_id}.json"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2))
    return {"ok": True}


@app.get("/api/eval/sessions")
async def eval_list_sessions():
    sessions = []
    for f in sorted(EVAL_DIR.glob("*.json"), reverse=True):
        try:
            d = json.loads(f.read_text())
            results = d.get("results", [])
            a1 = sum(1 for r in results if r.get("result") == "1-attempt")
            a2 = sum(1 for r in results if r.get("result") == "2-attempt")
            a3 = sum(1 for r in results if r.get("result") == "3-attempt")
            sessions.append({
                "id": d.get("session_id", f.stem),
                "name": d.get("session_name", ""),
                "model": d.get("model_name", ""),
                "task": d.get("task", ""),
                "target": d.get("target_runs", 0),
                "completed": d.get("completed", 0),
                "success": d.get("success", 0),
                "fail": d.get("fail", 0),
                "a1": a1, "a2": a2, "a3": a3,
                "created": d.get("created", ""),
            })
        except Exception:
            pass
    return {"ok": True, "sessions": sessions}


@app.get("/api/eval/sessions/{session_id}")
async def eval_get_session(session_id: str):
    path = EVAL_DIR / f"{session_id}.json"
    if not path.exists():
        return {"ok": False, "error": "Session not found"}
    return {"ok": True, **json.loads(path.read_text())}


@app.delete("/api/eval/sessions/{session_id}")
async def eval_delete_session(session_id: str):
    path = EVAL_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
    return {"ok": True}


# ══════════ SSH GPU Server ══════════

@app.post("/api/ssh/test")
async def ssh_test():
    gpu = _gpu_ssh_config()
    if not gpu["host"] or not gpu["user"]:
        return {"ok": False, "error": "SSH host and user are required. Configure in Setup > Config."}
    conda_env = gpu["conda_env"] or "lerobot"
    work_dir = gpu["work_dir"] or "~/vla_benchmark_ws"
    conda_init = _conda_init_cmd()

    test_cmd = (
        f'echo "=== SSH Connection ==="; '
        f'echo "Host: $(hostname)"; echo "User: $(whoami)"; '
        f'echo ""; '
        # Check tmux
        f'echo "=== tmux ==="; '
        f'if command -v tmux >/dev/null 2>&1; then echo "OK: $(tmux -V)"; else echo "MISSING: tmux not installed"; fi; '
        f'echo ""; '
        # Check conda
        f'echo "=== Conda ==="; '
        f'{conda_init} 2>/dev/null; '
        f'if command -v conda >/dev/null 2>&1; then echo "OK: $(conda --version)"; else echo "MISSING: conda not found"; fi; '
        f'echo ""; '
        # Check conda env
        f'echo "=== Conda Env: {conda_env} ==="; '
        f'if conda env list 2>/dev/null | grep -q "^{conda_env} "; then '
        f'echo "OK: env exists"; '
        f'conda activate {conda_env} 2>/dev/null && echo "Python: $(python --version 2>&1)"; '
        f'else echo "MISSING: env \'{conda_env}\' not found"; fi; '
        f'echo ""; '
        # Check lerobot
        f'echo "=== LeRobot ==="; '
        f'if conda activate {conda_env} 2>/dev/null && command -v lerobot-train >/dev/null 2>&1; then '
        f'echo "OK: lerobot-train found"; '
        f'else echo "MISSING: lerobot not installed in env \'{conda_env}\'"; fi; '
        f'echo ""; '
        # Check work dir
        f'echo "=== Working Directory ==="; '
        f'if [ -d "{work_dir}" ]; then echo "OK: {work_dir} exists"; '
        f'if [ -d "{work_dir}/logs" ]; then echo "OK: logs/ exists"; else echo "MISSING: {work_dir}/logs not found"; fi; '
        f'else echo "MISSING: {work_dir} not found"; fi; '
        f'echo ""; '
        # Check HF login (use saved token)
        f'echo "=== HuggingFace ==="; '
        + (
            f'if conda activate {conda_env} 2>/dev/null && HF_TOKEN={shlex.quote(load_config().get("hf_token", ""))} python -c "import os; from huggingface_hub import HfApi; print(\'OK: logged in as \' + HfApi(token=os.environ[\'HF_TOKEN\']).whoami()[\'name\'])" 2>/dev/null; then '
            f'echo ""; '
            f'else echo "WARNING: HF token invalid or expired"; fi; '
            if load_config().get("hf_token") else
            f'echo "WARNING: HF token not set (login in Setup > Hugging Face)"; '
        ) +
        f'echo ""; '
        # Check W&B
        f'echo "=== Weights & Biases ==="; '
        f'if conda activate {conda_env} 2>/dev/null && python -c "import wandb" 2>/dev/null; then '
        f'echo "OK: wandb package installed"; '
        + (f'echo "OK: WANDB_API_KEY configured in app"; ' if gpu.get("wandb_key") else f'echo "WARNING: WANDB_API_KEY not set (add in GPU Server config)"; ')
        + f'else echo "WARNING: wandb not installed (pip install wandb)"; fi'
    )
    result = _ssh_run_sync(test_cmd, timeout=30)
    if not result["ok"]:
        return {"ok": False, "error": result.get("error", "SSH failed")}

    output = result["stdout"]
    issues = [line for line in output.split("\n") if line.startswith("MISSING:")]
    warnings = [line for line in output.split("\n") if line.startswith("WARNING:")]
    return {
        "ok": True,
        "output": output,
        "issues": issues,
        "warnings": warnings,
        "needs_fix": len(issues) > 0,
    }


@app.post("/api/ssh/fix-env")
async def ssh_fix_env():
    gpu = _gpu_ssh_config()
    if not gpu["host"] or not gpu["user"]:
        return {"ok": False, "error": "SSH not configured."}
    conda_env = gpu["conda_env"] or "lerobot"
    work_dir = gpu["work_dir"] or "~/vla_benchmark_ws"
    conda_init = _conda_init_cmd()

    fix_cmd = (
        # Install tmux
        f'echo ">>> Checking tmux..."; '
        f'if ! command -v tmux >/dev/null 2>&1; then '
        f'echo "Installing tmux..."; sudo apt-get install -y tmux 2>/dev/null || echo "WARN: cannot install tmux (no sudo). Install manually: sudo apt install tmux"; '
        f'else echo "tmux already installed"; fi; '
        f'echo ""; '
        # Init conda
        f'{conda_init} 2>/dev/null; '
        # Create conda env if missing
        f'echo ">>> Checking conda env: {conda_env}..."; '
        f'if ! conda env list 2>/dev/null | grep -q "^{conda_env} "; then '
        f'echo "Creating conda env {conda_env} with Python 3.12..."; '
        f'conda create -n {conda_env} python=3.12 -y; '
        f'else echo "Conda env {conda_env} already exists"; fi; '
        f'echo ""; '
        # Install lerobot
        f'echo ">>> Checking lerobot..."; '
        f'conda activate {conda_env}; '
        f'if ! command -v lerobot-train >/dev/null 2>&1; then '
        f'echo "Installing lerobot..."; pip install lerobot; '
        f'else echo "lerobot already installed"; fi; '
        f'echo ""; '
        # Create work dir
        f'echo ">>> Creating working directory..."; '
        f'mkdir -p {work_dir}/logs; '
        f'echo "Created {work_dir}/logs"; '
        f'echo ""; '
        f'echo "=== Done ==="'
    )
    result = _ssh_run_sync(fix_cmd, timeout=300)
    if result["ok"]:
        return {"ok": True, "output": result["stdout"] + (result.get("stderr") or "")}
    return {"ok": False, "error": result.get("error", "SSH failed"), "output": result.get("stdout", "")}


# ══════════ HF Models / Datasets Discovery ══════════

@app.get("/api/hf-whoami")
async def hf_whoami():
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.whoami()
        # Return masked token from config
        cfg = load_config()
        hf_token = cfg.get("hf_token", "")
        masked = (hf_token[:4] + "..." + hf_token[-4:]) if hf_token and len(hf_token) > 8 else ""
        return {"ok": True, "username": info.get("name", ""), "fullname": info.get("fullname", ""), "hf_token_masked": masked}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/hf-login")
async def hf_login(request: Request):
    try:
        body = await request.json()
        token = body.get("token", "").strip()
        if not token:
            return {"ok": False, "error": "No token provided"}
        from huggingface_hub import HfApi, login
        login(token=token)
        api = HfApi()
        info = api.whoami()
        username = info.get("name", "")
        # Save token and username to config
        cfg = load_config()
        cfg["hf_token"] = token
        if username:
            cfg["hf_repo_name"] = username
        _write_config(cfg)
        return {"ok": True, "username": username}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/hf-logout")
async def hf_logout():
    try:
        from huggingface_hub import logout
        logout()
        # Clear saved token from config
        cfg = load_config()
        cfg.pop("hf_token", None)
        _write_config(cfg)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


@app.get("/api/hf-models")
async def list_hf_models():
    cfg = load_config()
    hf_user = str(cfg.get("hf_repo_name", "")).strip()
    if not hf_user:
        return {"ok": True, "models": [], "hf_user": ""}
    try:
        from huggingface_hub import HfApi
        token = _read_env().get("HF_TOKEN") or None
        api = HfApi(token=token)
        models = []
        for m in api.list_models(author=hf_user, sort="lastModified", limit=100):
            name = m.id.rsplit("/", 1)[-1]
            local_path = MODELS_DIR / name
            local_exists = local_path.exists() and any(local_path.iterdir()) if local_path.exists() else False
            models.append({"repo_id": m.id, "id": m.id, "name": name, "local_exists": local_exists})
        return {"ok": True, "models": models, "hf_user": hf_user}
    except Exception as e:
        # Fallback: show locally downloaded models even if HF API fails
        local_models = []
        if MODELS_DIR.exists():
            for d in sorted(MODELS_DIR.iterdir()):
                if d.is_dir() and (d / "config.json").exists():
                    local_models.append({
                        "repo_id": f"{hf_user}/{d.name}",
                        "id": f"{hf_user}/{d.name}",
                        "name": d.name,
                        "local_exists": True,
                    })
        return {"ok": True, "models": local_models, "hf_user": hf_user,
                "warning": f"HF API unavailable ({type(e).__name__}), showing local models only"}


# ── Model Download ──
_model_download_state = {"running": False, "status": "", "log_lines": [], "error": None}


@app.post("/api/model-download")
async def model_download_start(request: Request):
    body = await request.json()
    repo_id = body.get("repo_id", "")
    model_name = body.get("model_name", "")
    if not repo_id:
        return {"ok": False, "error": "No repo_id provided"}
    if _model_download_state["running"]:
        return {"ok": False, "error": "A download is already in progress"}

    _model_download_state.update({"running": True, "status": "Starting download...", "log_lines": [], "error": None})

    import threading
    def _download():
        try:
            from huggingface_hub import snapshot_download, HfApi
            token = _read_env().get("HF_TOKEN") or None
            local_dir = MODELS_DIR / (model_name or repo_id.rsplit("/", 1)[-1])
            # List files for progress tracking
            try:
                api = HfApi(token=token)
                repo_files = list(api.list_repo_files(repo_id, token=token))
                total_files = len(repo_files)
                _model_download_state["log_lines"].append(f"Downloading {repo_id} ({total_files} files)")
            except Exception:
                total_files = 0
                _model_download_state["log_lines"].append(f"Downloading {repo_id}")

            # Monitor progress in background
            _stop_monitor = threading.Event()
            def _monitor():
                seen = set()
                while not _stop_monitor.is_set():
                    if local_dir.exists():
                        current = set(str(p.relative_to(local_dir)) for p in local_dir.rglob("*") if p.is_file() and ".cache" not in str(p))
                        new_files = current - seen
                        for nf in sorted(new_files):
                            _model_download_state["log_lines"].append(f"  ✓ {nf}")
                        seen = current
                        if total_files:
                            _model_download_state["status"] = f"Downloading {repo_id}... ({len(seen)}/{total_files} files)"
                    _stop_monitor.wait(2)
            mon = threading.Thread(target=_monitor, daemon=True)
            mon.start()

            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
            )
            _stop_monitor.set()
            mon.join(timeout=3)
            _model_download_state["status"] = "Download complete"
            _model_download_state["log_lines"].append(f"Download complete! ({total_files} files)")
        except Exception as e:
            _model_download_state["error"] = str(e)
            _model_download_state["status"] = f"Error: {e}"
            _model_download_state["log_lines"].append(f"Error: {e}")
        finally:
            _model_download_state["running"] = False

    threading.Thread(target=_download, daemon=True).start()
    return {"ok": True}


@app.get("/api/model-download/status")
async def model_download_status():
    return {
        "running": _model_download_state["running"],
        "status": _model_download_state["status"],
        "log_lines": _model_download_state["log_lines"],
        "error": _model_download_state["error"],
    }


@app.post("/api/model-download-batch")
async def model_download_batch(request: Request):
    body = await request.json()
    models = body.get("models", [])
    if not models:
        return {"ok": False, "error": "No models provided"}
    if _model_download_state["running"]:
        return {"ok": False, "error": "A download is already in progress"}

    _model_download_state.update({"running": True, "status": "Starting batch download...", "log_lines": [], "error": None})

    import threading
    def _download_batch():
        try:
            from huggingface_hub import snapshot_download
            token = _read_env().get("HF_TOKEN") or None
            total = len(models)
            for i, m in enumerate(models):
                repo_id = m.get("repo_id", "")
                name = m.get("name", repo_id.rsplit("/", 1)[-1])
                local_dir = MODELS_DIR / name
                _model_download_state["status"] = f"Downloading {i+1}/{total}: {name}"
                _model_download_state["log_lines"].append(f"[{i+1}/{total}] Downloading {repo_id}...")
                try:
                    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), token=token)
                    _model_download_state["log_lines"].append(f"[{i+1}/{total}] Done: {name}")
                except Exception as e:
                    _model_download_state["log_lines"].append(f"[{i+1}/{total}] Error: {name} - {e}")
            _model_download_state["status"] = f"Batch download complete ({total} models)"
            _model_download_state["log_lines"].append(f"All {total} models downloaded!")
        except Exception as e:
            _model_download_state["error"] = str(e)
            _model_download_state["status"] = f"Error: {e}"
        finally:
            _model_download_state["running"] = False

    threading.Thread(target=_download_batch, daemon=True).start()
    return {"ok": True}


@app.get("/api/hf-datasets")
async def list_hf_datasets():
    cfg = load_config()
    hf_user = str(cfg.get("hf_repo_name", "")).strip()
    if not hf_user:
        return {"ok": True, "datasets": []}
    try:
        from huggingface_hub import HfApi
        token = _read_env().get("HF_TOKEN") or None
        api = HfApi(token=token)
        datasets = []
        for d in api.list_datasets(author=hf_user, limit=200):
            name = d.id.rsplit("/", 1)[-1]
            datasets.append({"repo_id": d.id, "name": name})
        return {"ok": True, "datasets": datasets}
    except Exception as e:
        return {"ok": False, "error": str(e), "datasets": []}


# ══════════ Startup ══════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
