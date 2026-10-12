"""Safe subprocess lifecycle for bounded VLA plan-step execution."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path


def build_infer_command(
    python: str,
    script: Path,
    model: dict,
    task: str,
    robot: dict,
    cameras: dict,
    max_steps: int,
    verification_image: Path | None = None,
    verification_timeout: float = 120.0,
) -> list[str]:
    required_names = [
        feature.rsplit(".", 1)[-1]
        for feature in model.get("camera_features", [])
    ]
    missing = [name for name in required_names if name not in cameras]
    if missing:
        raise ValueError(f"Missing configured cameras: {', '.join(missing)}")
    camera_indices = {
        name: int(cameras[name].get("index", 0))
        for name in required_names
    }
    first_camera = cameras[required_names[0]] if required_names else {}
    command = [
        python,
        str(script),
        f"--model-path={model['local_path']}",
        f"--task={task}",
        f"--robot-port={robot.get('port', '')}",
        f"--robot-id={robot.get('id', 'my_awesome_follower_arm')}",
        f"--cameras={json.dumps(camera_indices, separators=(',', ':'))}",
        f"--fps={int(first_camera.get('fps', 30))}",
        f"--width={int(first_camera.get('w', first_camera.get('width', 320)))}",
        f"--height={int(first_camera.get('h', first_camera.get('height', 240)))}",
        f"--max-steps={int(max_steps)}",
    ]
    if verification_image is not None:
        command.extend([
            f"--verification-image={verification_image}",
            f"--verification-timeout={float(verification_timeout)}",
        ])
    if model.get("policy_type") == "smolvla":
        command.append("--cache-language")
    return command


def build_worker_command(
    python: str,
    script: Path,
    model: dict,
    robot: dict,
    cameras: dict,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    chunk_size: int = 0,
    n_action_steps: int = 0,
    fourcc: str = "",
) -> list[str]:
    required_names = [
        feature.rsplit(".", 1)[-1]
        for feature in model.get("camera_features", [])
    ]
    missing = [name for name in required_names if name not in cameras]
    if missing:
        raise ValueError(f"Missing configured cameras: {', '.join(missing)}")
    camera_indices = {
        name: int(cameras[name].get("index", 0))
        for name in required_names
    }
    first_camera = cameras[required_names[0]] if required_names else {}
    command = [
        python,
        str(script),
        f"--model-path={model['local_path']}",
        f"--model-id={model['id']}",
        f"--robot-port={robot.get('port', '')}",
        f"--robot-id={robot.get('id', 'my_awesome_follower_arm')}",
        f"--cameras={json.dumps(camera_indices, separators=(',', ':'))}",
        f"--fps={int(fps if fps is not None else first_camera.get('fps', 30))}",
        f"--width={int(width if width is not None else first_camera.get('w', first_camera.get('width', 320)))}",
        f"--height={int(height if height is not None else first_camera.get('h', first_camera.get('height', 240)))}",
        "--persistent-worker",
    ]
    if chunk_size > 0:
        command.append(f"--chunk-size={int(chunk_size)}")
    if n_action_steps > 0:
        command.append(f"--n-action-steps={int(n_action_steps)}")
    if fourcc:
        command.append(f"--fourcc={fourcc}")
    if model.get("policy_type") == "smolvla":
        command.append("--cache-language")
    return command


class VlaProcessManager:
    def __init__(self, max_log_lines: int = 500):
        self.max_log_lines = max_log_lines
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._process: subprocess.Popen | None = None
        self._state = "idle"
        self._model_id: str | None = None
        self._task: str | None = None
        self._exit_code: int | None = None
        self._lines: list[str] = []
        self._cycle_ready_count = 0
        self._verification_image: str | None = None
        self._model_ready = False
        self._hardware_connected = False
        self._active_task: str | None = None
        self._last_error: dict | None = None
        self._persistent = False
        self._command: list[str] | None = None

    def start(self, command: list[str], model_id: str, task: str | None) -> dict:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return {"ok": False, "code": "VLA_ALREADY_RUNNING", "error": "VLA inference is already running"}
            self._state = "starting"
            self._model_id = model_id
            self._task = task
            self._exit_code = None
            self._lines = []
            self._cycle_ready_count = 0
            self._verification_image = None
            self._model_ready = False
            self._hardware_connected = False
            self._active_task = None
            self._last_error = None
            self._persistent = "--persistent-worker" in command or "--worker-protocol-test" in command
            self._command = list(command)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._exit_code = -1
                self._lines = [str(exc)]
            return {"ok": False, "code": "VLA_LAUNCH_FAILED", "error": str(exc)}
        with self._lock:
            self._process = process
            self._state = "running"
        threading.Thread(target=self._reader, args=(process,), daemon=True).start()
        return {"ok": True, "pid": process.pid, "model_id": model_id, "task": task}

    def _reader(self, process: subprocess.Popen) -> None:
        stream = process.stdout
        if stream is not None:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                with self._lock:
                    self._lines.append(line)
                    event_name, separator, raw_event = line.partition(" ")
                    event = None
                    if separator and event_name in {
                        "MODEL_READY", "HARDWARE_READY", "CYCLE_READY",
                        "HARDWARE_RELEASED", "WORKER_ERROR",
                    }:
                        try:
                            event = json.loads(raw_event)
                        except json.JSONDecodeError:
                            event = None
                    if event_name == "MODEL_READY" and isinstance(event, dict):
                        self._model_ready = True
                        self._state = "model_ready"
                    elif event_name == "HARDWARE_READY" and isinstance(event, dict):
                        self._hardware_connected = True
                        self._active_task = str(event.get("task", ""))
                        self._task = self._active_task
                        self._state = "running"
                    elif event_name == "CYCLE_READY" and isinstance(event, dict):
                        try:
                            self._cycle_ready_count += 1
                            self._verification_image = str(event.get("image", ""))
                            self._state = "waiting_for_verification"
                        except (TypeError, ValueError):
                            pass
                    elif event_name == "HARDWARE_RELEASED" and isinstance(event, dict):
                        self._hardware_connected = False
                        self._active_task = None
                        self._task = None
                        self._state = "model_ready"
                    elif event_name == "WORKER_ERROR" and isinstance(event, dict):
                        self._last_error = event
                    self._condition.notify_all()
                    if len(self._lines) > self.max_log_lines:
                        self._lines = self._lines[-self.max_log_lines:]
        exit_code = process.wait()
        with self._lock:
            self._exit_code = exit_code
            if self._state != "stopped":
                self._state = "completed" if exit_code == 0 else "failed"
            self._hardware_connected = False
            self._model_ready = False
            self._condition.notify_all()

    def status(self) -> dict:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {
                "ok": True,
                "state": self._state,
                "running": running,
                "model_id": self._model_id,
                "task": self._task,
                "exit_code": self._exit_code,
                "pid": self._process.pid if self._process is not None else None,
                "cycle_ready_count": self._cycle_ready_count,
                "verification_image": self._verification_image,
                "model_ready": self._model_ready,
                "hardware_connected": self._hardware_connected,
                "active_task": self._active_task,
                "last_error": dict(self._last_error) if self._last_error else None,
                "lines": list(self._lines),
            }

    def _send_command(self, payload: dict, legacy: str | None = None) -> dict:
        with self._lock:
            process = self._process
            stream = process.stdin if process is not None else None
            running = process is not None and process.poll() is None
            persistent = self._persistent
        if not running or stream is None:
            return {"ok": False, "code": "VLA_CONTROL_UNAVAILABLE", "error": "VLA control channel is unavailable"}
        line = json.dumps(payload, separators=(",", ":")) if persistent or legacy is None else legacy
        try:
            stream.write((line + "\n").encode())
            stream.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._lock:
                self._state = "failed"
            return {"ok": False, "code": "VLA_CONTROL_FAILED", "error": str(exc)}
        return {"ok": True, "pid": process.pid}

    def _wait_for(self, predicate, timeout: float, code: str, message: str) -> dict:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not predicate():
                process = self._process
                if process is None or process.poll() is not None:
                    return {"ok": False, "code": "VLA_PROCESS_EXITED", "error": "VLA worker exited before acknowledgement"}
                if self._last_error:
                    return {
                        "ok": False,
                        "code": str(self._last_error.get("code", "WORKER_ERROR")),
                        "error": str(self._last_error.get("error", "VLA worker error")),
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"ok": False, "code": code, "error": message}
                self._condition.wait(remaining)
            process = self._process
            return {"ok": True, "pid": process.pid if process is not None else None}

    def ensure_model(self, command: list[str], model_id: str, timeout: float = 120.0) -> dict:
        current = self.status()
        with self._lock:
            same_command = self._command == list(command)
        if current["running"] and current["model_id"] == model_id and current["model_ready"] and same_command:
            return {"ok": True, "pid": current["pid"], "reused": True}
        if current["running"]:
            released = self.release_hardware()
            if not released.get("ok"):
                return released
            self.stop()
        started = self.start(command, model_id, None)
        if not started.get("ok"):
            return started
        ready = self._wait_for(
            lambda: self._model_ready,
            timeout,
            "MODEL_LOAD_TIMEOUT",
            f"Timed out loading VLA model {model_id}",
        )
        if ready.get("ok"):
            ready["reused"] = False
        return ready

    def run_task(
        self,
        task: str,
        max_steps: int,
        verification_image: str,
        verification_timeout: float = 120.0,
        timeout: float = 30.0,
    ) -> dict:
        current = self.status()
        if not current["model_ready"]:
            return {"ok": False, "code": "VLA_NOT_READY", "error": "VLA model is not ready for a new task"}
        with self._lock:
            self._state = "connecting_vla_hardware"
            self._active_task = None  # reset so _wait_for detects new HARDWARE_READY
        sent = self._send_command({
            "command": "run_task",
            "task": task,
            "max_steps": int(max_steps),
            "verification_image": verification_image,
            "verification_timeout": float(verification_timeout),
        })
        if not sent.get("ok"):
            return sent
        return self._wait_for(
            lambda: self._hardware_connected and self._active_task == task,
            timeout,
            "HARDWARE_CONNECT_TIMEOUT",
            "Timed out connecting VLA hardware",
        )

    def release_hardware(self, timeout: float = 10.0) -> dict:
        current = self.status()
        if not current["running"] or not current["hardware_connected"]:
            return {"ok": True, "pid": current["pid"], "already_released": True}
        with self._lock:
            self._state = "releasing_hardware"
        sent = self._send_command({"command": "release_hardware"})
        if not sent.get("ok"):
            return sent
        return self._wait_for(
            lambda: not self._hardware_connected,
            timeout,
            "HARDWARE_RELEASE_TIMEOUT",
            "Timed out releasing VLA hardware",
        )

    def stop_cycle(self, timeout: float = 10.0) -> dict:
        """Stop current inference cycle but keep hardware connected."""
        current = self.status()
        if not current["running"]:
            return {"ok": True, "already_stopped": True}
        if current["state"] == "waiting_for_verification":
            return {"ok": True, "already_stopped": True}
        sent = self._send_command({"command": "stop_cycle"})
        if not sent.get("ok"):
            return sent
        return self._wait_for(
            lambda: self._state == "waiting_for_verification",
            timeout,
            "STOP_CYCLE_TIMEOUT",
            "Timed out stopping inference cycle",
        )

    def continue_cycle(self) -> dict:
        with self._lock:
            process = self._process
            if self._state != "waiting_for_verification" or process is None or process.poll() is not None:
                return {"ok": False, "code": "VLA_NOT_WAITING", "error": "VLA process is not waiting for verification"}
            stream = process.stdin
            self._state = "running"
        if stream is None:
            return {"ok": False, "code": "VLA_CONTROL_UNAVAILABLE", "error": "VLA control channel is unavailable"}
        return self._send_command({"command": "continue"}, legacy="continue")

    def stop_if_pid(self, expected_pid: int, timeout: float = 5.0) -> dict:
        with self._lock:
            process = self._process
            actual_pid = process.pid if process is not None and process.poll() is None else None
            if actual_pid != expected_pid:
                return {
                    "ok": False,
                    "code": "VLA_WORKER_CHANGED",
                    "error": f"Expected VLA worker {expected_pid}, active worker is {actual_pid}",
                }
        return self._stop_process(process, timeout)

    def stop(self, timeout: float = 5.0) -> dict:
        with self._lock:
            process = self._process
        return self._stop_process(process, timeout)

    def _stop_process(self, process: subprocess.Popen | None, timeout: float) -> dict:
        if process is None or process.poll() is not None:
            return {"ok": True, "state": self.status()["state"]}
        with self._lock:
            is_current = self._process is process
            persistent = self._persistent if is_current else False
            stream = process.stdin
        if persistent and stream is not None:
            try:
                stream.write((json.dumps({"command": "shutdown"}, separators=(",", ":")) + "\n").encode())
                stream.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                process.wait(timeout=min(timeout, 1.0))
            except subprocess.TimeoutExpired:
                pass
            else:
                with self._lock:
                    if self._process is process:
                        self._state = "stopped"
                        self._exit_code = process.returncode
                        self._model_ready = False
                        self._hardware_connected = False
                return {"ok": True, "state": "stopped"}
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            try:
                process.send_signal(signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                process.kill()
            process.wait(timeout=timeout)
        with self._lock:
            if self._process is process:
                self._state = "stopped"
                self._exit_code = process.returncode
                self._model_ready = False
                self._hardware_connected = False
        return {"ok": True, "state": "stopped"}
