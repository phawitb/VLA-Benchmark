"""Release robot hardware between processes without dropping the held pose."""

from __future__ import annotations


def disconnect_robot_hardware(robot, keep_torque: bool = False) -> None:
    """Disconnect a LeRobot robot, optionally preserving servo torque for handoff."""
    if not keep_torque:
        robot.disconnect()
        return

    config = getattr(robot, "config", None)
    if config is None or not hasattr(config, "disable_torque_on_disconnect"):
        raise RuntimeError("Robot does not support torque-preserving disconnect")
    previous = config.disable_torque_on_disconnect
    config.disable_torque_on_disconnect = False
    try:
        robot.disconnect()
    finally:
        config.disable_torque_on_disconnect = previous
