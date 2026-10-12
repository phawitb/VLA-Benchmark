"""Discover robot policy checkpoints and their declared training capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq


SUPPORTED_POLICY_TYPES = {"smolvla", "act", "pi0", "pi05"}
MODEL_PREFIXES = (
    "smolvla_", "smolvla-", "pi05_", "pi05-", "pi0_", "pi0-",
    "act_", "act-", "diffusion_", "diffusion-", "vla_", "vla-",
    "model_", "model-",
)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _read_tasks_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        tasks = pq.read_table(path).to_pydict().get("task", [])
        return _dedupe(tasks)
    except Exception:
        return []


def _dataset_root(project_root: Path, dataset: dict) -> Path | None:
    raw_root = dataset.get("root")
    configured = str(raw_root).strip() if raw_root is not None else ""
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = project_root / path
        return path.resolve()
    repo_id = str(dataset.get("repo_id", "")).strip()
    if repo_id:
        return (project_root / "data" / repo_id.rsplit("/", 1)[-1]).resolve()
    return None


def _local_record(model_dir: Path, project_root: Path, train: dict, policy: dict) -> dict:
    dataset = train.get("dataset", {}) if isinstance(train.get("dataset", {}), dict) else {}
    dataset_id = str(dataset.get("repo_id", "")).strip() or None
    dataset_root = _dataset_root(project_root, dataset)
    tasks = _read_tasks_file(dataset_root / "meta" / "tasks.parquet") if dataset_root else []
    policy_type = str(policy.get("type", "unknown"))
    camera_features = sorted(
        key for key in policy.get("input_features", {})
        if str(key).startswith("observation.images.")
    )
    return {
        "id": model_dir.name,
        "repo_id": policy.get("repo_id"),
        "display_name": model_dir.name,
        "policy_type": policy_type,
        "downloaded": True,
        "local_path": model_dir.relative_to(project_root).as_posix(),
        "dataset_id": dataset_id,
        "tasks": tasks,
        "camera_features": camera_features,
        "selectable": True,
        "unavailable_reason": None,
    }


def load_local_models(root: Path, configured_cameras: set[str]) -> list[dict]:
    root = Path(root).resolve()
    if not root.exists():
        return []
    project_root = root.parent
    records = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        resolved = model_dir.resolve()
        if root not in resolved.parents:
            continue
        train_path = model_dir / "train_config.json"
        config_path = model_dir / "config.json"
        if not train_path.exists() or not config_path.exists():
            continue
        try:
            train = json.loads(train_path.read_text())
            policy = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        record = _local_record(model_dir, project_root, train, policy)
        required = {key.rsplit(".", 1)[-1] for key in record["camera_features"]}
        missing = sorted(required - set(configured_cameras))
        if record["policy_type"] not in SUPPORTED_POLICY_TYPES:
            record["selectable"] = False
            record["unavailable_reason"] = f"Unsupported policy type: {record['policy_type']}"
        elif missing:
            record["selectable"] = False
            record["unavailable_reason"] = f"Missing configured cameras: {', '.join(missing)}"
        records.append(record)
    return records


def _strip_model_prefix(name: str) -> str:
    lowered = name.lower()
    for prefix in MODEL_PREFIXES:
        if lowered.startswith(prefix):
            return name[len(prefix):]
    return name


def _match_dataset(model_name: str, dataset_ids: list[str]) -> str | None:
    stripped = _strip_model_prefix(model_name)
    best: tuple[int, str] | None = None
    for dataset_id in dataset_ids:
        dataset_name = dataset_id.rsplit("/", 1)[-1]
        if stripped == dataset_name:
            return dataset_id
        if dataset_name in model_name:
            score = len(dataset_name)
        elif stripped and stripped in dataset_name:
            score = len(stripped)
        else:
            continue
        if best is None or score > best[0]:
            best = (score, dataset_id)
    return best[1] if best and best[0] >= 2 else None


def load_remote_models(
    models: list[dict],
    dataset_ids: list[str],
    task_loader: Callable[[str], list[str]],
) -> list[dict]:
    records = []
    for model in models:
        repo_id = str(model.get("id", ""))
        name = str(model.get("name") or repo_id.rsplit("/", 1)[-1])
        dataset_id = _match_dataset(name, dataset_ids)
        tasks = []
        if dataset_id:
            try:
                tasks = _dedupe(task_loader(dataset_id))
            except Exception:
                tasks = []
        records.append({
            "id": name,
            "repo_id": repo_id or None,
            "display_name": name,
            "policy_type": None,
            "downloaded": False,
            "local_path": None,
            "dataset_id": dataset_id,
            "tasks": tasks,
            "camera_features": [],
            "selectable": False,
            "unavailable_reason": "Download required",
        })
    return records


def merge_model_records(local: list[dict], remote: list[dict]) -> list[dict]:
    merged = {record["id"]: dict(record) for record in remote}
    for local_record in local:
        record = dict(local_record)
        remote_record = merged.get(record["id"])
        if remote_record and remote_record.get("repo_id"):
            record["repo_id"] = remote_record["repo_id"]
        merged[record["id"]] = record
    return [merged[key] for key in sorted(merged, key=str.casefold)]


def get_model_record(records: list[dict], model_id: str) -> dict | None:
    exact = next((record for record in records if record.get("id") == model_id), None)
    if exact is not None:
        return exact
    by_repo = next((record for record in records if record.get("repo_id") == model_id), None)
    if by_repo is not None:
        return by_repo
    # Try matching by short name (e.g., "user/model_name" → "model_name")
    short = model_id.rsplit("/", 1)[-1] if "/" in model_id else None
    if short:
        return next((record for record in records if record.get("id") == short), None)
    return None
