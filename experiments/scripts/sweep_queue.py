#!/usr/bin/env python3
"""Persistent sweep queue helpers."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config_utils import RESULTS_DIR, now_iso, slugify, unique_run_id
from launch_runtime import load_manifest, save_manifest, wait_for_run

SWEEPS_DIR = RESULTS_DIR / "_sweeps"
_LOCK = threading.Lock()


def queue_path(sweep_id: str) -> Path:
    return SWEEPS_DIR / sweep_id / "queue.json"


def new_sweep_id(config_file: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slugify(Path(config_file).stem)}-{stamp}"


def create_queue(config_file: str, runs: list[Any]) -> dict[str, Any]:
    sweep_id = new_sweep_id(config_file)
    payload = {
        "sweep_id": sweep_id,
        "config_file": config_file,
        "created": now_iso(),
        "runs": [
            {
                "label": run.label,
                "env": run.env,
                "metadata": run.metadata,
                "status": "pending",
                "host": None,
                "run_id": None,
            }
            for run in runs
        ],
    }
    save_queue(payload)
    return payload


def load_queue(sweep_id: str) -> dict[str, Any]:
    path = queue_path(sweep_id)
    if not path.exists():
        raise FileNotFoundError(f"Unknown sweep_id: {sweep_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_queue(payload: dict[str, Any]) -> Path:
    path = queue_path(str(payload["sweep_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


@contextmanager
def locked_queue(sweep_id: str) -> Iterator[dict[str, Any]]:
    with _LOCK:
        payload = load_queue(sweep_id)
        yield payload
        save_queue(payload)


def claim_next_run(sweep_id: str, host: str) -> dict[str, Any] | None:
    with locked_queue(sweep_id) as payload:
        for run in payload["runs"]:
            if run["status"] != "pending":
                continue
            run["status"] = "running"
            run["host"] = host
            run["run_id"] = unique_run_id(run["label"])
            return dict(run)
    return None


def sync_entry_from_manifest(sweep_id: str, run_id: str) -> dict[str, Any] | None:
    manifest = load_manifest(run_id)
    with locked_queue(sweep_id) as payload:
        for run in payload["runs"]:
            if run.get("run_id") != run_id:
                continue
            if manifest is None:
                run["status"] = "lost"
                return dict(run)
            run["status"] = str(manifest.get("status") or run["status"])
            run["failure_reason"] = manifest.get("failure_reason")
            run["estimated_cost"] = manifest.get("estimated_cost")
            return dict(run)
    return None


def running_entries_for_host(sweep_id: str, host: str) -> list[dict[str, Any]]:
    payload = load_queue(sweep_id)
    return [dict(run) for run in payload["runs"] if run.get("status") == "running" and run.get("host") == host]


def reconcile_running_entry(sweep_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    run_id = str(entry.get("run_id") or "")
    manifest = load_manifest(run_id)
    if manifest is None:
        return sync_entry_from_manifest(sweep_id, run_id) or {**entry, "status": "lost"}
    if manifest.get("status") == "running":
        wait_for_run(manifest, summarize=False)
    updated_manifest = load_manifest(run_id)
    if updated_manifest is not None:
        save_manifest(updated_manifest)
    return sync_entry_from_manifest(sweep_id, run_id) or entry


def all_runs_complete_for_host(sweep_id: str, host: str) -> bool:
    payload = load_queue(sweep_id)
    return all(run["status"] in {"success", "failed", "lost"} for run in payload["runs"] if run.get("host") == host)


def sweep_done(sweep_id: str) -> bool:
    payload = load_queue(sweep_id)
    return all(run["status"] in {"success", "failed", "lost"} for run in payload["runs"])


def requeue_runs(sweep_id: str, *, failed: bool, lost: bool) -> int:
    statuses = set()
    if failed:
        statuses.add("failed")
    if lost:
        statuses.add("lost")
    if not statuses:
        statuses.update({"failed", "lost"})
    count = 0
    with locked_queue(sweep_id) as payload:
        for run in payload["runs"]:
            if run["status"] not in statuses:
                continue
            run["status"] = "pending"
            run["host"] = None
            run["run_id"] = None
            run.pop("failure_reason", None)
            run.pop("estimated_cost", None)
            count += 1
    return count
