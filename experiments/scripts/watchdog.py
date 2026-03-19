#!/usr/bin/env python3
"""Watch a remote Parameter Golf run and stop it on obvious failures."""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from typing import Any

from launch_runtime import load_manifest, now_iso, remote_log_snapshot, save_manifest, terminate_remote_run

TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+train_loss:(?P<loss>[-+.\deEinfnaINFNA]+)\s+"
    r"train_time:(?P<train_ms>[-+.\deE]+)ms\s+step_avg:(?P<step_avg_ms>[-+.\deE]+)ms"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[-+.\deE]+)\s+val_bpb:(?P<val_bpb>[-+.\deE]+)"
)
OOM_MARKERS = ("out of memory", "cuda out of memory")
NCCL_MARKERS = ("nccl error", "nccl fatal", "unhandled system error", "ncclinternalerror")


def _parse_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _visible(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if not key.startswith("_")}


def _stall_timeout_seconds(step_avg_seconds: float, train_log_every: int) -> float:
    expected_log_gap = step_avg_seconds * max(train_log_every, 1)
    return max(60.0, expected_log_gap * 3.0)


def evaluate_snapshot(
    snapshot: dict[str, Any],
    *,
    now_epoch: float | None = None,
    train_log_every: int = 1,
) -> dict[str, Any]:
    now_epoch = now_epoch or time.time()
    lines = [str(line) for line in snapshot.get("lines", []) if str(line).strip()]
    last_train_step: int | None = None
    last_train_loss: float | None = None
    step_avg_seconds: float | None = None
    val_bpbs: list[float] = []

    for raw_line in lines:
        lower = raw_line.lower()
        if any(marker in lower for marker in OOM_MARKERS):
            return {"triggered": True, "failure_reason": "cuda_oom"}
        if any(marker in lower for marker in NCCL_MARKERS):
            return {"triggered": True, "failure_reason": "nccl_fatal"}
        if "train_loss:" in lower:
            match = TRAIN_RE.search(raw_line)
            if not match:
                if "train_loss:nan" in lower or "train_loss:inf" in lower:
                    return {"triggered": True, "failure_reason": "invalid_train_loss"}
                continue
            loss = _parse_float(match.group("loss"))
            if loss is None or not math.isfinite(loss):
                return {"triggered": True, "failure_reason": "invalid_train_loss"}
            step = int(match.group("step"))
            if step > 100 and loss > 10.0:
                return {"triggered": True, "failure_reason": "diverging_train_loss"}
            last_train_step = step
            last_train_loss = loss
            step_avg = _parse_float(match.group("step_avg_ms"))
            if step_avg is not None and step_avg > 0:
                step_avg_seconds = step_avg / 1000.0
        if "val_bpb:" in lower:
            match = VAL_RE.search(raw_line)
            if match:
                val_bpb = _parse_float(match.group("val_bpb"))
                if val_bpb is not None:
                    val_bpbs.append(val_bpb)

    if len(val_bpbs) >= 4:
        tail = val_bpbs[-4:]
        if tail[0] < tail[1] < tail[2] < tail[3]:
            return {"triggered": True, "failure_reason": "regressing_val_bpb"}

    mtime_epoch = snapshot.get("mtime_epoch")
    if mtime_epoch is not None and step_avg_seconds is not None:
        staleness_seconds = max(0.0, now_epoch - float(mtime_epoch))
        stall_timeout_seconds = _stall_timeout_seconds(step_avg_seconds, train_log_every)
        if staleness_seconds > stall_timeout_seconds:
            return {
                "triggered": True,
                "failure_reason": "stalled",
                "staleness_seconds": round(staleness_seconds, 1),
                "stall_timeout_seconds": round(stall_timeout_seconds, 1),
            }

    return {
        "triggered": False,
        "latest_step": last_train_step,
        "latest_train_loss": last_train_loss,
        "last_log_update_epoch": mtime_epoch,
    }


def check_watchdog(manifest: dict[str, Any]) -> dict[str, Any]:
    snapshot = remote_log_snapshot(manifest, lines=50)
    resolved_env = manifest.get("resolved_env", {})
    try:
        train_log_every = int(resolved_env.get("TRAIN_LOG_EVERY", 200))
    except (TypeError, ValueError):
        train_log_every = 200
    outcome = evaluate_snapshot(snapshot, train_log_every=train_log_every)
    manifest["last_log_line"] = snapshot.get("last_line")
    manifest["last_log_update_epoch"] = outcome.get("last_log_update_epoch") or snapshot.get("mtime_epoch")
    manifest["estimated_cost"] = manifest.get("estimated_cost")
    if outcome.get("triggered"):
        reason = str(outcome["failure_reason"])
        manifest["failure_reason"] = reason
        manifest["status"] = "failed"
        manifest["watchdog_triggered_at"] = now_iso()
        save_manifest(_visible(manifest))
        terminate_remote_run(manifest, failure_reason=reason)
        return outcome
    save_manifest(_visible(manifest))
    return outcome


def monitor_run(run_id: str, *, poll_seconds: int = 15) -> int:
    manifest = load_manifest(run_id)
    if manifest is None:
        print(f"Unknown run_id: {run_id}", file=sys.stderr)
        return 1
    while True:
        outcome = check_watchdog(manifest)
        if outcome.get("triggered"):
            print(f"{run_id}: killed by watchdog ({outcome['failure_reason']})")
            return 0
        state = manifest.get("status")
        if state in {"success", "failed"}:
            return 0
        time.sleep(max(1, poll_seconds))
        manifest = load_manifest(run_id) or manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor a running experiment and kill it on bad signals")
    parser.add_argument("run_id")
    parser.add_argument("--poll-seconds", type=int, default=15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return monitor_run(args.run_id, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
