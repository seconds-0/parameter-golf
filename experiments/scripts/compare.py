#!/usr/bin/env python3
"""Compare Parameter Golf experiment results side-by-side."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_number(value: float | int | None, width: int, precision: int = 4, *, star: bool = False) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    text = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    if star:
        text += "*"
    return f"{text:>{width}}"


def main():
    if len(sys.argv) < 2:
        print("Usage: compare.py <metrics1.json> [metrics2.json ...]", file=sys.stderr)
        print("       compare.py experiments/results/*/metrics.json", file=sys.stderr)
        sys.exit(1)

    runs = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        data = load_metrics(path)
        run_id = data.get("run_id") or path.parent.name
        final = data.get("final", {})
        config = data.get("config", {})
        runs.append({
            "run_id": run_id,
            "val_bpb": final.get("val_bpb"),
            "val_loss": final.get("val_loss"),
            "steps": final.get("stop_step") or (data["train_steps"][-1]["step"] if data.get("train_steps") else None),
            "time_s": (data["train_steps"][-1]["train_time_ms"] / 1000) if data.get("train_steps") else None,
            "model_mb": final.get("total_submission_bytes", 0) / 1_000_000,
            "peak_mem": final.get("peak_memory_allocated_mib"),
            "params": config.get("model_params"),
            "host": data.get("host"),
            "cost": data.get("cost"),
            "status": data.get("status", "failed"),
            "group": data.get("group"),
            "hypothesis_id": data.get("hypothesis_id"),
            "notes": data.get("notes"),
        })

    if not runs:
        print("No runs to compare.", file=sys.stderr)
        sys.exit(1)

    # Sort by val_bpb (best first)
    runs.sort(key=lambda r: r["val_bpb"] if r["val_bpb"] is not None else float("inf"))

    # Find best
    best_bpb = runs[0]["val_bpb"] if runs[0]["val_bpb"] is not None else None

    # Print table
    hdr = (
        f"{'Run ID':<30} | {'Group':<14} | {'Hypothesis':<16} | {'Host':<12} | {'Status':<7} | "
        f"{'val_bpb':>9} | {'val_loss':>8} | {'Cost($)':>7} | {'Steps':>6} | {'Time(s)':>7} | {'Notes':<18}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in runs:
        best = r["val_bpb"] == best_bpb and best_bpb is not None
        host = str(r["host"] or "N/A")[:12]
        group = str(r["group"] or "N/A")[:14]
        hypothesis = str(r["hypothesis_id"] or "N/A")[:16]
        notes = str(r["notes"] or "")[:18]
        cost = fmt_number(r["cost"], 7, 2)
        time_s = fmt_number(r["time_s"], 7, 1)
        print(
            f"{r['run_id']:<30} | {group:<14} | {hypothesis:<16} | {host:<12} | {str(r['status']):<7} | "
            f"{fmt_number(r['val_bpb'], 9, 4, star=best)} | {fmt_number(r['val_loss'], 8, 4)} | "
            f"{cost} | {fmt_number(r['steps'], 6)} | {time_s} | {notes:<18}"
        )


if __name__ == "__main__":
    main()
