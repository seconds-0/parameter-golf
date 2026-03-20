#!/usr/bin/env python3
"""Compare Parameter Golf experiment results side-by-side."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def replay_delta(final: dict, key: str, val_key: str, ref_key: str) -> float | None:
    if final.get(key) is not None:
        return final.get(key)
    value = final.get(val_key)
    reference = final.get(ref_key)
    if value is None or reference is None:
        return None
    return value - reference


def fmt_number(value: float | int | None, width: int, precision: int = 4, *, star: bool = False) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    text = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    if star:
        text += "*"
    return f"{text:>{width}}"


def fmt_delta(value: float | None, width: int, precision: int = 4) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:+.{precision}f}".rjust(width)


def fmt_compact(value: float | int | None, width: int, precision: int = 2) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    suffix = ""
    number = float(value)
    if abs(number) >= 1_000_000:
        number /= 1_000_000
        suffix = "M"
    elif abs(number) >= 1_000:
        number /= 1_000
        suffix = "K"
    text = f"{number:.{precision}f}{suffix}"
    return f"{text:>{width}}"


def fmt_percent(value: float | None, width: int, precision: int = 1) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{100.0 * value:.{precision}f}%".rjust(width)


def main():
    if len(sys.argv) < 2:
        print("Usage: compare.py <metrics1.json> [metrics2.json ...]", file=sys.stderr)
        print("       compare.py experiments/results/*/metrics.json", file=sys.stderr)
        sys.exit(1)

    runs = []
    reference_post_bpb: float | None = None
    reference_qgap_bpb: float | None = None
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        data = load_metrics(path)
        run_id = data.get("run_id") or path.parent.name
        final = data.get("final", {})
        config = data.get("config", {})
        run = {
            "run_id": run_id,
            "postquant_val_bpb": final.get("val_bpb"),
            "postquant_val_loss": final.get("val_loss"),
            "prequant_val_bpb": final.get("prequant_val_bpb"),
            "qgap_bpb": final.get("qgap_bpb"),
            "uncompiled_delta_bpb": replay_delta(
                final, "uncompiled_check_delta_bpb", "uncompiled_check_val_bpb", "prequant_val_bpb"
            ),
            "reloaded_postquant_delta_bpb": replay_delta(
                final, "reloaded_postquant_delta_bpb", "reloaded_postquant_val_bpb", "postquant_val_bpb"
            ),
            "steps": final.get("stop_step") or (data["train_steps"][-1]["step"] if data.get("train_steps") else None),
            "time_s": (data["train_steps"][-1]["train_time_ms"] / 1000) if data.get("train_steps") else None,
            "step_avg_ms": (data["train_steps"][-1]["step_avg_ms"] if data.get("train_steps") else None),
            "tok_s": data.get("tok_s"),
            "supervised_target_fraction": final.get("supervised_target_fraction"),
            "slack_bytes": final.get("artifact_slack_bytes"),
            "peak_mem": final.get("peak_memory_allocated_mib"),
            "params": config.get("model_params"),
            "host": data.get("host"),
            "cost": data.get("cost"),
            "status": data.get("status", "failed"),
            "group": data.get("group"),
            "hypothesis_id": data.get("hypothesis_id"),
            "notes": data.get("notes"),
        }
        if reference_post_bpb is None:
            reference_post_bpb = run["postquant_val_bpb"]
            reference_qgap_bpb = run["qgap_bpb"]
        run["delta_pq"] = (
            run["postquant_val_bpb"] - reference_post_bpb
            if run["postquant_val_bpb"] is not None and reference_post_bpb is not None
            else None
        )
        run["delta_qgap"] = (
            run["qgap_bpb"] - reference_qgap_bpb
            if run["qgap_bpb"] is not None and reference_qgap_bpb is not None
            else None
        )
        runs.append(run)

    if not runs:
        print("No runs to compare.", file=sys.stderr)
        sys.exit(1)

    # Sort by post-roundtrip val_bpb (best first).
    runs.sort(key=lambda r: r["postquant_val_bpb"] if r["postquant_val_bpb"] is not None else float("inf"))

    # Find best
    best_bpb = runs[0]["postquant_val_bpb"] if runs[0]["postquant_val_bpb"] is not None else None

    # Print table
    hdr = (
        f"{'Run ID':<30} | {'Status':<7} | {'PostBPB':>9} | {'Δpq':>8} | {'PreBPB':>9} | {'qgap':>8} | "
        f"{'EgrΔ':>8} | {'RldΔ':>8} | "
        f"{'StepMs':>7} | {'Tok/s':>7} | {'Sup%':>6} | {'Slack':>7} | {'Cost($)':>7} | {'Host':<12} | {'Group':<12} | {'Hypothesis':<16}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in runs:
        best = r["postquant_val_bpb"] == best_bpb and best_bpb is not None
        host = str(r["host"] or "N/A")[:12]
        group = str(r["group"] or "N/A")[:12]
        hypothesis = str(r["hypothesis_id"] or "N/A")[:16]
        cost = fmt_number(r["cost"], 7, 2)
        print(
            f"{r['run_id']:<30} | {str(r['status']):<7} | "
            f"{fmt_number(r['postquant_val_bpb'], 9, 4, star=best)} | {fmt_delta(r['delta_pq'], 8, 4)} | "
            f"{fmt_number(r['prequant_val_bpb'], 9, 4)} | {fmt_number(r['qgap_bpb'], 8, 4)} | "
            f"{fmt_delta(r['uncompiled_delta_bpb'], 8, 4)} | {fmt_delta(r['reloaded_postquant_delta_bpb'], 8, 4)} | "
            f"{fmt_number(r['step_avg_ms'], 7, 1)} | {fmt_compact(r['tok_s'], 7)} | {fmt_percent(r['supervised_target_fraction'], 6)} | {fmt_compact(r['slack_bytes'], 7)} | "
            f"{cost} | {host:<12} | {group:<12} | {hypothesis:<16}"
        )


if __name__ == "__main__":
    main()
