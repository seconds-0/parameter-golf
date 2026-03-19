#!/usr/bin/env python3
"""Summarize experiment spend from manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"
MACHINES_FILE = ROOT / "experiments" / "machines.yaml"


def load_manifests() -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("*/manifest.json")):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return manifests


def load_budget_default() -> float | None:
    env_budget = os.environ.get("PGOLF_TOTAL_BUDGET")
    if env_budget:
        try:
            return float(env_budget)
        except ValueError:
            return None
    if not MACHINES_FILE.exists():
        return None
    try:
        payload = yaml.safe_load(MACHINES_FILE.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    value = payload.get("budget_total")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fmt_money(value: float | None) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


def fmt_table(rows: list[tuple[str, str]]) -> str:
    width = max((len(label) for label, _ in rows), default=0)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)


def summarize_costs(manifests: list[dict[str, Any]], *, budget: float | None) -> str:
    now = time.time()
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    total = 0.0
    today_total = 0.0
    running_total = 0.0
    by_host: dict[str, float] = defaultdict(float)
    by_status: dict[str, float] = defaultdict(float)
    by_date: dict[str, float] = defaultdict(float)

    for manifest in manifests:
        try:
            cost = float(manifest.get("estimated_cost") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        total += cost
        status = str(manifest.get("status") or "unknown")
        host = str(manifest.get("host_name") or manifest.get("host") or "unknown")
        date = str(manifest.get("start_time") or "")[:10] or "unknown"
        by_host[host] += cost
        by_status[status] += cost
        by_date[date] += cost
        if date == today:
            today_total += cost
        if status == "running":
            running_total += cost

    rows = [
        ("Total spend", fmt_money(total)),
        ("Today's spend", fmt_money(today_total)),
        ("Running costs", fmt_money(running_total)),
    ]
    if budget is not None:
        rows.append(("Remaining budget", fmt_money(max(budget - total, 0.0))))

    sections = ["Summary", fmt_table(rows), "", "By host"]
    host_rows = [(host, fmt_money(cost)) for host, cost in sorted(by_host.items())]
    sections.append(fmt_table(host_rows) if host_rows else "No manifests found.")
    sections.extend(["", "By status", fmt_table([(status, fmt_money(cost)) for status, cost in sorted(by_status.items())]) or "No data"])
    sections.extend(["", "By date", fmt_table([(date, fmt_money(cost)) for date, cost in sorted(by_date.items())]) or "No data"])
    return "\n".join(sections).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize experiment spend from local manifests")
    parser.add_argument("--budget", type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifests = load_manifests()
    budget = args.budget if args.budget is not None else load_budget_default()
    print(summarize_costs(manifests, budget=budget))
    return 0


if __name__ == "__main__":
    sys.exit(main())
