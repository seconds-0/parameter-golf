#!/usr/bin/env python3
"""Remote cleanup for old Parameter Golf temp directories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from config_utils import RESULTS_DIR, resolve_machine


def parse_age(value: str) -> float:
    text = value.strip().lower()
    if text.endswith("h"):
        return float(text[:-1]) * 3600.0
    if text.endswith("d"):
        return float(text[:-1]) * 86400.0
    if text.endswith("m"):
        return float(text[:-1]) * 60.0
    return float(text) * 3600.0


def ssh_capture(host: str, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ssh", host, command], text=True, capture_output=True, check=False)


def is_collected(run_id: str) -> bool:
    result_dir = RESULTS_DIR / run_id
    if not (result_dir / "manifest.json").exists():
        return False
    return any((result_dir / name).exists() for name in ("train.log", "launcher.stdout", "metrics.json", "train_gpt.py"))


def list_remote_dirs(host_or_machine: str) -> tuple[str, list[dict[str, Any]]]:
    machine_name, machine, errors = resolve_machine(host_or_machine)
    if errors or machine is None:
        raise RuntimeError("; ".join(errors))
    host = str(machine["host"])
    command = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        "import time\n"
        "rows = []\n"
        "for path in sorted(Path('/tmp').glob('pgolf_*')):\n"
        "    try:\n"
        "        stat = path.stat()\n"
        "    except OSError:\n"
        "        continue\n"
        "    rows.append({'path': str(path), 'mtime': stat.st_mtime, 'age_seconds': time.time() - stat.st_mtime})\n"
        "print(json.dumps(rows))\n"
        "PY"
    )
    proc = ssh_capture(host, command)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "failed to list remote directories")
    try:
        rows = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid remote response: {exc}") from exc
    return host, rows


def run_gc(host_or_machine: str, *, older_than: str, dry_run: bool) -> dict[str, Any]:
    host, rows = list_remote_dirs(host_or_machine)
    threshold = parse_age(older_than)
    deletable: list[dict[str, Any]] = []
    for row in rows:
        path = str(row["path"])
        run_id = Path(path).name.removeprefix("pgolf_")
        if float(row.get("age_seconds") or 0.0) < threshold:
            continue
        if not is_collected(run_id):
            continue
        deletable.append({**row, "run_id": run_id})
    if deletable and not dry_run:
        joined = " ".join(f"'{entry['path']}'" for entry in deletable)
        proc = ssh_capture(host, f"rm -rf {joined}")
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "remote delete failed")
    return {"host": host, "all_rows": rows, "deletable": deletable, "dry_run": dry_run}


def print_report(payload: dict[str, Any]) -> None:
    print(f"Host: {payload['host']}")
    if not payload["all_rows"]:
        print("No /tmp/pgolf_* directories found.")
        return
    print(f"{'Path':<50} {'Age(h)':>7} {'Collected':>10} {'Delete':>8}")
    print("-" * 80)
    deletable_paths = {entry["path"] for entry in payload["deletable"]}
    for row in payload["all_rows"]:
        path = str(row["path"])
        run_id = Path(path).name.removeprefix("pgolf_")
        age_hours = float(row.get("age_seconds") or 0.0) / 3600.0
        collected = "yes" if is_collected(run_id) else "no"
        delete = "yes" if path in deletable_paths else "no"
        print(f"{path[:50]:<50} {age_hours:>7.1f} {collected:>10} {delete:>8}")
    action = "Would delete" if payload["dry_run"] else "Deleted"
    print(f"{action} {len(payload['deletable'])} directories.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Delete old remote /tmp/pgolf_* directories")
    parser.add_argument("host")
    parser.add_argument("--older-than", default="24h")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = run_gc(args.host, older_than=args.older_than, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print_report(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
