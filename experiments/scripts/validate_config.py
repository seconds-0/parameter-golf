#!/usr/bin/env python3
"""Validate Parameter Golf experiment configs before launching runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config_utils import resolve_path, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Parameter Golf experiment config")
    parser.add_argument("config", help="Path to a YAML config")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of text")
    return parser


def text_report(config_path: Path, result) -> int:
    print(f"Config: {config_path}")
    print(f"Runs:   {len(result.runs)}")
    for index, run in enumerate(result.runs[:5], start=1):
        preview = ", ".join(f"{key}={value}" for key, value in sorted(run.combo.items()))
        print(f"  {index}. {run.label}" + (f" [{preview}]" if preview else ""))
    if len(result.runs) > 5:
        print(f"  ... {len(result.runs) - 5} more runs")
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    print("\nValidation passed with warnings." if result.warnings else "\nValidation passed.")
    return 0


def json_report(config_path: Path, result) -> int:
    payload = {
        "config": str(config_path),
        "ok": result.ok,
        "warnings": result.warnings,
        "errors": result.errors,
        "runs": [{"label": run.label, "env": run.env, "combo": run.combo} for run in result.runs],
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.ok else 1


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve_path(args.config)
    result = validate_config(config_path)
    if args.json:
        return json_report(config_path, result)
    return text_report(config_path, result)


if __name__ == "__main__":
    sys.exit(main())
