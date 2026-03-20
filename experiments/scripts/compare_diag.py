#!/usr/bin/env python3
"""Compare structured DIAG lines between two logs and report the first mismatch."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

DIAG_RE = re.compile(r"^DIAG:([^:]+):(.*)$")


def extract_diag_entries(text: str) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = DIAG_RE.match(line)
        if not match:
            continue
        label = match.group(1)
        payload = json.loads(match.group(2))
        entries.append((label, payload))
    return entries


def _find_first_difference(left: Any, right: Any, path: str) -> tuple[str, Any, Any] | None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys():
            child_path = f"{path}.{key}" if path else key
            if key not in right:
                return child_path, left[key], "<missing>"
            diff = _find_first_difference(left[key], right[key], child_path)
            if diff is not None:
                return diff
        for key in right.keys():
            if key not in left:
                child_path = f"{path}.{key}" if path else key
                return child_path, "<missing>", right[key]
        return None
    if isinstance(left, list) and isinstance(right, list):
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            child_path = f"{path}[{idx}]"
            diff = _find_first_difference(left_item, right_item, child_path)
            if diff is not None:
                return diff
        if len(left) != len(right):
            child_path = f"{path}.length" if path else "length"
            return child_path, len(left), len(right)
        return None
    if left != right:
        return path, left, right
    return None


def compare_diag_text(left_text: str, right_text: str) -> dict[str, Any]:
    left_entries = extract_diag_entries(left_text)
    right_entries = extract_diag_entries(right_text)
    right_map = {label: payload for label, payload in right_entries}
    left_labels = {label for label, _payload in left_entries}

    labels_checked: list[str] = []
    for label, left_payload in left_entries:
        labels_checked.append(label)
        if label not in right_map:
            return {
                "labels_checked": labels_checked,
                "missing_label": label,
                "first_diff_path": label,
                "left_value": "<present>",
                "right_value": "<missing>",
            }
        diff = _find_first_difference(left_payload, right_map[label], label)
        if diff is not None:
            path, left_value, right_value = diff
            return {
                "labels_checked": labels_checked,
                "missing_label": None,
                "first_diff_path": path,
                "left_value": left_value,
                "right_value": right_value,
            }

    extra_labels = [label for label, _payload in right_entries if label not in left_labels]
    return {
        "labels_checked": labels_checked,
        "missing_label": extra_labels[0] if extra_labels else None,
        "first_diff_path": extra_labels[0] if extra_labels else None,
        "left_value": "<missing>" if extra_labels else None,
        "right_value": "<present>" if extra_labels else None,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: compare_diag.py <train-or-left.log> <replay-or-right.log>", file=sys.stderr)
        return 1

    left_path = Path(sys.argv[1])
    right_path = Path(sys.argv[2])
    result = compare_diag_text(
        left_path.read_text(encoding="utf-8"),
        right_path.read_text(encoding="utf-8"),
    )

    if result["first_diff_path"] is None and result["missing_label"] is None:
        print("DIAG_MATCH")
        return 0

    if result["missing_label"] is not None and result["first_diff_path"] == result["missing_label"]:
        print(f"DIAG_MISSING_LABEL {result['missing_label']}")
        print(f"LEFT  {result['left_value']}")
        print(f"RIGHT {result['right_value']}")
        return 2

    print(f"FIRST_DIFF_AT {result['first_diff_path']}")
    print(f"LEFT  {result['left_value']}")
    print(f"RIGHT {result['right_value']}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
