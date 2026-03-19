#!/usr/bin/env python3
"""Parse Parameter Golf plain-text training logs into structured JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUMBER_RE = r"([\d.]+(?:[eE][+-]?\d+)?)"


def parse_kv(line: str) -> dict[str, str]:
    """Extract key:value pairs from a log line."""
    return dict(re.findall(r"(\w+):([\d.eE+-]+|True|False|[^\s:]+)", line))


def load_manifest(log_path: Path) -> dict[str, object]:
    manifest_path = log_path.with_name("manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def parse_log(text: str) -> dict[str, object]:
    result = {
        "config": {},
        "train_steps": [],
        "val_steps": [],
        "final": {},
    }
    saw_exact_roundtrip = False

    for line in text.splitlines():
        line = line.strip()

        # Skip noise (NCCL warnings, rank messages, warmup, code dump)
        if not line or line.startswith("[") or line.startswith("W0") or line.startswith("NCCL"):
            continue

        # Train step: step:N/T train_loss:F train_time:Fms step_avg:Fms
        m = re.match(
            rf"step:(\d+)/(\d+)\s+train_loss:{NUMBER_RE}\s+train_time:{NUMBER_RE}ms\s+step_avg:{NUMBER_RE}ms(?:\s+tok_s:{NUMBER_RE})?",
            line,
        )
        if m:
            entry = {
                "step": int(m.group(1)),
                "total_steps": int(m.group(2)),
                "train_loss": float(m.group(3)),
                "train_time_ms": float(m.group(4)),
                "step_avg_ms": float(m.group(5)),
            }
            if m.group(6):
                entry["tok_s"] = float(m.group(6))
            result["train_steps"].append(entry)
            continue

        # Val step: step:N/T val_loss:F val_bpb:F train_time:Fms step_avg:Fms
        m = re.match(
            rf"step:(\d+)/(\d+)\s+val_loss:{NUMBER_RE}\s+val_bpb:{NUMBER_RE}\s+train_time:{NUMBER_RE}ms\s+step_avg:{NUMBER_RE}ms",
            line,
        )
        if m:
            result["val_steps"].append({
                "step": int(m.group(1)),
                "total_steps": int(m.group(2)),
                "val_loss": float(m.group(3)),
                "val_bpb": float(m.group(4)),
                "train_time_ms": float(m.group(5)),
                "step_avg_ms": float(m.group(6)),
            })
            continue

        # Config: model_params:N
        if line.startswith("model_params:"):
            result["config"]["model_params"] = int(parse_kv(line).get("model_params", 0))
            continue

        # Config: world_size:N grad_accum_steps:N
        if line.startswith("world_size:"):
            kv = parse_kv(line)
            result["config"]["world_size"] = int(kv.get("world_size", 1))
            result["config"]["grad_accum_steps"] = int(kv.get("grad_accum_steps", 1))
            continue

        # Config: tie_embeddings:B embed_lr:F ...
        if line.startswith("tie_embeddings:"):
            kv = parse_kv(line)
            for k, v in kv.items():
                if v in ("True", "False"):
                    result["config"][k] = v == "True"
                else:
                    try:
                        result["config"][k] = float(v)
                    except ValueError:
                        result["config"][k] = v
            continue

        # Config: train_batch_tokens:N train_seq_len:N ...
        if line.startswith("train_batch_tokens:"):
            kv = parse_kv(line)
            for k, v in kv.items():
                try:
                    result["config"][k] = float(v) if "." in v else int(v)
                except ValueError:
                    result["config"][k] = v
            continue

        # Config: attention_mode, seed, run_id, etc.
        if any(line.startswith(p) for p in ("attention_mode:", "seed:", "run_id:", "sdp_backends:", "optimizer:")):
            kv = parse_kv(line)
            for k, v in kv.items():
                try:
                    result["config"][k] = float(v) if "." in v else int(v)
                except ValueError:
                    result["config"][k] = v
            continue

        # Early stopping
        if line.startswith("stopping_early:"):
            kv = parse_kv(line)
            result["final"]["stopped_early"] = True
            result["final"]["stop_reason"] = kv.get("stopping_early", "wallclock_cap")
            if "step" in kv:
                result["final"]["stop_step"] = int(kv["step"].split("/")[0])
            continue

        # Peak memory
        m = re.search(r"peak memory allocated:\s*(\d+)\s*MiB\s*reserved:\s*(\d+)\s*MiB", line)
        if m:
            result["final"]["peak_memory_allocated_mib"] = int(m.group(1))
            result["final"]["peak_memory_reserved_mib"] = int(m.group(2))
            continue

        # Serialized model sizes
        m = re.match(r"Serialized model:\s*(\d+)\s*bytes", line)
        if m:
            result["final"]["model_bytes_raw"] = int(m.group(1))
            continue

        m = re.match(r"Serialized model int8\+zlib:\s*(\d+)\s*bytes", line)
        if m:
            result["final"]["model_bytes_int8_zlib"] = int(m.group(1))
            continue

        m = re.match(r"Total submission size int8\+zlib:\s*(\d+)\s*bytes", line)
        if m:
            result["final"]["total_submission_bytes"] = int(m.group(1))
            continue

        m = re.match(r"Code size:\s*(\d+)\s*bytes", line)
        if m:
            result["final"]["code_bytes"] = int(m.group(1))
            continue

        # Final roundtrip exact
        m = re.match(
            rf"final_int8_zlib_roundtrip_exact\s+val_loss:{NUMBER_RE}\s+val_bpb:{NUMBER_RE}",
            line,
        )
        if m:
            saw_exact_roundtrip = True
            result["final"]["val_loss"] = float(m.group(1))
            result["final"]["val_bpb"] = float(m.group(2))
            continue

        # Final roundtrip (non-exact, with eval_time)
        m = re.match(
            rf"final_int8_zlib_roundtrip\s+val_loss:{NUMBER_RE}\s+val_bpb:{NUMBER_RE}\s+eval_time:(\d+)ms",
            line,
        )
        if m:
            result["final"]["roundtrip_eval_time_ms"] = int(m.group(3))
            # Don't overwrite exact values if already set
            if "val_loss" not in result["final"]:
                result["final"]["val_loss"] = float(m.group(1))
                result["final"]["val_bpb"] = float(m.group(2))
            continue

    # Derive run_id from config if available
    if "run_id" in result["config"]:
        result["run_id"] = result["config"].pop("run_id")
    result["status"] = "success" if saw_exact_roundtrip else "failed"

    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse_log.py <train.log> [output.json]", file=sys.stderr)
        sys.exit(1)

    log_path = Path(sys.argv[1])
    text = log_path.read_text(encoding="utf-8")
    parsed = parse_log(text)
    manifest = load_manifest(log_path)
    if manifest:
        if "host_name" in manifest or "host" in manifest:
            parsed["host"] = manifest.get("host_name") or manifest.get("host")
        if "estimated_cost" in manifest:
            parsed["cost"] = manifest.get("estimated_cost")
        if "exit_code" in manifest:
            parsed["exit_code"] = manifest.get("exit_code")
        for field_name in ("hypothesis_id", "group", "notes", "failure_reason"):
            if field_name in manifest:
                parsed[field_name] = manifest.get(field_name)

    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
        out_path.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}", file=sys.stderr)
    else:
        print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
