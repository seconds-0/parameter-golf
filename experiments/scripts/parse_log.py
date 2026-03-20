#!/usr/bin/env python3
"""Parse Parameter Golf plain-text training logs into structured JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUMBER_RE = r"([\d.]+(?:[eE][+-]?\d+)?)"
EXACT_METRIC_FIELDS = {
    "uncompiled_check": "uncompiled_check",
    "reloaded_prequant_exact": "reloaded_prequant",
    "reloaded_int8_zlib_roundtrip_exact": "reloaded_postquant",
}


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


def parse_exact_eval(prefix: str, line: str) -> dict[str, float]:
    m = re.match(rf"{prefix}\s+val_loss:{NUMBER_RE}\s+val_bpb:{NUMBER_RE}", line)
    if not m:
        return {}
    values: dict[str, float] = {
        "val_loss": float(m.group(1)),
        "val_bpb": float(m.group(2)),
    }
    kv = parse_kv(line)
    if "delta_loss" in kv:
        values["delta_loss"] = float(kv["delta_loss"])
    if "delta_bpb" in kv:
        values["delta_bpb"] = float(kv["delta_bpb"])
    if "eval_time" in kv and kv["eval_time"].endswith("ms"):
        values["eval_time_ms"] = float(kv["eval_time"][:-2])
    elif "eval_time" in kv:
        values["eval_time_ms"] = float(kv["eval_time"])
    return values


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
            kv = parse_kv(line)
            if m.group(6):
                entry["tok_s"] = float(m.group(6))
                result["tok_s"] = entry["tok_s"]
            if "train_tokens_seen" in kv:
                entry["train_tokens_seen"] = int(float(kv["train_tokens_seen"]))
                result["train_tokens_seen"] = entry["train_tokens_seen"]
            if "train_supervised_tokens_seen" in kv:
                entry["train_supervised_tokens_seen"] = int(float(kv["train_supervised_tokens_seen"]))
                result["train_supervised_tokens_seen"] = entry["train_supervised_tokens_seen"]
            if "ignored_target_tokens_seen" in kv:
                entry["ignored_target_tokens_seen"] = int(float(kv["ignored_target_tokens_seen"]))
                result["ignored_target_tokens_seen"] = entry["ignored_target_tokens_seen"]
            if "supervised_target_fraction" in kv:
                entry["supervised_target_fraction"] = float(kv["supervised_target_fraction"])
                result["supervised_target_fraction"] = entry["supervised_target_fraction"]
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
        if any(
            line.startswith(p)
            for p in ("attention_mode:", "seed:", "run_id:", "sdp_backends:", "optimizer:", "doc_aligned_batching:")
        ):
            kv = parse_kv(line)
            for k, v in kv.items():
                if v in ("True", "False"):
                    result["config"][k] = v == "True"
                else:
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

        if line.startswith("train_tokens_seen:"):
            kv = parse_kv(line)
            if "train_tokens_seen" in kv:
                value = int(float(kv["train_tokens_seen"]))
                result["train_tokens_seen"] = value
                result["final"]["train_tokens_seen"] = value
            if "train_supervised_tokens_seen" in kv:
                value = int(float(kv["train_supervised_tokens_seen"]))
                result["train_supervised_tokens_seen"] = value
                result["final"]["train_supervised_tokens_seen"] = value
            if "ignored_target_tokens_seen" in kv:
                value = int(float(kv["ignored_target_tokens_seen"]))
                result["ignored_target_tokens_seen"] = value
                result["final"]["ignored_target_tokens_seen"] = value
            if "supervised_target_fraction" in kv:
                value = float(kv["supervised_target_fraction"])
                result["supervised_target_fraction"] = value
                result["final"]["supervised_target_fraction"] = value
            if "tok_s" in kv:
                result["tok_s"] = float(kv["tok_s"])
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
            payload_match = re.search(r"payload:(\d+)\s+raw_torch:(\d+)\s+payload_ratio:([\d.]+)x", line)
            if payload_match:
                result["final"]["payload_bytes"] = int(payload_match.group(1))
                result["final"]["payload_raw_torch_bytes"] = int(payload_match.group(2))
                result["final"]["payload_ratio"] = float(payload_match.group(3))
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
            rf"final_prequant_exact\s+val_loss:{NUMBER_RE}\s+val_bpb:{NUMBER_RE}",
            line,
        )
        if m:
            result["final"]["prequant_val_loss"] = float(m.group(1))
            result["final"]["prequant_val_bpb"] = float(m.group(2))
            kv = parse_kv(line)
            if "train_tokens_seen" in kv:
                value = int(float(kv["train_tokens_seen"]))
                result["train_tokens_seen"] = value
                result["final"]["train_tokens_seen"] = value
            if "train_supervised_tokens_seen" in kv:
                value = int(float(kv["train_supervised_tokens_seen"]))
                result["train_supervised_tokens_seen"] = value
                result["final"]["train_supervised_tokens_seen"] = value
            if "ignored_target_tokens_seen" in kv:
                value = int(float(kv["ignored_target_tokens_seen"]))
                result["ignored_target_tokens_seen"] = value
                result["final"]["ignored_target_tokens_seen"] = value
            if "supervised_target_fraction" in kv:
                value = float(kv["supervised_target_fraction"])
                result["supervised_target_fraction"] = value
                result["final"]["supervised_target_fraction"] = value
            continue

        for prefix, field_prefix in EXACT_METRIC_FIELDS.items():
            parsed_exact = parse_exact_eval(prefix, line)
            if parsed_exact:
                result["final"][f"{field_prefix}_val_loss"] = parsed_exact["val_loss"]
                result["final"][f"{field_prefix}_val_bpb"] = parsed_exact["val_bpb"]
                if "delta_loss" in parsed_exact:
                    result["final"][f"{field_prefix}_delta_loss"] = parsed_exact["delta_loss"]
                if "delta_bpb" in parsed_exact:
                    result["final"][f"{field_prefix}_delta_bpb"] = parsed_exact["delta_bpb"]
                if "eval_time_ms" in parsed_exact:
                    result["final"][f"{field_prefix}_eval_time_ms"] = int(parsed_exact["eval_time_ms"])
                break
        else:
            m = re.match(
                rf"final_int8_zlib_roundtrip_exact\s+val_loss:{NUMBER_RE}\s+val_bpb:{NUMBER_RE}",
                line,
            )
            if m:
                saw_exact_roundtrip = True
                result["final"]["postquant_val_loss"] = float(m.group(1))
                result["final"]["postquant_val_bpb"] = float(m.group(2))
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
                    result["final"]["postquant_val_loss"] = float(m.group(1))
                    result["final"]["postquant_val_bpb"] = float(m.group(2))
                    result["final"]["val_loss"] = float(m.group(1))
                    result["final"]["val_bpb"] = float(m.group(2))
                continue

            m = re.match(
                rf"quantization_delta_exact\s+val_loss:{NUMBER_RE}\s+val_bpb:{NUMBER_RE}",
                line,
            )
            if m:
                result["final"]["qgap_loss"] = float(m.group(1))
                result["final"]["qgap_bpb"] = float(m.group(2))
                kv = parse_kv(line)
                if "train_tokens_seen" in kv:
                    value = int(float(kv["train_tokens_seen"]))
                    result["train_tokens_seen"] = value
                    result["final"]["train_tokens_seen"] = value
                if "train_supervised_tokens_seen" in kv:
                    value = int(float(kv["train_supervised_tokens_seen"]))
                    result["train_supervised_tokens_seen"] = value
                    result["final"]["train_supervised_tokens_seen"] = value
                if "ignored_target_tokens_seen" in kv:
                    value = int(float(kv["ignored_target_tokens_seen"]))
                    result["ignored_target_tokens_seen"] = value
                    result["final"]["ignored_target_tokens_seen"] = value
                if "supervised_target_fraction" in kv:
                    value = float(kv["supervised_target_fraction"])
                    result["supervised_target_fraction"] = value
                    result["final"]["supervised_target_fraction"] = value
                continue

            m = re.match(rf"checkpoint_save_verify\s+max_abs_diff:{NUMBER_RE}\s+tensors_mismatched:(\d+)", line)
            if m:
                result["final"]["checkpoint_save_verify_max_abs_diff"] = float(m.group(1))
                result["final"]["checkpoint_save_verify_tensors_mismatched"] = int(m.group(2))
                continue

    # Derive run_id from config if available
    if "run_id" in result["config"]:
        result["run_id"] = result["config"].pop("run_id")
    if "prequant_val_bpb" in result["final"] and "postquant_val_bpb" in result["final"]:
        result["final"].setdefault(
            "qgap_bpb",
            result["final"]["postquant_val_bpb"] - result["final"]["prequant_val_bpb"],
        )
    if "prequant_val_loss" in result["final"] and "postquant_val_loss" in result["final"]:
        result["final"].setdefault(
            "qgap_loss",
            result["final"]["postquant_val_loss"] - result["final"]["prequant_val_loss"],
        )
    if "reloaded_prequant_val_bpb" in result["final"] and "prequant_val_bpb" in result["final"]:
        result["final"].setdefault(
            "replay_trust_prequant_delta_bpb",
            result["final"]["reloaded_prequant_val_bpb"] - result["final"]["prequant_val_bpb"],
        )
    if "reloaded_prequant_val_loss" in result["final"] and "prequant_val_loss" in result["final"]:
        result["final"].setdefault(
            "replay_trust_prequant_delta_loss",
            result["final"]["reloaded_prequant_val_loss"] - result["final"]["prequant_val_loss"],
        )
    if "reloaded_postquant_val_bpb" in result["final"] and "postquant_val_bpb" in result["final"]:
        result["final"].setdefault(
            "replay_trust_postquant_delta_bpb",
            result["final"]["reloaded_postquant_val_bpb"] - result["final"]["postquant_val_bpb"],
        )
    if "reloaded_postquant_val_loss" in result["final"] and "postquant_val_loss" in result["final"]:
        result["final"].setdefault(
            "replay_trust_postquant_delta_loss",
            result["final"]["reloaded_postquant_val_loss"] - result["final"]["postquant_val_loss"],
        )
    if "total_submission_bytes" in result["final"]:
        result["final"]["artifact_slack_bytes"] = 16_000_000 - int(result["final"]["total_submission_bytes"])
    if "train_tokens_seen" in result:
        result["final"]["train_tokens_seen"] = int(result["train_tokens_seen"])
    if "train_supervised_tokens_seen" in result:
        result["final"]["train_supervised_tokens_seen"] = int(result["train_supervised_tokens_seen"])
    if "ignored_target_tokens_seen" in result:
        result["final"]["ignored_target_tokens_seen"] = int(result["ignored_target_tokens_seen"])
    if "supervised_target_fraction" in result:
        result["final"]["supervised_target_fraction"] = float(result["supervised_target_fraction"])
    elif "train_supervised_tokens_seen" in result and "ignored_target_tokens_seen" in result:
        total_targets = int(result["train_supervised_tokens_seen"]) + int(result["ignored_target_tokens_seen"])
        result["final"]["supervised_target_fraction"] = (
            int(result["train_supervised_tokens_seen"]) / total_targets if total_targets else 1.0
        )
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
