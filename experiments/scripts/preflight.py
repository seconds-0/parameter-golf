#!/usr/bin/env python3
"""Verify that a remote machine is ready before launching an experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

from config_utils import MACHINES_FILE, resolve_machine


def ssh_capture(host: str, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        text=True,
        capture_output=True,
        check=False,
    )


def _ok(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": True, "detail": detail}


def _fail(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": False, "detail": detail}


def run_preflight_target(host_or_machine: str, *, print_output: bool = True) -> dict[str, Any]:
    machine_name, machine, errors = resolve_machine(host_or_machine)
    results: list[dict[str, Any]] = []
    if errors or machine is None:
        summary = f"{'; '.join(errors)}. Expected a configured machine in {MACHINES_FILE}"
        payload = {"ok": False, "host": host_or_machine, "machine_name": host_or_machine, "results": [_fail("target", summary)], "summary": summary}
        if print_output:
            print_report(payload)
        return payload

    host = str(machine["host"])
    shared_dir = str(machine.get("remote_dir") or "")
    expected_gpus = int(machine.get("gpus") or 0)
    data_path = f"{shared_dir}/data/datasets/fineweb10B_sp1024"
    tokenizer_path = f"{shared_dir}/data/tokenizers/fineweb_1024_bpe.model"

    connectivity = ssh_capture(host, "echo ok")
    if connectivity.returncode != 0 or connectivity.stdout.strip() != "ok":
        detail = connectivity.stderr.strip() or "SSH connection failed"
        payload = {"ok": False, "host": host, "machine_name": machine_name, "results": [_fail("ssh", detail)], "summary": detail}
        if print_output:
            print_report(payload)
        return payload
    results.append(_ok("ssh", "connected"))

    gpu_proc = ssh_capture(host, "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
    gpu_count = int(gpu_proc.stdout.strip() or "0") if gpu_proc.returncode == 0 else 0
    if gpu_proc.returncode == 0 and gpu_count == expected_gpus:
        results.append(_ok("gpu_count", f"{gpu_count} GPUs"))
    else:
        results.append(_fail("gpu_count", f"expected {expected_gpus}, got {gpu_count}"))

    torch_proc = ssh_capture(host, "python3 -c 'import torch; print(torch.cuda.device_count())'")
    torch_count = int(torch_proc.stdout.strip() or "0") if torch_proc.returncode == 0 else 0
    if torch_proc.returncode == 0 and torch_count == expected_gpus:
        results.append(_ok("torch_cuda", f"torch sees {torch_count} GPUs"))
    else:
        detail = torch_proc.stderr.strip() or f"torch sees {torch_count} GPUs"
        results.append(_fail("torch_cuda", detail))

    deps_proc = ssh_capture(host, "python3 -c 'import sentencepiece, numpy; print(\"ok\")'")
    if deps_proc.returncode == 0 and deps_proc.stdout.strip() == "ok":
        results.append(_ok("deps", "sentencepiece and numpy import"))
    else:
        results.append(_fail("deps", deps_proc.stderr.strip() or deps_proc.stdout.strip() or "dependency import failed"))

    disk_proc = ssh_capture(host, "python3 -c 'import shutil; print(int(shutil.disk_usage(\"/tmp\").free / (1024**3)))'")
    free_gb = int(disk_proc.stdout.strip() or "0") if disk_proc.returncode == 0 else 0
    if disk_proc.returncode == 0 and free_gb > 10:
        results.append(_ok("disk", f"{free_gb}GB free in /tmp"))
    else:
        results.append(_fail("disk", f"only {free_gb}GB free in /tmp"))

    data_proc = ssh_capture(host, f"test -d '{data_path}' && test -f '{tokenizer_path}'")
    if data_proc.returncode == 0:
        results.append(_ok("data", "dataset and tokenizer present"))
    else:
        results.append(_fail("data", f"missing {data_path} or {tokenizer_path}"))

    scratch_proc = ssh_capture(
        host,
        "mkdir -p /tmp/pgolf_preflight_test && echo ok > /tmp/pgolf_preflight_test/check && rm -rf /tmp/pgolf_preflight_test",
    )
    if scratch_proc.returncode == 0:
        results.append(_ok("scratch", "scratch dir writable"))
    else:
        results.append(_fail("scratch", scratch_proc.stderr.strip() or "scratch dir not writable"))

    ok = all(result["ok"] for result in results)
    summary = "preflight passed" if ok else "preflight failed"
    payload = {"ok": ok, "host": host, "machine_name": machine_name, "results": results, "summary": summary}
    if print_output:
        print_report(payload)
    return payload


def print_report(payload: dict[str, Any]) -> None:
    print(f"Target: {payload['machine_name']} ({payload['host']})")
    for result in payload["results"]:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"[{status}] {result['name']}: {result['detail']}")
    print(payload["summary"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run remote host preflight checks")
    parser.add_argument("host_or_machine")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_preflight_target(args.host_or_machine, print_output=True)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
