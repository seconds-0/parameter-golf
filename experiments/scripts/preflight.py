#!/usr/bin/env python3
"""Verify that a remote machine is ready before launching an experiment."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from config_utils import MACHINES_FILE, remote_python_command, resolve_machine, resolve_path, validate_config, with_default_paths

COMPILE_TOOLCHAIN_CHECK = """PYTHON_BIN=${PGOLF_PYTHON_BIN:-python3}
if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN=python3; fi
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from sysconfig import get_paths
import shutil

header = Path(get_paths()["include"]) / "Python.h"
compiler = shutil.which("gcc") or shutil.which("cc")
print("ok" if header.is_file() and compiler else f"missing header={header.is_file()} compiler={bool(compiler)} include={header}")
PY"""


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


def load_preflight_env(config_path_arg: str) -> tuple[dict[str, str] | None, list[str], list[str]]:
    config_path = resolve_path(config_path_arg)
    result = validate_config(config_path)
    warnings = list(result.warnings)
    errors = list(result.errors)
    if errors:
        return None, warnings, errors
    if len(result.runs) != 1:
        return None, warnings, [f"Config {config_path} expands to {len(result.runs)} runs; preflight requires exactly 1 run"]
    return result.runs[0].env, warnings, []


def run_preflight_target(
    host_or_machine: str,
    *,
    config_env: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    print_output: bool = True,
) -> dict[str, Any]:
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
    for warning in warnings or []:
        results.append(_ok("config_warning", warning))
    resolved_env = with_default_paths(config_env or {}, shared_dir)
    data_path = resolved_env["DATA_PATH"]
    tokenizer_path = resolved_env["TOKENIZER_PATH"]

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

    torch_proc = ssh_capture(host, remote_python_command(shared_dir, "-c 'import torch; print(torch.cuda.device_count())'"))
    torch_count = int(torch_proc.stdout.strip() or "0") if torch_proc.returncode == 0 else 0
    if torch_proc.returncode == 0 and torch_count == expected_gpus:
        results.append(_ok("torch_cuda", f"torch sees {torch_count} GPUs"))
    else:
        detail = torch_proc.stderr.strip() or f"torch sees {torch_count} GPUs"
        results.append(_fail("torch_cuda", detail))

    deps_proc = ssh_capture(host, remote_python_command(shared_dir, "-c 'import sentencepiece, numpy; print(\"ok\")'"))
    if deps_proc.returncode == 0 and deps_proc.stdout.strip() == "ok":
        results.append(_ok("deps", "sentencepiece and numpy import"))
    else:
        results.append(_fail("deps", deps_proc.stderr.strip() or deps_proc.stdout.strip() or "dependency import failed"))

    toolchain_proc = ssh_capture(
        host,
        f"export PGOLF_PYTHON_BIN={shlex.quote(str(Path(shared_dir) / '.venv' / 'bin' / 'python'))}; {COMPILE_TOOLCHAIN_CHECK}",
    )
    if toolchain_proc.returncode == 0 and toolchain_proc.stdout.strip() == "ok":
        results.append(_ok("compile_toolchain", "Python.h and gcc present"))
    else:
        detail = toolchain_proc.stderr.strip() or toolchain_proc.stdout.strip() or "missing Python build headers or compiler"
        results.append(_fail("compile_toolchain", detail))

    disk_proc = ssh_capture(host, remote_python_command(shared_dir, "-c 'import shutil; print(int(shutil.disk_usage(\"/tmp\").free / (1024**3)))'"))
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
    parser.add_argument("--config")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_env: dict[str, str] | None = None
    warnings: list[str] = []
    if args.config:
        config_env, warnings, errors = load_preflight_env(args.config)
        if errors:
            summary = "; ".join(errors)
            payload = {
                "ok": False,
                "host": args.host_or_machine,
                "machine_name": args.host_or_machine,
                "results": [_fail("config", summary)],
                "summary": summary,
            }
            print_report(payload)
            return 1
    payload = run_preflight_target(args.host_or_machine, config_env=config_env, warnings=warnings, print_output=True)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
