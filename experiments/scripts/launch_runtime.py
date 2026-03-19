#!/usr/bin/env python3
"""Shared runtime helpers for the Parameter Golf launcher."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from config_utils import (
    MACHINES_FILE,
    REPO_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TRAINER_PATH,
    git_sha,
    now_iso,
    resolve_machine,
    sha256_file,
    unique_run_id,
)


def q(value: str) -> str:
    return shlex.quote(value)


def run_cmd(cmd: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=capture, check=check)


def ssh(host: str, command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_cmd(["ssh", host, command], capture=capture, check=check)


def scp_from(host: str, remote_path: str, local_path: Path) -> bool:
    proc = run_cmd(["scp", "-q", f"{host}:{remote_path}", str(local_path)], capture=True, check=False)
    return proc.returncode == 0


def save_manifest(manifest: dict[str, Any]) -> Path:
    result_dir = RESULTS_DIR / manifest["run_id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(run_id: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / run_id / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_cost(manifest: dict[str, Any]) -> float | None:
    rate = manifest.get("hourly_rate")
    start = manifest.get("start_time_epoch")
    end = manifest.get("end_time_epoch") or time.time()
    if rate is None or start is None:
        return None
    return round(max(end - start, 0.0) / 3600.0 * float(rate), 4)


def parse_hosts(hosts: list[str]) -> list[str]:
    parsed: list[str] = []
    for item in hosts:
        parsed.extend(part.strip() for part in item.split(",") if part.strip())
    return parsed


def build_manifest(config_path: Path, env: dict[str, str], label: str, host_arg: str, gpus: int, remote_dir: str | None) -> dict[str, Any]:
    machine_name, machine, errors = resolve_machine(host_arg)
    if errors or machine is None:
        raise RuntimeError(f"{'; '.join(errors)}. Expected an entry in {MACHINES_FILE}")
    host = str(machine["host"])
    shared_dir = remote_dir or str(machine.get("remote_dir") or "")
    if not shared_dir:
        raise RuntimeError(f"Machine {machine_name} is missing remote_dir in {MACHINES_FILE}")
    run_id = unique_run_id(label)
    remote_root = f"/tmp/pgolf_{run_id}"
    forwarded_env = {
        **env,
        "RUN_ID": run_id,
        "DATA_PATH": f"{shared_dir}/data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": f"{shared_dir}/data/tokenizers/fineweb_1024_bpe.model",
    }
    if "WANDB_API_KEY" in os.environ:
        forwarded_env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    manifest_env = {**forwarded_env}
    if "WANDB_API_KEY" in manifest_env:
        manifest_env["WANDB_API_KEY"] = "<redacted>"
    start_epoch = time.time()
    manifest = {
        "run_id": run_id,
        "config_file": str(config_path),
        "resolved_env": manifest_env,
        "git_sha": git_sha(),
        "host": host,
        "host_name": machine_name,
        "gpus": gpus,
        "start_time": now_iso(),
        "start_time_epoch": start_epoch,
        "end_time": None,
        "end_time_epoch": None,
        "exit_code": None,
        "status": "running",
        "hourly_rate": machine.get("hourly_rate"),
        "estimated_cost": None,
        "trainer_snapshot_sha": sha256_file(TRAINER_PATH),
        "remote_root": remote_root,
        "remote_repo_dir": f"{remote_root}/repo",
        "remote_log_path": f"{remote_root}/repo/logs/{run_id}.txt",
        "remote_stdout_path": f"{remote_root}/launcher.stdout",
        "remote_exit_code_path": f"{remote_root}/exit_code",
        "remote_pid_path": f"{remote_root}/pid",
        "remote_wrapper_path": f"{remote_root}/run.sh",
        "pid": None,
        "_forwarded_env": forwarded_env,
    }
    manifest["estimated_cost"] = compute_cost(manifest)
    return manifest


def _visible(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if not key.startswith("_")}


def sync_repo(manifest: dict[str, Any]) -> None:
    host = manifest["host"]
    ssh(host, f"rm -rf {q(manifest['remote_root'])} && mkdir -p {q(manifest['remote_repo_dir'])}")
    run_cmd(
        [
            "rsync", "-az",
            "--exclude=.git", "--exclude=.codex-reviews", "--exclude=.venv", "--exclude=__pycache__",
            "--exclude=data/datasets", "--exclude=data/tokenizers", "--exclude=experiments/results", "--exclude=logs",
            f"{REPO_DIR}/",
            f"{host}:{manifest['remote_repo_dir']}/",
        ]
    )


def ensure_remote_data(manifest: dict[str, Any]) -> None:
    shared = manifest["resolved_env"]["DATA_PATH"].split("/data/datasets/", 1)[0]
    ssh(
        manifest["host"],
        " ".join(
            [
                f"test -f {q(manifest['resolved_env']['TOKENIZER_PATH'])}",
                "&&",
                f"test -d {q(manifest['resolved_env']['DATA_PATH'])}",
                "||",
                f"(cd {q(shared)} && flock -x /tmp/pgolf_data.lock python3 data/cached_challenge_fineweb.py --variant sp1024)",
            ]
        ),
    )


def maybe_install_wandb(manifest: dict[str, Any]) -> None:
    if "WANDB_PROJECT" in manifest["_forwarded_env"]:
        ssh(manifest["host"], "python3 -m pip install wandb -q >/dev/null 2>&1 || true", check=False)


def verify_remote_dependencies(manifest: dict[str, Any]) -> None:
    ssh(manifest["host"], f"cd {q(manifest['remote_repo_dir'])} && python3 -c {q('import torch; import sentencepiece; import numpy')}")


def _env_heredoc_delimiter(run_id: str) -> str:
    return f"PGOLF_ENV_EOF_{sha256(run_id.encode('utf-8')).hexdigest()[:12]}"


def start_remote_run(manifest: dict[str, Any]) -> None:
    export_lines = "\n".join(
        f"export {key}={q(value)}" for key, value in sorted(manifest["_forwarded_env"].items())
    )
    delimiter = _env_heredoc_delimiter(manifest["run_id"])
    wrapper = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -uo pipefail",
            f"cd {q(manifest['remote_repo_dir'])}",
            "mkdir -p logs",
            export_lines,
            f"echo $$ > {q(manifest['remote_pid_path'])}",
            f"python3 -m torch.distributed.run --standalone --nproc_per_node={manifest['gpus']} train_gpt.py > {q(manifest['remote_stdout_path'])} 2>&1",
            "status=$?",
            f'printf "%s\\n" "$status" > {q(manifest["remote_exit_code_path"])}',
            'exit "$status"',
        ]
    )
    remote_cmd = "\n".join(
        [
            f"cat > {q(manifest['remote_wrapper_path'])} <<'{delimiter}'",
            wrapper,
            delimiter,
            f"chmod +x {q(manifest['remote_wrapper_path'])}",
            f"nohup setsid bash {q(manifest['remote_wrapper_path'])} >/dev/null 2>&1 < /dev/null &",
            "sleep 2",
            (
                f"if [ -f {q(manifest['remote_pid_path'])} ] "
                f"&& kill -0 \"$(cat {q(manifest['remote_pid_path'])})\" 2>/dev/null; then "
                f"cat {q(manifest['remote_pid_path'])}; "
                "else "
                "echo START_FAILED; "
                f"tail -n 50 {q(manifest['remote_stdout_path'])} 2>/dev/null || true; "
                "exit 1; fi"
            ),
        ]
    )
    proc = ssh(manifest["host"], remote_cmd, capture=True)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    pid = lines[-1] if lines else ""
    manifest["pid"] = int(pid) if pid.isdigit() else None
    manifest["estimated_cost"] = compute_cost(manifest)
    save_manifest(_visible(manifest))


def remote_state(manifest: dict[str, Any]) -> tuple[str, int | None]:
    command = (
        f"if [ -f {q(manifest['remote_exit_code_path'])} ]; then "
        f'printf "EXIT:%s\\n" "$(cat {q(manifest["remote_exit_code_path"])})"; '
        f"elif [ -f {q(manifest['remote_pid_path'])} ] && kill -0 \"$(cat {q(manifest['remote_pid_path'])})\" 2>/dev/null; then "
        "echo RUNNING; else echo UNKNOWN; fi"
    )
    proc = ssh(manifest["host"], command, capture=True, check=False)
    text = proc.stdout.strip()
    if text.startswith("EXIT:"):
        value = text.split(":", 1)[1].strip()
        return "finished", int(value) if value.lstrip("-").isdigit() else None
    if text == "RUNNING":
        return "running", None
    return "unknown", None


def remote_states(manifests: list[dict[str, Any]]) -> dict[str, tuple[str, int | None]]:
    if not manifests:
        return {}
    states: dict[str, tuple[str, int | None]] = {}
    by_host: dict[str, list[dict[str, Any]]] = {}
    for manifest in manifests:
        by_host.setdefault(str(manifest["host"]), []).append(manifest)
    for host, host_manifests in by_host.items():
        lines: list[str] = []
        for manifest in host_manifests:
            run_id = q(manifest["run_id"])
            exit_code_path = q(manifest["remote_exit_code_path"])
            pid_path = q(manifest["remote_pid_path"])
            lines.append(
                " ".join(
                    [
                        f"if [ -f {exit_code_path} ]; then",
                        f'printf "%s\\tEXIT:%s\\n" {run_id} "$(cat {exit_code_path})";',
                        f"elif [ -f {pid_path} ] && kill -0 \"$(cat {pid_path})\" 2>/dev/null; then",
                        f'printf "%s\\tRUNNING\\n" {run_id};',
                        "else",
                        f'printf "%s\\tUNKNOWN\\n" {run_id};',
                        "fi",
                    ]
                )
            )
        proc = ssh(host, "\n".join(lines), capture=True, check=False)
        for line in proc.stdout.splitlines():
            run_id, _, state_text = line.partition("\t")
            state_text = state_text.strip()
            if not run_id:
                continue
            if state_text.startswith("EXIT:"):
                value = state_text.split(":", 1)[1].strip()
                states[run_id] = ("finished", int(value) if value.lstrip("-").isdigit() else None)
            elif state_text == "RUNNING":
                states[run_id] = ("running", None)
            elif state_text == "UNKNOWN":
                states[run_id] = ("unknown", None)
    return states


def collect_run(manifest: dict[str, Any], *, summarize: bool) -> None:
    result_dir = RESULTS_DIR / manifest["run_id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    exit_state, exit_code = remote_state(manifest)
    if exit_state == "finished":
        manifest["exit_code"] = exit_code
        manifest["end_time"] = manifest.get("end_time") or now_iso()
        manifest["end_time_epoch"] = manifest.get("end_time_epoch") or time.time()
    scp_from(manifest["host"], manifest["remote_log_path"], result_dir / "train.log")
    scp_from(manifest["host"], manifest["remote_stdout_path"], result_dir / "launcher.stdout")
    scp_from(manifest["host"], f"{manifest['remote_repo_dir']}/train_gpt.py", result_dir / "train_gpt.py")
    scp_from(manifest["host"], f"{manifest['remote_repo_dir']}/final_model.pt", result_dir / "final_model.pt")
    scp_from(manifest["host"], f"{manifest['remote_repo_dir']}/final_model.int8.ptz", result_dir / "final_model.int8.ptz")
    train_log = result_dir / "train.log"
    if not train_log.exists() and (result_dir / "launcher.stdout").exists():
        train_log.write_text((result_dir / "launcher.stdout").read_text(encoding="utf-8"), encoding="utf-8")
    metrics_path = result_dir / "metrics.json"
    if train_log.exists():
        run_cmd([sys.executable, str(SCRIPTS_DIR / "parse_log.py"), str(train_log), str(metrics_path)], check=False)
    metrics_status = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_status = metrics.get("status")
    if manifest.get("exit_code") is None:
        manifest["status"] = "running"
    elif manifest["exit_code"] == 0 and metrics_status == "success":
        manifest["status"] = "success"
    else:
        manifest["status"] = "failed"
    manifest["estimated_cost"] = compute_cost(manifest)
    save_manifest(_visible(manifest))
    if summarize and metrics_path.exists():
        print()
        run_cmd([sys.executable, str(SCRIPTS_DIR / "compare.py"), str(metrics_path)], check=False)


def wait_for_run(manifest: dict[str, Any], *, summarize: bool) -> None:
    try:
        while True:
            state, exit_code = remote_state(manifest)
            if state == "finished":
                manifest["exit_code"] = exit_code
                manifest["end_time"] = now_iso()
                manifest["end_time_epoch"] = time.time()
                break
            if state == "unknown":
                manifest["exit_code"] = -1
                manifest["end_time"] = now_iso()
                manifest["end_time_epoch"] = time.time()
                break
            time.sleep(15)
    finally:
        collect_run(manifest, summarize=summarize)


def launch_single(config_path: Path, env: dict[str, str], label: str, host: str, gpus: int, remote_dir: str | None, *, wait: bool) -> str:
    manifest = build_manifest(config_path, env, label, host, gpus, remote_dir)
    save_manifest(_visible(manifest))
    try:
        sync_repo(manifest)
        ensure_remote_data(manifest)
        maybe_install_wandb(manifest)
        verify_remote_dependencies(manifest)
        start_remote_run(manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["exit_code"] = -1
        manifest["end_time"] = now_iso()
        manifest["end_time_epoch"] = time.time()
        manifest["estimated_cost"] = compute_cost(manifest)
        manifest["error"] = str(exc)
        save_manifest(_visible(manifest))
        raise
    print(f"Started {manifest['run_id']} on {manifest['host_name']} ({manifest['host']})")
    if wait:
        wait_for_run(manifest, summarize=True)
    return manifest["run_id"]
