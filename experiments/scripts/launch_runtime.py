#!/usr/bin/env python3
"""Shared runtime helpers for the Parameter Golf launcher."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from config_utils import (
    DEFAULT_DATASET_VARIANT,
    MACHINES_FILE,
    REPO_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TRAINER_PATH,
    git_sha,
    now_iso,
    remote_python_command,
    resolve_machine,
    sha256_file,
    unique_run_id,
    with_default_paths,
)


def q(value: str) -> str:
    return shlex.quote(value)


def run_cmd(cmd: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=capture, check=check)


def ssh(host: str, command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_cmd(["ssh", "-o", "ConnectTimeout=30", host, command], capture=capture, check=check)


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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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


def build_manifest(
    config_path: Path,
    env: dict[str, str],
    label: str,
    host_arg: str,
    gpus: int | None,
    remote_dir: str | None,
    *,
    run_id: str | None = None,
    metadata: dict[str, str] | None = None,
    sweep_id: str | None = None,
    shutdown_enabled: bool = False,
) -> dict[str, Any]:
    machine_name, machine, errors = resolve_machine(host_arg)
    if errors or machine is None:
        raise RuntimeError(f"{'; '.join(errors)}. Expected an entry in {MACHINES_FILE}")
    host = str(machine["host"])
    shared_dir = remote_dir or str(machine.get("remote_dir") or "")
    if not shared_dir:
        raise RuntimeError(f"Machine {machine_name} is missing remote_dir in {MACHINES_FILE}")
    resolved_gpus = gpus if gpus is not None else int(machine.get("gpus") or 0)
    if resolved_gpus <= 0:
        raise RuntimeError(f"Machine {machine_name} must define a positive GPU count to launch runs")
    current_run_id = run_id or unique_run_id(label)
    remote_root = f"/tmp/pgolf_{current_run_id}"
    metadata = dict(metadata or {})
    explicit_runtime_paths = "DATA_PATH" in env or "TOKENIZER_PATH" in env
    forwarded_env = with_default_paths({**env, "RUN_ID": current_run_id}, shared_dir)
    if not explicit_runtime_paths:
        forwarded_env.setdefault("DATASET_VARIANT", DEFAULT_DATASET_VARIANT)
    if metadata.get("group"):
        forwarded_env["PGOLF_WANDB_GROUP"] = metadata["group"]
    if metadata.get("notes"):
        forwarded_env["PGOLF_WANDB_NOTES"] = metadata["notes"]
    tags = [f"hypothesis:{metadata['hypothesis_id']}"] if metadata.get("hypothesis_id") else []
    if tags:
        forwarded_env["PGOLF_WANDB_TAGS"] = ",".join(tags)
    if "WANDB_API_KEY" in os.environ:
        forwarded_env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    manifest_env = {**forwarded_env}
    if "WANDB_API_KEY" in manifest_env:
        manifest_env["WANDB_API_KEY"] = "<redacted>"
    start_epoch = time.time()
    shutdown_minutes = int(machine.get("shutdown_after_idle_minutes", 2) or 2)
    manifest = {
        "run_id": current_run_id,
        "label": label,
        "config_file": str(config_path),
        "resolved_env": manifest_env,
        "git_sha": git_sha(),
        "host": host,
        "host_name": machine_name,
        "gpus": resolved_gpus,
        "machine_gpus": machine.get("gpus"),
        "machine_remote_dir": shared_dir,
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
        "remote_log_path": f"{remote_root}/repo/logs/{current_run_id}.txt",
        "remote_stdout_path": f"{remote_root}/launcher.stdout",
        "remote_exit_code_path": f"{remote_root}/exit_code",
        "remote_pid_path": f"{remote_root}/pid",
        "remote_wrapper_path": f"{remote_root}/run.sh",
        "pid": None,
        "failure_reason": None,
        "hypothesis_id": metadata.get("hypothesis_id"),
        "group": metadata.get("group"),
        "notes": metadata.get("notes"),
        "sweep_id": sweep_id,
        "shutdown_enabled": shutdown_enabled,
        "shutdown_after_idle_minutes": shutdown_minutes,
        "shutdown_command": None,
        "last_log_line": None,
        "last_log_update_epoch": None,
        "_forwarded_env": forwarded_env,
    }
    manifest["estimated_cost"] = compute_cost(manifest)
    return manifest


def _visible(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if not key.startswith("_")}


def sync_repo(manifest: dict[str, Any]) -> None:
    host = manifest["host"]
    ssh(host, f"rm -rf {q(manifest['remote_root'])} && mkdir -p {q(manifest['remote_repo_dir'])}")
    excludes = [
        ".git",
        ".codex-reviews",
        ".venv",
        "__pycache__",
        "data/datasets",
        "data/tokenizers",
        "experiments/results",
        "logs",
    ]
    if shutil.which("rsync"):
        try:
            run_cmd(
                [
                    "rsync", "-az",
                    *(f"--exclude={item}" for item in excludes),
                    f"{REPO_DIR}/",
                    f"{host}:{manifest['remote_repo_dir']}/",
                ]
            )
            return
        except subprocess.CalledProcessError:
            # Some remote images do not ship rsync even when it exists locally.
            pass

    # Fallback for environments without local rsync, or when remote rsync is missing.
    tar_env = dict(os.environ)
    tar_env.setdefault("COPYFILE_DISABLE", "1")
    tar_cmd = [
        "tar",
        "--format",
        "ustar",
        *(f"--exclude={item}" for item in excludes),
        "-czf",
        "-",
        ".",
    ]
    remote_extract = (
        f"cd {q(manifest['remote_repo_dir'])} && "
        "tar --no-same-owner --no-same-permissions -xzmf -"
    )
    tar_proc = subprocess.Popen(
        tar_cmd,
        cwd=REPO_DIR,
        env=tar_env,
        stdout=subprocess.PIPE,
    )
    assert tar_proc.stdout is not None
    ssh_proc = subprocess.Popen(
        ["ssh", "-o", "ConnectTimeout=30", host, remote_extract],
        stdin=tar_proc.stdout,
    )
    tar_proc.stdout.close()
    ssh_return = ssh_proc.wait()
    tar_return = tar_proc.wait()
    if tar_return != 0:
        raise subprocess.CalledProcessError(tar_return, tar_cmd)
    if ssh_return != 0:
        raise subprocess.CalledProcessError(ssh_return, ["ssh", "-o", "ConnectTimeout=30", host, remote_extract])


def ensure_remote_data(manifest: dict[str, Any]) -> None:
    data_path = str(manifest["resolved_env"]["DATA_PATH"])
    tokenizer_path = str(manifest["resolved_env"]["TOKENIZER_PATH"])
    presence_check = f"test -f {q(tokenizer_path)} && test -d {q(data_path)}"
    if ssh(manifest["host"], presence_check, check=False).returncode == 0:
        return
    variant = manifest["resolved_env"].get("DATASET_VARIANT")
    if not variant:
        raise RuntimeError(
            "Missing dataset/tokenizer and no DATASET_VARIANT was provided for auto-download: "
            f"DATA_PATH={data_path!r} TOKENIZER_PATH={tokenizer_path!r}"
        )
    shared = str(manifest["machine_remote_dir"])
    ssh(
        manifest["host"],
        " ".join(
            [
                f"(cd {q(shared)} && flock -x /tmp/pgolf_data.lock {remote_python_command(shared, f'data/cached_challenge_fineweb.py --variant {q(variant)}')})",
            ]
        ),
    )


def maybe_install_wandb(manifest: dict[str, Any]) -> None:
    if "WANDB_PROJECT" in manifest["_forwarded_env"]:
        shared = str(manifest["machine_remote_dir"])
        ssh(manifest["host"], f"({remote_python_command(shared, '-m pip install wandb -q >/dev/null 2>&1')}) || true", check=False)


def verify_remote_dependencies(manifest: dict[str, Any]) -> None:
    shared = str(manifest["machine_remote_dir"])
    import_cmd = f"-c {q('import torch; import sentencepiece; import numpy')}"
    ssh(
        manifest["host"],
        f"cd {q(manifest['remote_repo_dir'])} && {remote_python_command(shared, import_cmd)}",
    )


def _env_heredoc_delimiter(run_id: str) -> str:
    return f"PGOLF_ENV_EOF_{sha256(run_id.encode('utf-8')).hexdigest()[:12]}"


def start_remote_run(manifest: dict[str, Any]) -> None:
    export_lines = "\n".join(
        f"export {key}={q(value)}" for key, value in sorted(manifest["_forwarded_env"].items())
    )
    delimiter = _env_heredoc_delimiter(manifest["run_id"])
    distributed_run = remote_python_command(
        str(manifest["machine_remote_dir"]),
        f"-m torch.distributed.run --standalone --nproc_per_node={manifest['gpus']} train_gpt.py",
    )
    wrapper = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -uo pipefail",
            f"cd {q(manifest['remote_repo_dir'])}",
            "mkdir -p logs",
            export_lines,
            f"export PYTHONPATH={q(str(Path(manifest['remote_repo_dir']) / 'experiments/scripts'))}${{PYTHONPATH:+:$PYTHONPATH}}",
            f"echo $$ > {q(manifest['remote_pid_path'])}",
            f"{distributed_run} > {q(manifest['remote_stdout_path'])} 2>&1",
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


def remote_log_snapshot(manifest: dict[str, Any], *, lines: int = 20) -> dict[str, Any]:
    log_path_raw = manifest.get("remote_log_path")
    stdout_path_raw = manifest.get("remote_stdout_path")
    if not log_path_raw and not stdout_path_raw:
        return {"lines": [], "last_line": None, "mtime_epoch": None, "source": None}
    command = "\n".join(
        [
            f"if [ -n {q(str(log_path_raw or ''))} ] && [ -f {q(str(log_path_raw or ''))} ]; then target={q(str(log_path_raw or ''))}; "
            f"elif [ -n {q(str(stdout_path_raw or ''))} ] && [ -f {q(str(stdout_path_raw or ''))} ]; then target={q(str(stdout_path_raw or ''))}; "
            "else exit 0; fi",
            f"tail -n {int(lines)} \"$target\" 2>/dev/null || true",
            "printf '\\n__PGOLF_MTIME__:%s\\n' \"$(python3 - <<'PY' \"$target\"\n"
            "from pathlib import Path\n"
            "import sys\n"
            "try:\n"
            "    print(int(Path(sys.argv[1]).stat().st_mtime))\n"
            "except OSError:\n"
            "    print('')\n"
            "PY\n"
            ")\"",
        ]
    )
    proc = ssh(manifest["host"], command, capture=True, check=False)
    output = proc.stdout.splitlines()
    mtime: float | None = None
    lines_out: list[str] = []
    for line in output:
        if line.startswith("__PGOLF_MTIME__:"):
            value = line.split(":", 1)[1].strip()
            if value.isdigit():
                mtime = float(value)
            continue
        lines_out.append(line)
    last_line = next((line for line in reversed(lines_out) if line.strip()), None)
    return {
        "lines": lines_out,
        "last_line": last_line,
        "mtime_epoch": mtime,
        "source": log_path_raw if lines_out else stdout_path_raw,
    }


def remote_log_summaries(manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not manifests:
        return {}
    summaries: dict[str, dict[str, Any]] = {}
    by_host: dict[str, list[dict[str, Any]]] = {}
    for manifest in manifests:
        by_host.setdefault(str(manifest["host"]), []).append(manifest)
    for host, host_manifests in by_host.items():
        blocks: list[str] = []
        for manifest in host_manifests:
            log_path_raw = manifest.get("remote_log_path")
            stdout_path_raw = manifest.get("remote_stdout_path")
            run_id = q(manifest["run_id"])
            if not log_path_raw and not stdout_path_raw:
                blocks.append(f'printf "%s\\t\\t\\n" {run_id}')
                continue
            log_path = q(str(log_path_raw or ""))
            stdout_path = q(str(stdout_path_raw or ""))
            block = "\n".join(
                [
                    f"if [ -f {log_path} ]; then target={log_path}; elif [ -f {stdout_path} ]; then target={stdout_path}; "
                    f"else printf \"%s\\t\\t\\n\" {run_id}; target=''; fi",
                    "[ -n \"$target\" ] || true",
                    "mtime=$(python3 - <<'PY' \"$target\"\n"
                    "from pathlib import Path\n"
                    "import sys\n"
                    "try:\n"
                    "    print(int(Path(sys.argv[1]).stat().st_mtime))\n"
                    "except OSError:\n"
                    "    print('')\n"
                    "PY\n"
                    ")",
                    "if [ -n \"$target\" ]; then last=$(tail -n 20 \"$target\" 2>/dev/null | tail -n 1 | tr '\\t' ' ' | tr '\\r' ' '); "
                    f'printf "%s\\t%s\\t%s\\n" {run_id} "$mtime" "$last"; fi',
                ]
            )
            blocks.append(block)
        proc = ssh(host, "\n".join(blocks), capture=True, check=False)
        for line in proc.stdout.splitlines():
            run_id, _, remainder = line.partition("\t")
            if not run_id:
                continue
            mtime_text, _, last_line = remainder.partition("\t")
            summaries[run_id] = {
                "mtime_epoch": float(mtime_text) if mtime_text.strip().isdigit() else None,
                "last_line": last_line.strip() or None,
            }
    return summaries


def terminate_remote_run(manifest: dict[str, Any], *, failure_reason: str | None = None) -> None:
    pid_path = q(manifest["remote_pid_path"])
    ssh(
        manifest["host"],
        f"if [ -f {pid_path} ]; then kill -TERM -- -\"$(cat {pid_path})\" 2>/dev/null || kill -TERM \"$(cat {pid_path})\" 2>/dev/null || true; fi",
        check=False,
    )
    if failure_reason:
        manifest["failure_reason"] = failure_reason
    manifest["status"] = "failed"
    manifest["exit_code"] = manifest.get("exit_code") or -9
    manifest["end_time"] = manifest.get("end_time") or now_iso()
    manifest["end_time_epoch"] = manifest.get("end_time_epoch") or time.time()
    manifest["estimated_cost"] = compute_cost(manifest)
    save_manifest(_visible(manifest))


def schedule_shutdown(manifests: list[dict[str, Any]]) -> str | None:
    if not manifests:
        return None
    minutes = int(manifests[0].get("shutdown_after_idle_minutes") or 2)
    command = f"sudo shutdown -h +{minutes}"
    ssh(manifests[0]["host"], command, check=False)
    for manifest in manifests:
        manifest["shutdown_command"] = command
        manifest["estimated_cost"] = compute_cost(manifest)
        save_manifest(_visible(manifest))
    return command


def collect_run(manifest: dict[str, Any], *, summarize: bool) -> None:
    result_dir = RESULTS_DIR / manifest["run_id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    exit_state, exit_code = remote_state(manifest)
    if exit_state == "finished" and manifest.get("exit_code") is None:
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
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
        metrics_status = metrics.get("status")
    if manifest.get("exit_code") is None:
        manifest["status"] = "running"
    elif manifest["exit_code"] == 0 and metrics_status == "success" and not manifest.get("failure_reason"):
        manifest["status"] = "success"
    else:
        manifest["status"] = "failed"
    manifest["estimated_cost"] = compute_cost(manifest)
    save_manifest(_visible(manifest))
    if summarize and metrics_path.exists():
        print()
        run_cmd([sys.executable, str(SCRIPTS_DIR / "compare.py"), str(metrics_path)], check=False)


def wait_for_run(manifest: dict[str, Any], *, summarize: bool, watchdog_enabled: bool = True) -> None:
    try:
        while True:
            state, exit_code = remote_state(manifest)
            if state == "finished":
                manifest["exit_code"] = exit_code
                manifest["end_time"] = now_iso()
                manifest["end_time_epoch"] = time.time()
                break
            if state == "unknown":
                manifest["exit_code"] = manifest.get("exit_code") or -1
                manifest["end_time"] = manifest.get("end_time") or now_iso()
                manifest["end_time_epoch"] = manifest.get("end_time_epoch") or time.time()
                break
            if watchdog_enabled:
                import watchdog

                outcome = watchdog.check_watchdog(manifest)
                if outcome.get("triggered"):
                    manifest["failure_reason"] = outcome.get("failure_reason")
                    manifest["exit_code"] = -9
                    manifest["end_time"] = now_iso()
                    manifest["end_time_epoch"] = time.time()
                    break
            time.sleep(15)
    finally:
        collect_run(manifest, summarize=summarize)


def launch_single(
    config_path: Path,
    env: dict[str, str],
    label: str,
    host: str,
    gpus: int,
    remote_dir: str | None,
    *,
    wait: bool,
    run_id: str | None = None,
    metadata: dict[str, str] | None = None,
    sweep_id: str | None = None,
    shutdown: bool = False,
    skip_preflight: bool = False,
) -> str:
    manifest = build_manifest(
        config_path,
        env,
        label,
        host,
        gpus,
        remote_dir,
        run_id=run_id,
        metadata=metadata,
        sweep_id=sweep_id,
        shutdown_enabled=shutdown,
    )
    save_manifest(_visible(manifest))
    try:
        if not skip_preflight:
            import preflight

            preflight_result = preflight.run_preflight_target(
                manifest["host_name"],
                config_env=manifest["resolved_env"],
                print_output=False,
            )
            if not preflight_result["ok"]:
                raise RuntimeError(preflight_result["summary"])
            dataset_info = preflight_result.get("dataset_info")
            if dataset_info is not None:
                manifest["dataset_snapshot"] = dataset_info
                save_manifest(_visible(manifest))
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
        manifest["failure_reason"] = manifest.get("failure_reason") or str(exc)
        manifest["error"] = str(exc)
        save_manifest(_visible(manifest))
        raise
    print(f"Started {manifest['run_id']} on {manifest['host_name']} ({manifest['host']})")
    if wait:
        wait_for_run(manifest, summarize=True)
        if shutdown:
            schedule_shutdown([manifest])
    return manifest["run_id"]
