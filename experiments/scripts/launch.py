#!/usr/bin/env python3
"""Launch and manage remote Parameter Golf experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config_utils import RESULTS_DIR, now_iso, resolve_path, validate_config
from launch_runtime import collect_run, launch_single, load_manifest, parse_hosts, remote_states, save_manifest, ssh


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def q(value: str) -> str:
    return shlex.quote(value)


def validate_or_die(config_path: Path):
    result = validate_config(config_path)
    for warning in result.warnings:
        eprint(f"Warning: {warning}")
    if result.errors:
        for error in result.errors:
            eprint(f"Error: {error}")
        raise SystemExit(1)
    return result


def spawn_process(args: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.Popen(args, stdout=handle, stderr=handle, start_new_session=True)


def spawn_monitor(run_id: str) -> None:
    spawn_process([sys.executable, str(Path(__file__).resolve()), "_monitor", run_id], RESULTS_DIR / run_id / "monitor.log")


def cmd_run(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    result = validate_or_die(config_path)
    if len(result.runs) != 1:
        eprint(f"{config_path} expands to {len(result.runs)} runs; use 'launch.py sweep' instead.")
        return 1
    spec = result.runs[0]
    run_id = launch_single(config_path, spec.env, spec.label, args.host, args.gpus, args.remote_dir, wait=not args.detach)
    if args.detach:
        spawn_monitor(run_id)
        print(f"Detached monitor started for {run_id}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    hosts = parse_hosts(args.hosts)
    if not hosts:
        eprint("At least one host is required")
        return 1
    if getattr(args, "detach", False):
        worker_args = [sys.executable, str(Path(__file__).resolve()), "_sweep_worker", args.config, "--hosts", *hosts, "--gpus", str(args.gpus)]
        if args.remote_dir:
            worker_args.extend(["--remote-dir", args.remote_dir])
        spawn_process(worker_args, RESULTS_DIR / f"sweep-{int(time.time())}.log")
        print("Detached sweep worker started.")
        return 0
    result = validate_or_die(resolve_path(args.config))
    assignments = {host: [] for host in hosts}
    for index, run in enumerate(result.runs):
        assignments[hosts[index % len(hosts)]].append(run)

    def worker(host: str) -> None:
        for run in assignments[host]:
            launch_single(resolve_path(args.config), run.env, run.label, host, args.gpus, args.remote_dir, wait=True)

    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = [executor.submit(worker, host) for host in hosts]
        for future in futures:
            future.result()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    manifests = sorted(RESULTS_DIR.glob("*/manifest.json"))
    rows = []
    running_manifests = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if args.host and args.host not in {manifest.get("host"), manifest.get("host_name")}:
            continue
        rows.append(manifest)
        if manifest.get("status") == "running":
            running_manifests.append(manifest)
    live_states = remote_states(running_manifests)
    for manifest in rows:
        if manifest.get("status") != "running":
            continue
        state, exit_code = live_states.get(manifest["run_id"], ("unknown", None))
        if state == "finished":
            manifest["exit_code"] = exit_code
            manifest["status"] = "success" if exit_code == 0 else "failed"
            manifest["end_time"] = now_iso()
            manifest["end_time_epoch"] = time.time()
            save_manifest(manifest)
    if not rows:
        print("No runs found.")
        return 0
    print(f"{'Run ID':<38} {'Host':<18} {'Status':<8} {'Exit':<5}")
    print("-" * 75)
    for manifest in sorted(rows, key=lambda item: item.get("start_time_epoch", 0), reverse=True):
        print(f"{manifest['run_id']:<38} {manifest.get('host_name', ''):<18} {manifest.get('status', ''):<8} {str(manifest.get('exit_code', '')):<5}")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_id)
    if manifest is None:
        eprint(f"Unknown run_id: {args.run_id}")
        return 1
    host = args.host or manifest["host"]
    pid_path = q(manifest["remote_pid_path"])
    ssh(host, f"if [ -f {pid_path} ]; then kill -TERM -- -\"$(cat {pid_path})\" 2>/dev/null || kill -TERM \"$(cat {pid_path})\" 2>/dev/null || true; fi", check=False)
    manifest["status"] = "failed"
    save_manifest(manifest)
    collect_run(manifest, summarize=False)
    print(f"Cancel requested for {args.run_id}")
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_id)
    if manifest is None:
        eprint(f"Unknown run_id: {args.run_id}")
        return 1
    host = args.host or manifest["host"]
    cmd = f"tail -n 50 -F {q(manifest['remote_log_path'])} 2>/dev/null || tail -n 50 -F {q(manifest['remote_stdout_path'])}"
    return ssh(host, cmd, check=False).returncode


def cmd_collect(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_id)
    if manifest is None:
        eprint(f"Unknown run_id: {args.run_id}")
        return 1
    if args.host:
        manifest["host"] = args.host
    collect_run(manifest, summarize=True)
    print(f"Collected {args.run_id}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_id)
    if manifest is None:
        eprint(f"Unknown run_id: {args.run_id}")
        return 1
    from launch_runtime import wait_for_run
    wait_for_run(manifest, summarize=True)
    return 0


def build_parser(*, include_internal: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and manage remote Parameter Golf experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Launch one config on one host")
    run_parser.add_argument("config")
    run_parser.add_argument("--host", required=True)
    run_parser.add_argument("--gpus", type=int, default=8)
    run_parser.add_argument("--remote-dir")
    group = run_parser.add_mutually_exclusive_group()
    group.add_argument("--detach", action="store_true")
    group.add_argument("--wait", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    sweep_parser = subparsers.add_parser("sweep", help="Run a sweep across one or more hosts")
    sweep_parser.add_argument("config")
    sweep_parser.add_argument("--hosts", nargs="+", required=True)
    sweep_parser.add_argument("--gpus", type=int, default=8)
    sweep_parser.add_argument("--remote-dir")
    sweep_parser.add_argument("--detach", action="store_true")
    sweep_parser.set_defaults(func=cmd_sweep)

    status_parser = subparsers.add_parser("status", help="Show running and recent runs")
    status_parser.add_argument("--host")
    status_parser.set_defaults(func=cmd_status)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a remote run")
    cancel_parser.add_argument("run_id")
    cancel_parser.add_argument("--host")
    cancel_parser.set_defaults(func=cmd_cancel)

    tail_parser = subparsers.add_parser("tail", help="Tail a remote run log")
    tail_parser.add_argument("run_id")
    tail_parser.add_argument("--host")
    tail_parser.set_defaults(func=cmd_tail)

    collect_parser = subparsers.add_parser("collect", help="Collect logs and artifacts for a run")
    collect_parser.add_argument("run_id")
    collect_parser.add_argument("--host")
    collect_parser.set_defaults(func=cmd_collect)

    if include_internal:
        monitor_parser = subparsers.add_parser("_monitor", help=argparse.SUPPRESS)
        monitor_parser.add_argument("run_id")
        monitor_parser.set_defaults(func=cmd_monitor)

        sweep_worker = subparsers.add_parser("_sweep_worker", help=argparse.SUPPRESS)
        sweep_worker.add_argument("config")
        sweep_worker.add_argument("--hosts", nargs="+", required=True)
        sweep_worker.add_argument("--gpus", type=int, default=8)
        sweep_worker.add_argument("--remote-dir")
        sweep_worker.set_defaults(func=cmd_sweep)
    return parser


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    include_internal = len(sys.argv) > 1 and sys.argv[1] in {"_monitor", "_sweep_worker"}
    args = build_parser(include_internal=include_internal).parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
