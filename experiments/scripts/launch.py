#!/usr/bin/env python3
"""Launch and manage remote Parameter Golf experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cost
import preflight
import sweep_queue
from config_utils import RESULTS_DIR, now_iso, resolve_path, validate_config
from launch_runtime import (
    collect_run,
    launch_single,
    load_manifest,
    parse_hosts,
    remote_log_summaries,
    remote_states,
    save_manifest,
    schedule_shutdown,
    ssh,
    terminate_remote_run,
    wait_for_run,
)

TRAIN_RE = re.compile(r"step:(?P<step>\d+)/\d+\s+train_loss:(?P<loss>[-+.\deEinfnaINFNA]+)")
VAL_RE = re.compile(r"step:(?P<step>\d+)/\d+\s+val_loss:[-+.\deE]+\s+val_bpb:(?P<val_bpb>[-+.\deE]+)")


def _load_local_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc_script = _load_local_module("pgolf_gc", "gc.py")


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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.Popen(args, stdout=handle, stderr=handle, start_new_session=True)


def spawn_monitor(run_id: str) -> None:
    spawn_process([sys.executable, str(Path(__file__).resolve()), "_monitor", run_id], RESULTS_DIR / run_id / "monitor.log")


def _parse_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    total = int(max(seconds, 0))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def extract_progress(lines: list[str]) -> dict[str, Any]:
    latest_step: int | None = None
    latest_train_loss: float | None = None
    latest_val_bpb: float | None = None
    for line in lines:
        train_match = TRAIN_RE.search(line)
        if train_match:
            latest_step = int(train_match.group("step"))
            latest_train_loss = _parse_float(train_match.group("loss"))
        val_match = VAL_RE.search(line)
        if val_match:
            latest_step = int(val_match.group("step"))
            latest_val_bpb = _parse_float(val_match.group("val_bpb"))
    return {
        "latest_step": latest_step,
        "latest_train_loss": latest_train_loss,
        "latest_val_bpb": latest_val_bpb,
    }


def tail_local_lines(run_id: str, *, limit: int = 20) -> list[str]:
    result_dir = RESULTS_DIR / run_id
    for name in ("train.log", "launcher.stdout"):
        path = result_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    return []


def _status_filters(args: argparse.Namespace, manifest: dict[str, Any]) -> bool:
    if args.host and args.host not in {manifest.get("host"), manifest.get("host_name")}:
        return False
    if args.running and manifest.get("status") != "running":
        return False
    if args.failed and manifest.get("status") != "failed":
        return False
    return True


def collect_status_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifests = []
    running_manifests = []
    for path in sorted(RESULTS_DIR.glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not _status_filters(args, manifest):
            continue
        manifests.append(manifest)
        if manifest.get("status") == "running":
            running_manifests.append(manifest)

    live_states = remote_states(running_manifests)
    log_summaries = remote_log_summaries(running_manifests)
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        if manifest.get("status") == "running":
            state, exit_code = live_states.get(manifest["run_id"], ("unknown", None))
            if state == "finished":
                manifest["exit_code"] = exit_code
                manifest["status"] = "success" if exit_code == 0 else "failed"
                manifest["end_time"] = now_iso()
                manifest["end_time_epoch"] = time.time()
                save_manifest(manifest)
            elif state == "unknown":
                manifest["exit_code"] = manifest.get("exit_code") or -1
                manifest["status"] = "failed"
                manifest["failure_reason"] = manifest.get("failure_reason") or "remote_state_unknown"
                manifest["end_time"] = now_iso()
                manifest["end_time_epoch"] = time.time()
                save_manifest(manifest)
            summary = log_summaries.get(manifest["run_id"], {})
            manifest["last_log_update_epoch"] = summary.get("mtime_epoch") or manifest.get("last_log_update_epoch")
            manifest["last_log_line"] = summary.get("last_line") or manifest.get("last_log_line")
        if not _status_filters(args, manifest):
            continue
        rows.append(manifest)
    return rows


def render_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No runs found."
    headers = [
        ("Run ID", 34),
        ("Group", 12),
        ("Host", 14),
        ("Status", 8),
        ("Step", 6),
        ("Train", 8),
        ("ValBPB", 8),
        ("Elapsed", 8),
        ("Stale", 8),
        ("Cost", 7),
        ("Hypothesis", 14),
        ("Notes", 16),
        ("Failure", 20),
    ]
    lines = [" ".join(f"{name:<{width}}" for name, width in headers), "-" * 183]
    now_epoch = time.time()
    for manifest in sorted(rows, key=lambda item: item.get("start_time_epoch", 0), reverse=True):
        status = str(manifest.get("status") or "")
        progress = extract_progress(
            [manifest["last_log_line"]] if status == "running" and manifest.get("last_log_line") else tail_local_lines(manifest["run_id"])
        )
        latest_step = progress["latest_step"]
        latest_train_loss = progress["latest_train_loss"]
        latest_val_bpb = progress["latest_val_bpb"]
        elapsed_end = manifest.get("end_time_epoch") if status != "running" else now_epoch
        elapsed = None
        if manifest.get("start_time_epoch") is not None:
            elapsed = float(elapsed_end) - float(manifest["start_time_epoch"])
        staleness = None
        if manifest.get("last_log_update_epoch") is not None:
            staleness = now_epoch - float(manifest["last_log_update_epoch"])
        cost_text = manifest.get("estimated_cost")
        cost_display = f"{float(cost_text):.2f}" if isinstance(cost_text, (int, float)) else "N/A"
        values = [
            f"{manifest['run_id'][:34]:<34}",
            f"{str(manifest.get('group') or '')[:12]:<12}",
            f"{str(manifest.get('host_name') or manifest.get('host') or '')[:14]:<14}",
            f"{status[:8]:<8}",
            f"{str(latest_step or '')[:6]:<6}",
            f"{'' if latest_train_loss is None else f'{latest_train_loss:.4f}':<8}",
            f"{'' if latest_val_bpb is None else f'{latest_val_bpb:.4f}':<8}",
            f"{human_duration(elapsed):<8}",
            f"{human_duration(staleness):<8}",
            f"{cost_display:<7}",
            f"{str(manifest.get('hypothesis_id') or '')[:14]:<14}",
            f"{str(manifest.get('notes') or '')[:16]:<16}",
            f"{str(manifest.get('failure_reason') or '')[:20]:<20}",
        ]
        lines.append(" ".join(values))
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    result = validate_or_die(config_path)
    if len(result.runs) != 1:
        eprint(f"{config_path} expands to {len(result.runs)} runs; use 'launch.py sweep' instead.")
        return 1
    spec = result.runs[0]
    run_id = launch_single(
        config_path,
        spec.env,
        spec.label,
        args.host,
        args.gpus,
        args.remote_dir,
        wait=not args.detach,
        metadata=spec.metadata,
        shutdown=args.shutdown,
        skip_preflight=args.skip_preflight,
    )
    if args.detach:
        spawn_monitor(run_id)
        print(f"Detached monitor started for {run_id}")
    return 0


def _host_manifests_for_shutdown(sweep_id: str, host: str) -> list[dict[str, Any]]:
    queue = sweep_queue.load_queue(sweep_id)
    manifests: list[dict[str, Any]] = []
    for run in queue["runs"]:
        if run.get("host") != host or not run.get("run_id"):
            continue
        manifest = load_manifest(str(run["run_id"]))
        if manifest is not None:
            manifests.append(manifest)
    return manifests


def _run_sweep_worker(args: argparse.Namespace) -> int:
    hosts = parse_hosts(args.hosts)
    if not hosts:
        eprint("At least one host is required")
        return 1

    def worker(host: str) -> None:
        had_runs = False
        for running in sweep_queue.running_entries_for_host(args.sweep_id, host):
            had_runs = True
            sweep_queue.reconcile_running_entry(args.sweep_id, running)
        while True:
            entry = sweep_queue.claim_next_run(args.sweep_id, host)
            if entry is None:
                break
            had_runs = True
            run_id = str(entry["run_id"])
            try:
                launch_single(
                    Path(sweep_queue.load_queue(args.sweep_id)["config_file"]),
                    dict(entry["env"]),
                    str(entry["label"]),
                    host,
                    args.gpus,
                    args.remote_dir,
                    wait=True,
                    run_id=run_id,
                    metadata=dict(entry.get("metadata") or {}),
                    sweep_id=args.sweep_id,
                    shutdown=False,
                    skip_preflight=args.skip_preflight,
                )
            finally:
                sweep_queue.sync_entry_from_manifest(args.sweep_id, run_id)
        if args.shutdown and had_runs:
            manifests = _host_manifests_for_shutdown(args.sweep_id, host)
            if manifests and not any(manifest.get("shutdown_command") for manifest in manifests):
                schedule_shutdown(manifests)

    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = [executor.submit(worker, host) for host in hosts]
        for future in futures:
            future.result()
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    result = validate_or_die(config_path)
    queue = sweep_queue.create_queue(str(config_path), result.runs)
    print(f"Sweep queue created: {queue['sweep_id']}")
    worker_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_sweep_worker",
        queue["sweep_id"],
        "--hosts",
        *parse_hosts(args.hosts),
        "--gpus",
        str(args.gpus),
    ]
    if args.remote_dir:
        worker_args.extend(["--remote-dir", args.remote_dir])
    if args.shutdown:
        worker_args.append("--shutdown")
    if args.skip_preflight:
        worker_args.append("--skip-preflight")
    if args.detach:
        spawn_process(worker_args, RESULTS_DIR / "_sweeps" / queue["sweep_id"] / "worker.log")
        print(f"Detached sweep worker started for {queue['sweep_id']}.")
        return 0
    return _run_sweep_worker(type("Args", (), {
        "sweep_id": queue["sweep_id"],
        "hosts": args.hosts,
        "gpus": args.gpus,
        "remote_dir": args.remote_dir,
        "shutdown": args.shutdown,
        "skip_preflight": args.skip_preflight,
    })())


def cmd_resume_sweep(args: argparse.Namespace) -> int:
    try:
        sweep_queue.load_queue(args.sweep_id)
    except FileNotFoundError as exc:
        eprint(str(exc))
        return 1
    return _run_sweep_worker(args)


def cmd_requeue(args: argparse.Namespace) -> int:
    try:
        count = sweep_queue.requeue_runs(args.sweep_id, failed=args.failed, lost=args.lost)
    except FileNotFoundError as exc:
        eprint(str(exc))
        return 1
    print(f"Requeued {count} runs in {args.sweep_id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    try:
        while True:
            output = render_status(collect_status_rows(args))
            if args.watch:
                print("\033[2J\033[H", end="")
            print(output)
            if not args.watch:
                return 0
            time.sleep(10)
    except KeyboardInterrupt:
        return 130


def cmd_cancel(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_id)
    if manifest is None:
        eprint(f"Unknown run_id: {args.run_id}")
        return 1
    if args.host:
        manifest["host"] = args.host
    terminate_remote_run(manifest, failure_reason="cancelled")
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


def cmd_watch(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_id)
    if manifest is None:
        eprint(f"Unknown run_id: {args.run_id}")
        return 1
    wait_for_run(manifest, summarize=True)
    refreshed = load_manifest(args.run_id)
    if refreshed and refreshed.get("shutdown_enabled") and not refreshed.get("shutdown_command"):
        schedule_shutdown([refreshed])
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    return cmd_watch(args)


def cmd_budget(args: argparse.Namespace) -> int:
    manifests = cost.load_manifests()
    budget = args.budget if args.budget is not None else cost.load_budget_default()
    print(cost.summarize_costs(manifests, budget=budget))
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    payload = preflight.run_preflight_target(args.host, print_output=True)
    return 0 if payload["ok"] else 1


def cmd_gc(args: argparse.Namespace) -> int:
    try:
        payload = gc_script.run_gc(args.host, older_than=args.older_than, dry_run=args.dry_run)
    except Exception as exc:
        eprint(f"Error: {exc}")
        return 1
    gc_script.print_report(payload)
    return 0


def build_parser(*, include_internal: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and manage remote Parameter Golf experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Launch one config on one host")
    run_parser.add_argument("config")
    run_parser.add_argument("--host", required=True)
    run_parser.add_argument("--gpus", type=int, default=8)
    run_parser.add_argument("--remote-dir")
    run_parser.add_argument("--shutdown", action="store_true")
    run_parser.add_argument("--skip-preflight", action="store_true")
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
    sweep_parser.add_argument("--shutdown", action="store_true")
    sweep_parser.add_argument("--skip-preflight", action="store_true")
    sweep_parser.set_defaults(func=cmd_sweep)

    resume_parser = subparsers.add_parser("resume-sweep", help="Resume a persistent sweep queue")
    resume_parser.add_argument("sweep_id")
    resume_parser.add_argument("--hosts", nargs="+", required=True)
    resume_parser.add_argument("--gpus", type=int, default=8)
    resume_parser.add_argument("--remote-dir")
    resume_parser.add_argument("--shutdown", action="store_true")
    resume_parser.add_argument("--skip-preflight", action="store_true")
    resume_parser.set_defaults(func=cmd_resume_sweep)

    requeue_parser = subparsers.add_parser("requeue", help="Requeue failed or lost runs from a sweep")
    requeue_parser.add_argument("sweep_id")
    requeue_parser.add_argument("--failed", action="store_true")
    requeue_parser.add_argument("--lost", action="store_true")
    requeue_parser.set_defaults(func=cmd_requeue)

    status_parser = subparsers.add_parser("status", help="Show running and recent runs")
    status_parser.add_argument("--host")
    status_parser.add_argument("--watch", action="store_true")
    status_parser.add_argument("--running", action="store_true")
    status_parser.add_argument("--failed", action="store_true")
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

    watch_parser = subparsers.add_parser("watch", help="Watch a run until it finishes")
    watch_parser.add_argument("run_id")
    watch_parser.set_defaults(func=cmd_watch)

    budget_parser = subparsers.add_parser("budget", help="Show cost summary")
    budget_parser.add_argument("--budget", type=float)
    budget_parser.set_defaults(func=cmd_budget)

    preflight_parser = subparsers.add_parser("preflight", help="Run remote host preflight checks")
    preflight_parser.add_argument("host")
    preflight_parser.set_defaults(func=cmd_preflight)

    gc_parser = subparsers.add_parser("gc", help="Delete old remote /tmp/pgolf_* directories")
    gc_parser.add_argument("host")
    gc_parser.add_argument("--older-than", default="24h")
    gc_parser.add_argument("--dry-run", action="store_true")
    gc_parser.set_defaults(func=cmd_gc)

    if include_internal:
        monitor_parser = subparsers.add_parser("_monitor", help=argparse.SUPPRESS)
        monitor_parser.add_argument("run_id")
        monitor_parser.set_defaults(func=cmd_monitor)

        sweep_worker = subparsers.add_parser("_sweep_worker", help=argparse.SUPPRESS)
        sweep_worker.add_argument("sweep_id")
        sweep_worker.add_argument("--hosts", nargs="+", required=True)
        sweep_worker.add_argument("--gpus", type=int, default=8)
        sweep_worker.add_argument("--remote-dir")
        sweep_worker.add_argument("--shutdown", action="store_true")
        sweep_worker.add_argument("--skip-preflight", action="store_true")
        sweep_worker.set_defaults(func=cmd_resume_sweep)
    return parser


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    include_internal = len(sys.argv) > 1 and sys.argv[1] in {"_monitor", "_sweep_worker"}
    args = build_parser(include_internal=include_internal).parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
