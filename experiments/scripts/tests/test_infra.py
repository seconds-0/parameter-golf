from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "experiments" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare
import config_utils
import launch
import launch_runtime
import parse_log
import submit

BASELINE_CONFIG = ROOT / "experiments" / "configs" / "baseline.yaml"
SWEEP_CONFIG = ROOT / "experiments" / "configs" / "sweep_lr.yaml"
BASELINE_LOG = ROOT / "records" / "track_10min_16mb" / "2026-03-17_NaiveBaseline" / "train.log"


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return path


def write_metrics(path: Path, *, run_id: str, val_bpb: float, val_loss: float, status: str = "success") -> None:
    payload = {
        "run_id": run_id,
        "status": status,
        "train_steps": [{"step": 123, "train_time_ms": 4567.0}],
        "config": {"model_params": 999},
        "final": {
            "val_bpb": val_bpb,
            "val_loss": val_loss,
            "total_submission_bytes": 123456,
            "peak_memory_allocated_mib": 2048,
            "stop_step": 123,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_validate_baseline_config() -> None:
    result = config_utils.validate_config(BASELINE_CONFIG)
    assert result.ok
    assert len(result.runs) == 1
    assert result.runs[0].env["VOCAB_SIZE"] == "1024"


def test_validate_sweep_config() -> None:
    result = config_utils.validate_config(SWEEP_CONFIG)
    assert result.ok
    assert len(result.runs) == 15


def test_detect_typo(tmp_path: Path) -> None:
    config_path = write_yaml(
        tmp_path / "typo.yaml",
        """
        name: typo
        env:
          VOCAB_SIZE: "1024"
          NUM_LAYERS: "9"
          MODEL_DIM: "512"
          NUM_HEADS: "8"
          NUM_KV_HEADS: "4"
          MLP_MULT: "2"
          TIE_EMBEDDINGS: "1"
          TRAIN_BATCH_TOKENS: "524288"
          TRAIN_SEQ_LEN: "1024"
          MATRIX_LRR: "0.04"
        """,
    )
    result = config_utils.validate_config(config_path)
    assert any("Unknown env var MATRIX_LRR" in warning for warning in result.warnings)


def test_detect_divisibility_error(tmp_path: Path) -> None:
    config_path = write_yaml(
        tmp_path / "bad_heads.yaml",
        """
        name: bad_heads
        env:
          VOCAB_SIZE: "1024"
          NUM_LAYERS: "9"
          MODEL_DIM: "512"
          NUM_HEADS: "8"
          NUM_KV_HEADS: "3"
          MLP_MULT: "2"
          TIE_EMBEDDINGS: "1"
          TRAIN_BATCH_TOKENS: "524288"
          TRAIN_SEQ_LEN: "1024"
        """,
    )
    result = config_utils.validate_config(config_path)
    assert any("NUM_HEADS must be divisible by NUM_KV_HEADS" in error for error in result.errors)


def test_extends_config(tmp_path: Path) -> None:
    parent = write_yaml(
        tmp_path / "parent.yaml",
        """
        name: parent
        env:
          VOCAB_SIZE: "1024"
          NUM_LAYERS: "9"
          MODEL_DIM: "512"
          NUM_HEADS: "8"
          NUM_KV_HEADS: "4"
          MLP_MULT: "2"
          TIE_EMBEDDINGS: "1"
          TRAIN_BATCH_TOKENS: "524288"
          TRAIN_SEQ_LEN: "1024"
          MATRIX_LR: "0.04"
        """,
    )
    child = write_yaml(
        tmp_path / "child.yaml",
        f"""
        extends: {parent.name}
        name: child
        env:
          SCALAR_LR: "0.03"
          MATRIX_LR: "0.05"
        """,
    )
    result = config_utils.validate_config(child)
    assert result.ok
    env = result.runs[0].env
    assert env["VOCAB_SIZE"] == "1024"
    assert env["MATRIX_LR"] == "0.05"
    assert env["SCALAR_LR"] == "0.03"


def test_random_sweep(tmp_path: Path) -> None:
    config_path = write_yaml(
        tmp_path / "random.yaml",
        """
        name: random
        base_env:
          VOCAB_SIZE: "1024"
          NUM_LAYERS: "9"
          MODEL_DIM: "512"
          NUM_HEADS: "8"
          NUM_KV_HEADS: "4"
          MLP_MULT: "2"
          TIE_EMBEDDINGS: "1"
          TRAIN_BATCH_TOKENS: "524288"
          TRAIN_SEQ_LEN: "1024"
        sweep:
          type: random
          seed: 7
          count: 3
          params:
            MATRIX_LR: ["0.02", "0.03", "0.04"]
            SCALAR_LR: ["0.03", "0.04"]
        naming: "{MATRIX_LR}_{SCALAR_LR}"
        """,
    )
    first = config_utils.validate_config(config_path)
    second = config_utils.validate_config(config_path)
    assert first.ok and second.ok
    assert [run.combo for run in first.runs] == [run.combo for run in second.runs]


def test_random_sweep_count_exceeds_combos(tmp_path: Path) -> None:
    config_path = write_yaml(
        tmp_path / "random_many.yaml",
        """
        name: random_many
        base_env:
          VOCAB_SIZE: "1024"
          NUM_LAYERS: "9"
          MODEL_DIM: "512"
          NUM_HEADS: "8"
          NUM_KV_HEADS: "4"
          MLP_MULT: "2"
          TIE_EMBEDDINGS: "1"
          TRAIN_BATCH_TOKENS: "524288"
          TRAIN_SEQ_LEN: "1024"
        sweep:
          type: random
          seed: 11
          count: 10
          params:
            MATRIX_LR: ["0.02", "0.03"]
            SCALAR_LR: ["0.03", "0.04"]
        naming: "{MATRIX_LR}_{SCALAR_LR}"
        """,
    )
    result = config_utils.validate_config(config_path)
    assert result.ok
    assert len(result.runs) == 10


def test_parse_baseline_log() -> None:
    parsed = parse_log.parse_log(BASELINE_LOG.read_text(encoding="utf-8"))
    assert parsed["status"] == "success"
    assert parsed["final"]["val_bpb"] == pytest.approx(1.2243657)


def test_parse_scientific_notation() -> None:
    parsed = parse_log.parse_log(
        "\n".join(
            [
                "step:1/10 train_loss:1.2e-05 train_time:3.4e+02ms step_avg:5.6e+02ms tok_s:7.8e+03",
                "step:1/10 val_loss:9.1e-06 val_bpb:1.2e+00 train_time:1.5e+02ms step_avg:2.5e+02ms",
                "final_int8_zlib_roundtrip_exact val_loss:1.2e-05 val_bpb:3.4e-06",
            ]
        )
    )
    assert parsed["train_steps"][0]["train_loss"] == pytest.approx(1.2e-05)
    assert parsed["train_steps"][0]["train_time_ms"] == pytest.approx(340.0)
    assert parsed["val_steps"][0]["val_loss"] == pytest.approx(9.1e-06)
    assert parsed["final"]["val_bpb"] == pytest.approx(3.4e-06)


def test_parse_failed_run() -> None:
    parsed = parse_log.parse_log("final_int8_zlib_roundtrip val_loss:2.0 val_bpb:1.2 eval_time:123ms")
    assert parsed["status"] == "failed"


def test_parse_empty_log() -> None:
    parsed = parse_log.parse_log("garbage\n\n")
    assert parsed["status"] == "failed"
    assert parsed["train_steps"] == []
    assert parsed["val_steps"] == []


def test_compare_single_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    metrics_path = tmp_path / "metrics.json"
    write_metrics(metrics_path, run_id="solo", val_bpb=1.23, val_loss=2.34)
    monkeypatch.setattr(sys, "argv", ["compare.py", str(metrics_path)])
    compare.main()
    output = capsys.readouterr().out
    assert "solo" in output
    assert "1.2300*" in output


def test_compare_sorts_by_bpb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    worse = tmp_path / "worse.json"
    better = tmp_path / "better.json"
    write_metrics(worse, run_id="worse", val_bpb=1.30, val_loss=2.40)
    write_metrics(better, run_id="better", val_bpb=1.20, val_loss=2.10)
    monkeypatch.setattr(sys, "argv", ["compare.py", str(worse), str(better)])
    compare.main()
    lines = capsys.readouterr().out.splitlines()
    data_lines = [line for line in lines if " | " in line and not line.startswith("Run ID")]
    assert data_lines[0].startswith("better")
    assert data_lines[1].startswith("worse")


def test_submit_uses_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_dir = tmp_path / "repo"
    result_dir = repo_dir / "experiments" / "results" / "run-123"
    result_dir.mkdir(parents=True)
    (repo_dir / "train_gpt.py").write_text("current copy\n", encoding="utf-8")
    (result_dir / "train_gpt.py").write_text("snapshot copy\n", encoding="utf-8")
    (result_dir / "train.log").write_text("log\n", encoding="utf-8")
    write_metrics(result_dir / "metrics.json", run_id="run-123", val_bpb=1.22, val_loss=2.07)
    monkeypatch.setattr(submit, "REPO_DIR", repo_dir)
    monkeypatch.setattr(submit, "RESULTS_DIR", repo_dir / "experiments" / "results")
    monkeypatch.setattr(
        sys,
        "argv",
        ["submit.py", "--run-id", "run-123", "--name", "SnapshotPreferred"],
    )
    submit.main()
    created = next((repo_dir / "records" / "track_10min_16mb").iterdir())
    assert (created / "train_gpt.py").read_text(encoding="utf-8") == "snapshot copy\n"


def test_validate_then_expand() -> None:
    result = config_utils.validate_config(SWEEP_CONFIG)
    assert result.ok
    assert len(result.runs) == 15
    assert result.runs[0].label.startswith("lr_mat")
    assert set(result.runs[0].combo) == {"MATRIX_LR", "SCALAR_LR"}


def test_manifest_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launch_runtime, "RESULTS_DIR", tmp_path / "results")
    manifest = {"run_id": "roundtrip", "status": "running", "host": "example"}
    launch_runtime.save_manifest(manifest)
    assert launch_runtime.load_manifest("roundtrip") == manifest


def test_collect_run_downloads_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launch_runtime, "RESULTS_DIR", tmp_path / "results")

    def fake_remote_state(manifest: dict[str, object]) -> tuple[str, int | None]:
        return "finished", 0

    def fake_scp_from(host: str, remote_path: str, local_path: Path) -> bool:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if remote_path.endswith("/train_gpt.py"):
            local_path.write_text("snapshot trainer\n", encoding="utf-8")
            return True
        if remote_path.endswith("launcher.stdout"):
            local_path.write_text("final_int8_zlib_roundtrip_exact val_loss:2.0 val_bpb:1.1\n", encoding="utf-8")
            return True
        return False

    def fake_run_cmd(cmd: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        if any(str(part).endswith("parse_log.py") for part in cmd):
            metrics_path = Path(cmd[-1])
            metrics_path.write_text(
                json.dumps({"status": "success", "final": {"val_bpb": 1.1, "val_loss": 2.0}}, indent=2) + "\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(launch_runtime, "remote_state", fake_remote_state)
    monkeypatch.setattr(launch_runtime, "scp_from", fake_scp_from)
    monkeypatch.setattr(launch_runtime, "run_cmd", fake_run_cmd)
    manifest = {
        "run_id": "collect-me",
        "host": "example",
        "remote_log_path": "/remote/log.txt",
        "remote_stdout_path": "/remote/launcher.stdout",
        "remote_repo_dir": "/remote/repo",
        "remote_exit_code_path": "/remote/exit_code",
        "remote_pid_path": "/remote/pid",
        "status": "running",
        "exit_code": None,
        "end_time": None,
        "end_time_epoch": None,
        "start_time_epoch": 0.0,
        "hourly_rate": None,
    }
    launch_runtime.collect_run(manifest, summarize=False)
    assert (tmp_path / "results" / "collect-me" / "train_gpt.py").read_text(encoding="utf-8") == "snapshot trainer\n"


def test_status_batches_by_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(launch, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(launch_runtime, "RESULTS_DIR", results_dir)
    manifests = [
        {
            "run_id": "run-a",
            "host": "host-1",
            "host_name": "alpha",
            "status": "running",
            "remote_exit_code_path": "/tmp/run-a/exit",
            "remote_pid_path": "/tmp/run-a/pid",
        },
        {
            "run_id": "run-b",
            "host": "host-1",
            "host_name": "alpha",
            "status": "running",
            "remote_exit_code_path": "/tmp/run-b/exit",
            "remote_pid_path": "/tmp/run-b/pid",
        },
        {
            "run_id": "run-c",
            "host": "host-2",
            "host_name": "beta",
            "status": "running",
            "remote_exit_code_path": "/tmp/run-c/exit",
            "remote_pid_path": "/tmp/run-c/pid",
        },
    ]
    for manifest in manifests:
        run_dir = results_dir / manifest["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    ssh_calls: list[tuple[str, str]] = []

    def fake_ssh(host: str, command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        ssh_calls.append((host, command))
        stdout = {
            "host-1": "run-a\tRUNNING\nrun-b\tEXIT:0\n",
            "host-2": "run-c\tUNKNOWN\n",
        }[host]
        return subprocess.CompletedProcess(["ssh", host, command], 0, stdout, "")

    monkeypatch.setattr(launch_runtime, "ssh", fake_ssh)
    assert launch.cmd_status(type("Args", (), {"host": None})()) == 0
    capsys.readouterr()
    assert [host for host, _ in ssh_calls] == ["host-1", "host-2"]
    assert sum(host == "host-1" for host, _ in ssh_calls) == 1


def test_ensure_remote_data_uses_flock(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[str] = []

    def fake_ssh(host: str, command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(["ssh", host, command], 0, "", "")

    monkeypatch.setattr(launch_runtime, "ssh", fake_ssh)
    manifest = {
        "host": "example",
        "resolved_env": {
            "DATA_PATH": "/shared/data/datasets/fineweb10B_sp1024",
            "TOKENIZER_PATH": "/shared/data/tokenizers/fineweb_1024_bpe.model",
        },
    }
    launch_runtime.ensure_remote_data(manifest)
    assert "flock -x /tmp/pgolf_data.lock python3 data/cached_challenge_fineweb.py --variant sp1024" in commands[0]


def test_verify_remote_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[str] = []

    def fake_ssh(host: str, command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(["ssh", host, command], 0, "", "")

    monkeypatch.setattr(launch_runtime, "ssh", fake_ssh)
    launch_runtime.verify_remote_dependencies({"host": "example", "remote_repo_dir": "/remote/repo"})
    assert "import torch; import sentencepiece; import numpy" in commands[0]


def test_start_remote_run_uses_unique_heredoc_and_verifies_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[str] = []

    def fake_ssh(host: str, command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(["ssh", host, command], 0, "4321\n", "")

    monkeypatch.setattr(launch_runtime, "ssh", fake_ssh)
    monkeypatch.setattr(launch_runtime, "save_manifest", lambda manifest: tmp_path / "manifest.json")
    manifest = {
        "run_id": "run-with-EOF",
        "host": "example",
        "gpus": 8,
        "remote_repo_dir": "/remote/repo",
        "remote_pid_path": "/remote/pid",
        "remote_exit_code_path": "/remote/exit_code",
        "remote_wrapper_path": "/remote/run.sh",
        "remote_stdout_path": "/remote/stdout",
        "_forwarded_env": {"SPECIAL": "contains EOF marker"},
        "start_time_epoch": 0.0,
        "hourly_rate": None,
    }
    launch_runtime.start_remote_run(manifest)
    command = commands[0]
    assert "PGOLF_ENV_EOF_" in command
    assert "<<'PGOLF_ENV_EOF_" in command
    assert "sleep 2" in command
    assert "kill -0" in command
    assert manifest["pid"] == 4321
