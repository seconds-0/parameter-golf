from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import time
import textwrap
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "experiments" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare
import compare_diag
import config_utils
import cost
import export_eval
import launch
import launch_runtime
import parse_log
import preflight
import submit
import sweep_queue
import train_schedule
import watchdog

gc_spec = importlib.util.spec_from_file_location("pgolf_gc_test", SCRIPTS_DIR / "gc.py")
assert gc_spec is not None and gc_spec.loader is not None
pgolf_gc = importlib.util.module_from_spec(gc_spec)
gc_spec.loader.exec_module(pgolf_gc)

cache_spec = importlib.util.spec_from_file_location("pgolf_cached_fineweb_test", ROOT / "data" / "cached_challenge_fineweb.py")
assert cache_spec is not None and cache_spec.loader is not None
cached_fineweb = importlib.util.module_from_spec(cache_spec)
cache_spec.loader.exec_module(cached_fineweb)

BASELINE_CONFIG = ROOT / "experiments" / "configs" / "baseline.yaml"
P0_SMOKE_CONFIG = ROOT / "experiments" / "configs" / "proxy_p0_smoke.yaml"
P1_FAST_CONFIG = ROOT / "experiments" / "configs" / "proxy_p1_fast.yaml"
E01_P1_CONFIG = ROOT / "experiments" / "configs" / "phase0_e01_baseline_p1.yaml"
E27_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e27_control_p1.yaml"
E27_DOC_ALIGNED_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e27_doc_aligned_p1.yaml"
E23_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e23_control_p1.yaml"
E23_EMA_0999_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e23_ema0999_p1.yaml"
E23_EMA_09999_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e23_ema09999_p1.yaml"
E32_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e32_control_p1.yaml"
E32_WSD_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e32_wsd_p1.yaml"
E35_WSD_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e35_wsd_control_p1.yaml"
E35_WSD_BETA2_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e35_wsd_beta2_p1.yaml"
E28_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e28_control_p1.yaml"
E28_ASYM3020_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e28_asym3020_p1.yaml"
E28_ASYM3015_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e28_asym3015_p1.yaml"
E28_ASYM2030_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e28_asym2030_p1.yaml"
E30_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e30_control_p1.yaml"
E30_BATCH_SCHEDULE_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e30_batch_schedule_p1.yaml"
E34A_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e34a_control_p1.yaml"
E34A_POLAREXPRESS5_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e34a_polarexpress5_p1.yaml"
E34A_POLAREXPRESS4_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e34a_polarexpress4_p1.yaml"
E34C_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e34c_control_p1.yaml"
E34C_NORMUON_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e34c_normuon_p1.yaml"
E24A_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e24a_control_p1.yaml"
E24A_WD01_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_e24a_wd01_p1.yaml"
CALIBRATION_FULL_CONFIG = ROOT / "experiments" / "configs" / "phase3_calibration_wsd_e28_e30.yaml"
CAL02_BASELINE_FULL_CONFIG = ROOT / "experiments" / "configs" / "phase3_cal02_runpod_baseline_control.yaml"
CAL03_E32_FULL_CONFIG = ROOT / "experiments" / "configs" / "phase3_cal03_e32_full.yaml"
CAL04_E32_E28_FULL_CONFIG = ROOT / "experiments" / "configs" / "phase3_cal04_e32_e28_full.yaml"
CAL06_E30_COMPILED_CONTROL_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_cal06_e30_compiled_control_p1.yaml"
CAL06_E30_COMPILED_BATCH_P1_CONFIG = ROOT / "experiments" / "configs" / "phase1_cal06_e30_compiled_batch_schedule_p1.yaml"
CAL07_E30_PHASEAWARE_CONTROL_P2_CONFIG = ROOT / "experiments" / "configs" / "phase2_cal07_e30_phaseaware_control_p2.yaml"
CAL07_E30_PHASEAWARE_BATCH_P2_CONFIG = ROOT / "experiments" / "configs" / "phase2_cal07_e30_phaseaware_batch_schedule_p2.yaml"
SWEEP_CONFIG = ROOT / "experiments" / "configs" / "sweep_lr.yaml"
BASELINE_LOG = ROOT / "records" / "track_10min_16mb" / "2026-03-17_NaiveBaseline" / "train.log"


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return path


def write_metrics(
    path: Path,
    *,
    run_id: str,
    val_bpb: float,
    val_loss: float,
    status: str = "success",
    prequant_val_bpb: float | None = None,
    qgap_bpb: float | None = None,
    artifact_slack_bytes: int = 123456,
    tok_s: float = 1_500_000.0,
    uncompiled_check_delta_bpb: float | None = None,
    reloaded_postquant_val_bpb: float | None = None,
    supervised_target_fraction: float | None = None,
) -> None:
    prequant = prequant_val_bpb if prequant_val_bpb is not None else val_bpb - 0.01
    reloaded_postquant = reloaded_postquant_val_bpb if reloaded_postquant_val_bpb is not None else None
    payload = {
        "run_id": run_id,
        "status": status,
        "train_steps": [{"step": 123, "train_time_ms": 4567.0, "step_avg_ms": 37.1, "train_loss": 2.22}],
        "config": {"model_params": 999},
        "tok_s": tok_s,
        "final": {
            "val_bpb": val_bpb,
            "val_loss": val_loss,
            "prequant_val_bpb": prequant,
            "qgap_bpb": qgap_bpb if qgap_bpb is not None else 0.01,
            "total_submission_bytes": 123456,
            "artifact_slack_bytes": artifact_slack_bytes,
            "peak_memory_allocated_mib": 2048,
            "stop_step": 123,
        },
    }
    if supervised_target_fraction is not None:
        payload["final"]["supervised_target_fraction"] = supervised_target_fraction
    if uncompiled_check_delta_bpb is not None:
        payload["final"]["uncompiled_check_delta_bpb"] = uncompiled_check_delta_bpb
    if reloaded_postquant is not None:
        payload["final"]["reloaded_postquant_val_bpb"] = reloaded_postquant
        payload["final"]["reloaded_postquant_delta_bpb"] = reloaded_postquant - val_bpb
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


def test_validate_p0_smoke_config() -> None:
    result = config_utils.validate_config(P0_SMOKE_CONFIG)
    assert result.ok
    env = result.runs[0].env
    assert env["VAL_LOSS_EVERY"] == "0"
    assert env["MAX_WALLCLOCK_SECONDS"] == "45"


def test_validate_p1_fast_config() -> None:
    result = config_utils.validate_config(P1_FAST_CONFIG)
    assert result.ok
    env = result.runs[0].env
    assert env["VAL_LOSS_EVERY"] == "0"
    assert env["MAX_WALLCLOCK_SECONDS"] == "300"
    assert env["TRAIN_LOG_EVERY"] == "25"


def test_validate_e01_p1_control_config() -> None:
    result = config_utils.validate_config(E01_P1_CONFIG)
    assert result.ok
    assert len(result.runs) == 2
    assert {run.env["SEED"] for run in result.runs} == {"1337", "2024"}


def test_validate_e27_control_config() -> None:
    result = config_utils.validate_config(E27_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["SEED"] == "1337"
    assert run.metadata["hypothesis_id"] == "E27-control"


def test_validate_e27_doc_aligned_config() -> None:
    result = config_utils.validate_config(E27_DOC_ALIGNED_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["SEED"] == "1337"
    assert run.env["DOC_ALIGNED_BATCHING"] == "1"
    assert run.metadata["hypothesis_id"] == "E27"


def test_validate_e23_control_config() -> None:
    result = config_utils.validate_config(E23_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["SEED"] == "1337"
    assert "EMA_EXPORT" not in run.env
    assert run.metadata["hypothesis_id"] == "E23-control"


def test_validate_e23_ema0999_config() -> None:
    result = config_utils.validate_config(E23_EMA_0999_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["EMA_EXPORT"] == "1"
    assert run.env["EMA_DECAY"] == "0.999"
    assert run.metadata["hypothesis_id"] == "E23-ema0999"


def test_validate_e23_ema09999_config() -> None:
    result = config_utils.validate_config(E23_EMA_09999_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["EMA_EXPORT"] == "1"
    assert run.env["EMA_DECAY"] == "0.9999"
    assert run.metadata["hypothesis_id"] == "E23-ema09999"


def test_validate_e32_control_config() -> None:
    result = config_utils.validate_config(E32_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["SEED"] == "1337"
    assert run.env["LR_SCHEDULE"] == "baseline"
    assert run.metadata["hypothesis_id"] == "E32-control"


def test_validate_e32_wsd_config() -> None:
    result = config_utils.validate_config(E32_WSD_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["SEED"] == "1337"
    assert run.env["LR_SCHEDULE"] == "wsd"
    assert run.env["WSD_WARMUP_FRAC"] == "0.01"
    assert run.env["WSD_STABLE_FRAC"] == "0.75"
    assert run.env["WSD_DECAY_STYLE"] == "cosine"
    assert run.metadata["hypothesis_id"] == "E32"


def test_validate_e35_wsd_control_config() -> None:
    result = config_utils.validate_config(E35_WSD_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["LR_SCHEDULE"] == "wsd"
    assert "ENABLE_COOLDOWN_BETA2" not in run.env
    assert run.metadata["hypothesis_id"] == "E35-control"


def test_validate_e35_wsd_beta2_config() -> None:
    result = config_utils.validate_config(E35_WSD_BETA2_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["LR_SCHEDULE"] == "wsd"
    assert run.env["ENABLE_COOLDOWN_BETA2"] == "1"
    assert run.env["COOLDOWN_BETA2"] == "0.98"
    assert run.metadata["hypothesis_id"] == "E35"


def test_validate_e28_control_config() -> None:
    result = config_utils.validate_config(E28_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["LR_SCHEDULE"] == "wsd"
    assert "LOGIT_SOFTCAP_POS" not in run.env
    assert run.metadata["hypothesis_id"] == "E28-control"


def test_validate_e28_asym3020_config() -> None:
    result = config_utils.validate_config(E28_ASYM3020_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["LOGIT_SOFTCAP_POS"] == "30"
    assert run.env["LOGIT_SOFTCAP_NEG"] == "20"
    assert run.metadata["hypothesis_id"] == "E28-3020"


def test_validate_e28_asym3015_config() -> None:
    result = config_utils.validate_config(E28_ASYM3015_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["LOGIT_SOFTCAP_POS"] == "30"
    assert run.env["LOGIT_SOFTCAP_NEG"] == "15"
    assert run.metadata["hypothesis_id"] == "E28-3015"


def test_validate_e28_asym2030_config() -> None:
    result = config_utils.validate_config(E28_ASYM2030_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["LOGIT_SOFTCAP_POS"] == "20"
    assert run.env["LOGIT_SOFTCAP_NEG"] == "30"
    assert run.metadata["hypothesis_id"] == "E28-2030"


def test_validate_e30_control_config() -> None:
    result = config_utils.validate_config(E30_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["LR_SCHEDULE"] == "wsd"
    assert run.env["LOGIT_SOFTCAP_POS"] == "20"
    assert run.env["LOGIT_SOFTCAP_NEG"] == "30"
    assert "BATCH_SCHEDULE" not in run.env
    assert run.metadata["hypothesis_id"] == "E30-control"


def test_validate_e30_batch_schedule_config() -> None:
    result = config_utils.validate_config(E30_BATCH_SCHEDULE_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["LR_SCHEDULE"] == "wsd"
    assert run.env["LOGIT_SOFTCAP_POS"] == "20"
    assert run.env["LOGIT_SOFTCAP_NEG"] == "30"
    assert run.env["BATCH_SCHEDULE"] == "0.3:131072,1.0:524288"
    assert run.metadata["hypothesis_id"] == "E30"


def test_validate_e34a_control_config() -> None:
    result = config_utils.validate_config(E34A_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["MUON_ORTHO_BACKEND"] == "newtonschulz5"
    assert run.env["MUON_BACKEND_STEPS"] == "5"
    assert run.env["BATCH_SCHEDULE"] == "0.3:131072,1.0:524288"
    assert run.metadata["hypothesis_id"] == "E34a-control"


def test_validate_e34a_polarexpress5_config() -> None:
    result = config_utils.validate_config(E34A_POLAREXPRESS5_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["MUON_ORTHO_BACKEND"] == "polar_express"
    assert run.env["MUON_BACKEND_STEPS"] == "5"
    assert run.metadata["hypothesis_id"] == "E34a-5"


def test_validate_e34a_polarexpress4_config() -> None:
    result = config_utils.validate_config(E34A_POLAREXPRESS4_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["MUON_ORTHO_BACKEND"] == "polar_express"
    assert run.env["MUON_BACKEND_STEPS"] == "4"
    assert run.metadata["hypothesis_id"] == "E34a-4"


def test_validate_e34c_control_config() -> None:
    result = config_utils.validate_config(E34C_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["MUON_ADAPTIVE_MODE"] == "none"
    assert run.metadata["hypothesis_id"] == "E34c-control"


def test_validate_e34c_normuon_config() -> None:
    result = config_utils.validate_config(E34C_NORMUON_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["MUON_ADAPTIVE_MODE"] == "normuon"
    assert run.env["MUON_ADAPTIVE_BETA2"] == "0.95"
    assert run.metadata["hypothesis_id"] == "E34c"


def test_validate_e24a_control_config() -> None:
    result = config_utils.validate_config(E24A_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["MUON_ORTHO_BACKEND"] == "newtonschulz5"
    assert run.env["MUON_ADAPTIVE_MODE"] == "none"
    assert run.metadata["hypothesis_id"] == "E24a-control"


def test_validate_e24a_wd01_config() -> None:
    result = config_utils.validate_config(E24A_WD01_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.env["FIXED_WEIGHT_DECAY"] == "0.1"
    assert run.metadata["hypothesis_id"] == "E24a-0.1"


def test_validate_full_calibration_config() -> None:
    result = config_utils.validate_config(CALIBRATION_FULL_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.label == "phase3_calibration_wsd_e28_e30"
    assert run.metadata["hypothesis_id"] == "CAL-01"
    assert run.env["LR_SCHEDULE"] == "wsd"
    assert run.env["LOGIT_SOFTCAP_POS"] == "20"
    assert run.env["LOGIT_SOFTCAP_NEG"] == "30"
    assert run.env["BATCH_SCHEDULE"] == "0.3:131072,1.0:524288"
    assert run.env["MAX_WALLCLOCK_SECONDS"] == "600"


def test_validate_cal02_baseline_full_config() -> None:
    result = config_utils.validate_config(CAL02_BASELINE_FULL_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.metadata["hypothesis_id"] == "CAL-02"
    assert run.env["WATCHDOG_POLICY"] == "calibration"
    assert "LR_SCHEDULE" not in run.env


def test_validate_cal03_e32_full_config() -> None:
    result = config_utils.validate_config(CAL03_E32_FULL_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.metadata["hypothesis_id"] == "CAL-03"
    assert run.env["WATCHDOG_POLICY"] == "calibration"
    assert run.env["LR_SCHEDULE"] == "wsd"
    assert "LOGIT_SOFTCAP_POS" not in run.env


def test_validate_cal04_e32_e28_full_config() -> None:
    result = config_utils.validate_config(CAL04_E32_E28_FULL_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.metadata["hypothesis_id"] == "CAL-04"
    assert run.env["WATCHDOG_POLICY"] == "calibration"
    assert run.env["LOGIT_SOFTCAP_POS"] == "20"
    assert run.env["LOGIT_SOFTCAP_NEG"] == "30"
    assert "BATCH_SCHEDULE" not in run.env


def test_validate_cal06_e30_compiled_control_config() -> None:
    result = config_utils.validate_config(CAL06_E30_COMPILED_CONTROL_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.metadata["hypothesis_id"] == "CAL-06-control"
    assert "TORCHDYNAMO_DISABLE" not in run.env
    assert "BATCH_SCHEDULE" not in run.env


def test_validate_cal06_e30_compiled_batch_config() -> None:
    result = config_utils.validate_config(CAL06_E30_COMPILED_BATCH_P1_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.metadata["hypothesis_id"] == "CAL-06"
    assert "TORCHDYNAMO_DISABLE" not in run.env
    assert run.env["BATCH_SCHEDULE"] == "0.3:131072,1.0:524288"


def test_validate_cal07_e30_phaseaware_control_config() -> None:
    result = config_utils.validate_config(CAL07_E30_PHASEAWARE_CONTROL_P2_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.metadata["hypothesis_id"] == "CAL-07-control"
    assert run.env["ITERATIONS"] == "3000"
    assert run.env["VAL_LOSS_EVERY"] == "100"
    assert run.env["WATCHDOG_POLICY"] == "calibration"
    assert run.env["TORCHDYNAMO_DISABLE"] == "1"
    assert "BATCH_SCHEDULE" not in run.env


def test_validate_cal07_e30_phaseaware_batch_config() -> None:
    result = config_utils.validate_config(CAL07_E30_PHASEAWARE_BATCH_P2_CONFIG)
    assert result.ok
    run = result.runs[0]
    assert run.metadata["hypothesis_id"] == "CAL-07"
    assert run.env["ITERATIONS"] == "3000"
    assert run.env["VAL_LOSS_EVERY"] == "100"
    assert run.env["WATCHDOG_POLICY"] == "calibration"
    assert run.env["TORCHDYNAMO_DISABLE"] == "1"
    assert run.env["BATCH_SCHEDULE"] == "0.3:131072,1.0:524288"


def test_export_eval_parses_env_overrides() -> None:
    parsed = export_eval.parse_env_overrides(["INT8_CLIP_PERCENTILE=99.99995", "INT8_KEEP_FLOAT_MAX_NUMEL=32768"])
    assert parsed == {
        "INT8_CLIP_PERCENTILE": "99.99995",
        "INT8_KEEP_FLOAT_MAX_NUMEL": "32768",
    }


def test_export_eval_rejects_bad_env_override() -> None:
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        export_eval.parse_env_overrides(["INT8_CLIP_PERCENTILE"])


def test_export_eval_default_outputs_do_not_clobber_run_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final_model.pt"
    assert export_eval.default_artifact_out(checkpoint).name == "final_model.export_eval.int8.ptz"
    assert export_eval.default_metrics_out(checkpoint).name == "final_model.export_eval.json"


def test_lr_schedule_multiplier_baseline_and_wsd() -> None:
    baseline = train_schedule.lr_schedule_multiplier(
        schedule="baseline",
        step=900,
        iterations=1000,
        warmdown_iters=200,
        elapsed_ms=90_000.0,
        max_wallclock_ms=None,
        wsd_warmup_frac=0.01,
        wsd_stable_frac=0.75,
        wsd_decay_style="cosine",
    )
    assert baseline == pytest.approx(0.5)

    wsd_warmup = train_schedule.lr_schedule_multiplier(
        schedule="wsd",
        step=5,
        iterations=1000,
        warmdown_iters=200,
        elapsed_ms=500.0,
        max_wallclock_ms=100_000.0,
        wsd_warmup_frac=0.01,
        wsd_stable_frac=0.75,
        wsd_decay_style="cosine",
    )
    assert wsd_warmup == pytest.approx(0.5)

    wsd_stable = train_schedule.lr_schedule_multiplier(
        schedule="wsd",
        step=400,
        iterations=1000,
        warmdown_iters=200,
        elapsed_ms=40_000.0,
        max_wallclock_ms=100_000.0,
        wsd_warmup_frac=0.01,
        wsd_stable_frac=0.75,
        wsd_decay_style="cosine",
    )
    assert wsd_stable == pytest.approx(1.0)

    wsd_decay = train_schedule.lr_schedule_multiplier(
        schedule="wsd",
        step=990,
        iterations=1000,
        warmdown_iters=200,
        elapsed_ms=88_000.0,
        max_wallclock_ms=100_000.0,
        wsd_warmup_frac=0.01,
        wsd_stable_frac=0.75,
        wsd_decay_style="cosine",
    )
    assert wsd_decay == pytest.approx(0.5)


def test_batch_schedule_parser_and_stage_selection() -> None:
    trainer_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_batch_schedule", ROOT / "train_gpt.py")
    assert trainer_spec is not None and trainer_spec.loader is not None
    trainer = importlib.util.module_from_spec(trainer_spec)
    trainer_spec.loader.exec_module(trainer)

    stages = trainer.parse_batch_schedule("1.0:524288,0.3:131072")
    assert stages == [(0.3, 131072), (1.0, 524288)]
    assert trainer.scheduled_batch_tokens(0, 100, stages, 999) == 131072
    assert trainer.scheduled_batch_tokens(29, 100, stages, 999) == 131072
    assert trainer.scheduled_batch_tokens(30, 100, stages, 999) == 524288
    assert trainer.scheduled_batch_tokens(99, 100, stages, 999) == 524288


def test_beta2_for_schedule_only_changes_during_wsd_decay() -> None:
    before_decay = train_schedule.beta2_for_schedule(
        base_beta2=0.95,
        cooldown_beta2=0.98,
        enable_cooldown_beta2=True,
        schedule="wsd",
        step=700,
        iterations=1000,
        elapsed_ms=70000.0,
        max_wallclock_ms=100_000.0,
        wsd_warmup_frac=0.01,
        wsd_stable_frac=0.75,
    )
    in_decay = train_schedule.beta2_for_schedule(
        base_beta2=0.95,
        cooldown_beta2=0.98,
        enable_cooldown_beta2=True,
        schedule="wsd",
        step=900,
        iterations=1000,
        elapsed_ms=90000.0,
        max_wallclock_ms=100_000.0,
        wsd_warmup_frac=0.01,
        wsd_stable_frac=0.75,
    )
    baseline_schedule = train_schedule.beta2_for_schedule(
        base_beta2=0.95,
        cooldown_beta2=0.98,
        enable_cooldown_beta2=True,
        schedule="baseline",
        step=900,
        iterations=1000,
        elapsed_ms=90000.0,
        max_wallclock_ms=100_000.0,
        wsd_warmup_frac=0.01,
        wsd_stable_frac=0.75,
    )
    assert before_decay == pytest.approx(0.95)
    assert in_decay == pytest.approx(0.98)
    assert baseline_schedule == pytest.approx(0.95)


def test_apply_logit_softcap_matches_symmetric_formula_and_supports_asymmetry() -> None:
    import torch

    softcap_spec = importlib.util.spec_from_file_location("pgolf_logit_softcap_test", ROOT / "logit_softcap.py")
    assert softcap_spec is not None and softcap_spec.loader is not None
    logit_softcap = importlib.util.module_from_spec(softcap_spec)
    softcap_spec.loader.exec_module(logit_softcap)

    x = torch.tensor([-60.0, -10.0, 0.0, 10.0, 60.0], dtype=torch.float32)
    symmetric = logit_softcap.apply_logit_softcap(x, 30.0, 30.0)
    expected = 30.0 * torch.tanh(x / 30.0)
    assert torch.allclose(symmetric, expected)

    asymmetric = logit_softcap.apply_logit_softcap(x, 30.0, 20.0)
    assert asymmetric[0].abs() < symmetric[0].abs()
    assert asymmetric[-1] == pytest.approx(symmetric[-1].item())


def test_block_only_threads_tok_emb_when_value_embed_gate_is_enabled() -> None:
    import torch

    train_gpt_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_block_test", ROOT / "train_gpt.py")
    assert train_gpt_spec is not None and train_gpt_spec.loader is not None
    train_gpt = importlib.util.module_from_spec(train_gpt_spec)
    train_gpt_spec.loader.exec_module(train_gpt)

    class DummyAttn(torch.nn.Module):
        def __init__(self, gate_enabled: bool):
            super().__init__()
            self.ve_gate_w = torch.nn.Parameter(torch.zeros(1)) if gate_enabled else None
            self.seen_tok_emb = "unset"

        def forward(self, x: torch.Tensor, tok_emb: torch.Tensor | None = None) -> torch.Tensor:
            self.seen_tok_emb = tok_emb
            return torch.zeros_like(x)

    block = train_gpt.Block(8, 2, 1, 2, 10000.0, 1.5, value_embed_gate=False)
    block.attn = DummyAttn(gate_enabled=False)
    x = torch.randn(2, 4, 8)
    x0 = torch.randn(2, 4, 8)
    block(x, x0)
    assert block.attn.seen_tok_emb is None

    gated_block = train_gpt.Block(8, 2, 1, 2, 10000.0, 1.5, value_embed_gate=True)
    gated_block.attn = DummyAttn(gate_enabled=True)
    gated_block(x, x0)
    assert gated_block.attn.seen_tok_emb is x0


def test_muon_orthogonalize_supports_polarexpress_backend() -> None:
    import torch

    train_gpt_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_muon_backend_test", ROOT / "train_gpt.py")
    assert train_gpt_spec is not None and train_gpt_spec.loader is not None
    train_gpt = importlib.util.module_from_spec(train_gpt_spec)
    train_gpt_spec.loader.exec_module(train_gpt)

    g = torch.randn(8, 4)
    ns = train_gpt.muon_orthogonalize(g, backend="newtonschulz5", steps=5)
    pe = train_gpt.muon_orthogonalize(g, backend="polar_express", steps=5)
    assert ns.shape == g.shape
    assert pe.shape == g.shape
    assert ns.dtype == torch.bfloat16
    assert pe.dtype == torch.bfloat16


def test_muon_adaptive_scaling_supports_normuon_mode() -> None:
    import torch

    train_gpt_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_muon_adaptive_test", ROOT / "train_gpt.py")
    assert train_gpt_spec is not None and train_gpt_spec.loader is not None
    train_gpt = importlib.util.module_from_spec(train_gpt_spec)
    train_gpt_spec.loader.exec_module(train_gpt)

    update = torch.randn(16, 8, dtype=torch.bfloat16)
    second_momentum = torch.zeros_like(update[..., :1], dtype=torch.float32)
    scaled = train_gpt.muon_apply_adaptive_scaling(update, second_momentum, mode="normuon", beta2=0.95)

    assert scaled.shape == update.shape
    assert second_momentum.shape == update[..., :1].shape
    assert second_momentum.dtype == torch.float32
    assert torch.isfinite(scaled).all()
    assert torch.isfinite(second_momentum).all()
    assert scaled.norm().item() == pytest.approx(update.norm().item(), rel=1e-2)


def test_muon_adaptive_scaling_tolerates_dtype_mismatch() -> None:
    import torch

    train_gpt_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_muon_adaptive_dtype_test", ROOT / "train_gpt.py")
    assert train_gpt_spec is not None and train_gpt_spec.loader is not None
    train_gpt = importlib.util.module_from_spec(train_gpt_spec)
    train_gpt_spec.loader.exec_module(train_gpt)

    update = torch.randn(12, 6, dtype=torch.bfloat16)
    second_momentum = torch.zeros_like(update[..., :1])
    scaled = train_gpt.muon_apply_adaptive_scaling(update, second_momentum, mode="normuon", beta2=0.95)

    assert scaled.shape == update.shape
    assert second_momentum.dtype == torch.bfloat16
    assert torch.isfinite(scaled).all()
    assert torch.isfinite(second_momentum).all()


def test_export_eval_resolve_replay_inputs_prefers_manifest_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "final_model.pt"
    checkpoint.write_bytes(b"checkpoint")
    trainer_snapshot = run_dir / "train_gpt.py"
    trainer_snapshot.write_text("VALUE = 'snapshot'\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "resolved_env": {
                    "RUN_ID": "from-manifest",
                    "DATA_PATH": "/remote/data",
                    "TOKENIZER_PATH": "/remote/tokenizer.model",
                },
                "group": "track-b",
                "hypothesis_id": "X-05",
                "notes": "manifest wins",
            }
        ),
        encoding="utf-8",
    )

    replay = export_eval.resolve_replay_inputs(checkpoint, None)

    assert replay.env_source == "manifest"
    assert replay.env["RUN_ID"] == "from-manifest"
    assert replay.metadata == {"group": "track-b", "hypothesis_id": "X-05", "notes": "manifest wins"}
    assert replay.trainer_path == trainer_snapshot


def test_export_eval_load_trainer_uses_explicit_snapshot_and_scrubs_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    trainer_path = fake_repo / "train_gpt.py"
    trainer_path.write_text(
        "VALUE = __import__('os').environ.get('EXPORT_EVAL_SENTINEL', 'missing')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(export_eval, "extract_allowlist", lambda: {"EXPORT_EVAL_SENTINEL"})
    monkeypatch.setenv("EXPORT_EVAL_SENTINEL", "ambient")

    trainer = export_eval.load_trainer_with_env(trainer_path, {"EXPORT_EVAL_SENTINEL": "from-replay"})

    assert trainer.VALUE == "from-replay"
    assert os.environ["EXPORT_EVAL_SENTINEL"] == "ambient"


def test_export_eval_maybe_call_noops_for_missing_helper() -> None:
    export_eval.maybe_call(types.SimpleNamespace(), "missing_helper", 1, 2, 3)


def test_export_eval_compat_raw_checkpoint_state_dict_backfills_rotary_inv_freq() -> None:
    class FakeModel:
        def state_dict(self):
            return {
                "blocks.0.attn.rotary.inv_freq": "buf0",
                "blocks.1.attn.rotary.inv_freq": "buf1",
                "tok_emb.weight": "weight",
            }

    patched = export_eval.compat_raw_checkpoint_state_dict(
        FakeModel(),
        {
            "tok_emb.weight": "checkpoint-weight",
            "blocks.1.attn.rotary.inv_freq": "checkpoint-buf1",
        },
    )

    assert patched["tok_emb.weight"] == "checkpoint-weight"
    assert patched["blocks.0.attn.rotary.inv_freq"] == "buf0"
    assert patched["blocks.1.attn.rotary.inv_freq"] == "checkpoint-buf1"


def test_cached_fineweb_defaults_to_full_train_split() -> None:
    args = cached_fineweb.build_parser().parse_args(["--variant", "sp1024"])
    assert cached_fineweb.resolve_train_shards(args, 195) == 195


def test_cached_fineweb_respects_explicit_train_shard_prefix() -> None:
    args = cached_fineweb.build_parser().parse_args(["--variant", "sp1024", "--train-shards", "80"])
    assert cached_fineweb.resolve_train_shards(args, 195) == 80


def test_trainer_model_serializes_all_registered_buffers() -> None:
    trainer_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_test", ROOT / "train_gpt.py")
    assert trainer_spec is not None and trainer_spec.loader is not None
    trainer = importlib.util.module_from_spec(trainer_spec)
    trainer_spec.loader.exec_module(trainer)

    model = trainer.GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    ).bfloat16()
    for module in model.modules():
        if isinstance(module, trainer.CastedLinear):
            module.float()
    trainer.restore_low_dim_params_to_fp32(model)

    state_keys = set(model.state_dict().keys())
    missing_buffers = [name for name, _ in model.named_buffers() if name not in state_keys]

    assert missing_buffers == []


def test_rotary_keeps_inv_freq_fp32_after_model_bfloat16() -> None:
    trainer_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_rotary_dtype", ROOT / "train_gpt.py")
    assert trainer_spec is not None and trainer_spec.loader is not None
    trainer = importlib.util.module_from_spec(trainer_spec)
    trainer_spec.loader.exec_module(trainer)

    model = trainer.GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    ).bfloat16()

    rotary_dtypes = {
        name: buffer.dtype
        for name, buffer in model.named_buffers()
        if name.endswith("rotary.inv_freq")
    }

    assert rotary_dtypes
    assert set(rotary_dtypes.values()) == {trainer.torch.float32}


def test_rotary_cache_rebuild_is_invariant_after_reset() -> None:
    trainer_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_rotary_reset", ROOT / "train_gpt.py")
    assert trainer_spec is not None and trainer_spec.loader is not None
    trainer = importlib.util.module_from_spec(trainer_spec)
    trainer_spec.loader.exec_module(trainer)

    rotary = trainer.Rotary(64).bfloat16()
    device = trainer.torch.device("cpu")
    first_cos, first_sin = rotary(1024, device, trainer.torch.bfloat16)
    trainer.reset_rotary_caches(rotary)
    second_cos, second_sin = rotary(1024, device, trainer.torch.bfloat16)

    assert rotary.inv_freq.dtype == trainer.torch.float32
    assert trainer.torch.equal(first_cos, second_cos)
    assert trainer.torch.equal(first_sin, second_sin)


def test_ema_export_updates_and_casts_back_to_model_dtype() -> None:
    ema_spec = importlib.util.spec_from_file_location("pgolf_ema_export_test", ROOT / "ema_export.py")
    assert ema_spec is not None and ema_spec.loader is not None
    ema_export = importlib.util.module_from_spec(ema_spec)
    ema_spec.loader.exec_module(ema_export)

    trainer_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_ema_test", ROOT / "train_gpt.py")
    assert trainer_spec is not None and trainer_spec.loader is not None
    trainer = importlib.util.module_from_spec(trainer_spec)
    trainer_spec.loader.exec_module(trainer)

    model = trainer.GPT(
        vocab_size=1024,
        num_layers=2,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    ).bfloat16()
    ema_params = ema_export.init_ema_parameters(model)
    sample_name = next(iter(ema_params))
    assert ema_params[sample_name].dtype == trainer.torch.float32

    with trainer.torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point():
                param.add_(1.0)
                break

    ema_export.update_ema_parameters(ema_params, model, 0.5)
    export_state = ema_export.build_export_state_dict(model, ema_params)
    assert export_state[sample_name].dtype == model.state_dict()[sample_name].dtype


def test_document_window_stream_pads_short_docs_without_crossing_boundaries() -> None:
    batching_spec = importlib.util.spec_from_file_location("pgolf_doc_batching_test", ROOT / "doc_batching.py")
    assert batching_spec is not None and batching_spec.loader is not None
    batching = importlib.util.module_from_spec(batching_spec)
    batching_spec.loader.exec_module(batching)

    tokens = iter([1, 11, 12, 1, 21, 22, 23, 1, 31, 32])
    stream = batching.DocumentWindowStream(lambda: next(tokens), boundary_token=1)
    x, y = stream.take(2, 4, pad_token=1, ignore_index=-100)

    assert x.tolist() == [[1, 11, 1, 1], [1, 21, 22, 1]]
    assert y.tolist() == [[11, 12, -100, -100], [21, 22, 23, -100]]


def test_document_window_stream_splits_long_docs_across_sequences() -> None:
    batching_spec = importlib.util.spec_from_file_location("pgolf_doc_batching_long_test", ROOT / "doc_batching.py")
    assert batching_spec is not None and batching_spec.loader is not None
    batching = importlib.util.module_from_spec(batching_spec)
    batching_spec.loader.exec_module(batching)

    tokens = iter([1, 10, 11, 12, 13, 14, 15, 1, 20, 21])
    stream = batching.DocumentWindowStream(lambda: next(tokens), boundary_token=1)
    x, y = stream.take(2, 3, pad_token=1, ignore_index=-100)

    assert x.tolist() == [[1, 10, 11], [12, 13, 14]]
    assert y.tolist() == [[10, 11, 12], [13, 14, 15]]


def test_trainer_forward_ignores_masked_targets() -> None:
    trainer_spec = importlib.util.spec_from_file_location("pgolf_train_gpt_ignore_index", ROOT / "train_gpt.py")
    assert trainer_spec is not None and trainer_spec.loader is not None
    trainer = importlib.util.module_from_spec(trainer_spec)
    trainer_spec.loader.exec_module(trainer)

    model = trainer.GPT(
        vocab_size=1024,
        num_layers=2,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    x = trainer.torch.randint(0, 1024, (2, 8))
    y = trainer.torch.randint(0, 1024, (2, 8))
    y[:, -2:] = trainer.LOSS_IGNORE_INDEX
    loss = model(x, y)

    assert trainer.torch.isfinite(loss)

def test_validate_exporter_env_vars_are_allowlisted(tmp_path: Path) -> None:
    config_path = write_yaml(
        tmp_path / "exporter_envs.yaml",
        """
        name: exporter_envs
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
          INT8_CLIP_PERCENTILE: "99.99995"
          INT8_KEEP_FLOAT_MAX_NUMEL: "32768"
        """,
    )
    result = config_utils.validate_config(config_path)
    assert result.ok
    assert not any("INT8_CLIP_PERCENTILE" in warning for warning in result.warnings)
    assert not any("INT8_KEEP_FLOAT_MAX_NUMEL" in warning for warning in result.warnings)


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


def test_warn_missing_metadata(tmp_path: Path) -> None:
    config_path = write_yaml(
        tmp_path / "missing_meta.yaml",
        """
        name: missing_meta
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
        """,
    )
    result = config_utils.validate_config(config_path)
    assert result.ok
    assert any("Missing optional metadata field hypothesis_id" in warning for warning in result.warnings)


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
                "step:1/10 train_loss:1.2e-05 train_time:3.4e+02ms step_avg:5.6e+02ms tok_s:7.8e+03 train_tokens_seen:5.2e+05",
                "step:1/10 val_loss:9.1e-06 val_bpb:1.2e+00 train_time:1.5e+02ms step_avg:2.5e+02ms",
                "final_prequant_exact val_loss:9.9e-06 val_bpb:1.1e+00 train_tokens_seen:5.2e+05 eval_time:111ms",
                "Serialized model int8+zlib: 123 bytes (payload:45 raw_torch:67 payload_ratio:8.9x)",
                "Total submission size int8+zlib: 1000 bytes",
                "final_int8_zlib_roundtrip_exact val_loss:1.2e-05 val_bpb:3.4e-06",
                "quantization_delta_exact val_loss:2.1e-06 val_bpb:4.5e-06 train_tokens_seen:5.2e+05",
            ]
        )
    )
    assert parsed["train_steps"][0]["train_loss"] == pytest.approx(1.2e-05)
    assert parsed["train_steps"][0]["train_time_ms"] == pytest.approx(340.0)
    assert parsed["train_steps"][0]["train_tokens_seen"] == 520000
    assert parsed["val_steps"][0]["val_loss"] == pytest.approx(9.1e-06)
    assert parsed["final"]["val_bpb"] == pytest.approx(3.4e-06)
    assert parsed["final"]["prequant_val_bpb"] == pytest.approx(1.1)
    assert parsed["final"]["payload_bytes"] == 45
    assert parsed["final"]["payload_ratio"] == pytest.approx(8.9)
    assert parsed["final"]["artifact_slack_bytes"] == 15_999_000
    assert parsed["train_tokens_seen"] == 520000


def test_parse_derives_qgap_without_explicit_delta_line() -> None:
    parsed = parse_log.parse_log(
        "\n".join(
            [
                "final_prequant_exact val_loss:2.00000000 val_bpb:1.00000000 train_tokens_seen:1024 eval_time:1ms",
                "Total submission size int8+zlib: 2048 bytes",
                "final_int8_zlib_roundtrip_exact val_loss:2.50000000 val_bpb:1.25000000",
            ]
        )
    )
    assert parsed["final"]["qgap_loss"] == pytest.approx(0.5)
    assert parsed["final"]["qgap_bpb"] == pytest.approx(0.25)
    assert parsed["final"]["artifact_slack_bytes"] == 15_997_952


def test_parse_replay_trust_signals() -> None:
    parsed = parse_log.parse_log(
        "\n".join(
            [
                "final_prequant_exact val_loss:2.00000000 val_bpb:1.00000000 train_tokens_seen:1024 eval_time:1ms",
                "uncompiled_check val_loss:2.40000000 val_bpb:1.25000000 delta_loss:0.40000000 delta_bpb:0.25000000 eval_time:3ms",
                "checkpoint_save_verify max_abs_diff:0.00000000e+00 tensors_mismatched:0",
                "reloaded_prequant_exact val_loss:2.30000000 val_bpb:1.20000000 delta_loss:0.30000000 delta_bpb:0.20000000 eval_time:4ms",
                "Serialized model int8+zlib: 123 bytes (payload:45 raw_torch:67 payload_ratio:8.9x)",
                "Total submission size int8+zlib: 1000 bytes",
                "final_int8_zlib_roundtrip_exact val_loss:2.50000000 val_bpb:1.25000000",
                "reloaded_int8_zlib_roundtrip_exact val_loss:2.60000000 val_bpb:1.35000000 delta_loss:0.10000000 delta_bpb:0.10000000 eval_time:5ms",
                "quantization_delta_exact val_loss:0.50000000 val_bpb:0.25000000 train_tokens_seen:1024",
            ]
        )
    )
    final = parsed["final"]
    assert final["uncompiled_check_val_bpb"] == pytest.approx(1.25)
    assert final["uncompiled_check_delta_bpb"] == pytest.approx(0.25)
    assert final["checkpoint_save_verify_max_abs_diff"] == pytest.approx(0.0)
    assert final["checkpoint_save_verify_tensors_mismatched"] == 0
    assert final["reloaded_prequant_val_bpb"] == pytest.approx(1.2)
    assert final["reloaded_prequant_delta_bpb"] == pytest.approx(0.2)
    assert final["reloaded_postquant_val_bpb"] == pytest.approx(1.35)
    assert final["reloaded_postquant_delta_bpb"] == pytest.approx(0.1)
    assert final["replay_trust_prequant_delta_bpb"] == pytest.approx(0.2)
    assert final["replay_trust_postquant_delta_bpb"] == pytest.approx(0.1)


def test_parse_document_aligned_training_stats() -> None:
    parsed = parse_log.parse_log(
        "\n".join(
            [
                "doc_aligned_batching:True boundary_token_id:1 ema_export:True ema_decay:0.999900",
                "seed:1337 lr_schedule:wsd wsd_warmup_frac:0.010000 wsd_stable_frac:0.750000 wsd_decay_style:cosine enable_cooldown_beta2:True cooldown_beta2:0.980000",
                "step:1/10 train_loss:1.2000 train_time:340ms step_avg:560ms tok_s:7800 train_tokens_seen:520000 train_supervised_tokens_seen:500000 ignored_target_tokens_seen:20000 supervised_target_fraction:0.961538",
                "final_prequant_exact val_loss:0.90000000 val_bpb:1.10000000 train_tokens_seen:520000 train_supervised_tokens_seen:500000 ignored_target_tokens_seen:20000 supervised_target_fraction:0.961538 eval_time:111ms",
                "Serialized model int8+zlib: 123 bytes (payload:45 raw_torch:67 payload_ratio:8.9x)",
                "Total submission size int8+zlib: 1000 bytes",
                "final_int8_zlib_roundtrip_exact val_loss:1.20000000 val_bpb:1.30000000",
                "quantization_delta_exact val_loss:0.30000000 val_bpb:0.20000000 train_tokens_seen:520000 train_supervised_tokens_seen:500000 ignored_target_tokens_seen:20000 supervised_target_fraction:0.961538",
            ]
        )
    )

    assert parsed["config"]["doc_aligned_batching"] is True
    assert parsed["config"]["ema_export"] is True
    assert parsed["config"]["ema_decay"] == pytest.approx(0.9999)
    assert parsed["config"]["lr_schedule"] == "wsd"
    assert parsed["config"]["wsd_warmup_frac"] == pytest.approx(0.01)
    assert parsed["config"]["wsd_stable_frac"] == pytest.approx(0.75)
    assert parsed["config"]["wsd_decay_style"] == "cosine"
    assert parsed["config"]["enable_cooldown_beta2"] is True
    assert parsed["config"]["cooldown_beta2"] == pytest.approx(0.98)
    assert parsed["train_steps"][0]["train_supervised_tokens_seen"] == 500000
    assert parsed["train_steps"][0]["ignored_target_tokens_seen"] == 20000
    assert parsed["final"]["train_supervised_tokens_seen"] == 500000
    assert parsed["final"]["ignored_target_tokens_seen"] == 20000
    assert parsed["final"]["supervised_target_fraction"] == pytest.approx(0.961538)


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
    assert "Δpq" in output
    assert "qgap" in output
    assert "1.2300*" in output


def test_compare_diag_finds_first_nested_difference() -> None:
    left = "\n".join(
        [
            'DIAG:train_reloaded_block0:{"emb":"1|2","attn_norm":"3|4","rotary_cache_before":{"seq_len_cached":0},"enc0":"5|6"}',
            'DIAG:train_reloaded_block0_cache_prewarmed:{"enc0":"7|8"}',
        ]
    )
    right = "\n".join(
        [
            'DIAG:train_reloaded_block0:{"emb":"1|2","attn_norm":"3|4","rotary_cache_before":{"seq_len_cached":16},"enc0":"5|6"}',
            'DIAG:train_reloaded_block0_cache_prewarmed:{"enc0":"7|8"}',
        ]
    )

    diff = compare_diag.compare_diag_text(left, right)

    assert diff["first_diff_path"] == "train_reloaded_block0.rotary_cache_before.seq_len_cached"
    assert diff["left_value"] == 0
    assert diff["right_value"] == 16


def test_compare_diag_reports_missing_label() -> None:
    left = 'DIAG:train_reloaded_block0_cache_cleared:{"enc0":"1|2"}'
    right = ""

    diff = compare_diag.compare_diag_text(left, right)

    assert diff["missing_label"] == "train_reloaded_block0_cache_cleared"
    assert diff["first_diff_path"] == "train_reloaded_block0_cache_cleared"


def test_compare_includes_replay_trust_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    metrics_path = tmp_path / "metrics.json"
    write_metrics(
        metrics_path,
        run_id="solo",
        val_bpb=1.23,
        val_loss=2.34,
        uncompiled_check_delta_bpb=0.25,
        reloaded_postquant_val_bpb=1.33,
    )
    monkeypatch.setattr(sys, "argv", ["compare.py", str(metrics_path)])
    compare.main()
    output = capsys.readouterr().out
    assert "EgrΔ" in output
    assert "RldΔ" in output
    assert "+0.2500" in output
    assert "+0.1000" in output


def test_compare_includes_supervised_fraction_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    metrics_path = tmp_path / "metrics.json"
    write_metrics(metrics_path, run_id="solo", val_bpb=1.23, val_loss=2.34, supervised_target_fraction=0.961538)
    monkeypatch.setattr(sys, "argv", ["compare.py", str(metrics_path)])
    compare.main()
    output = capsys.readouterr().out
    assert "Sup%" in output
    assert "96.2%" in output


def test_compare_sorts_by_bpb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    worse = tmp_path / "worse.json"
    better = tmp_path / "better.json"
    write_metrics(worse, run_id="worse", val_bpb=1.30, val_loss=2.40)
    write_metrics(better, run_id="better", val_bpb=1.20, val_loss=2.10)
    monkeypatch.setattr(sys, "argv", ["compare.py", str(worse), str(better)])
    compare.main()
    output = capsys.readouterr().out
    lines = output.splitlines()
    data_lines = [line for line in lines if " | " in line and not line.startswith("Run ID")]
    assert data_lines[0].startswith("better")
    assert data_lines[1].startswith("worse")
    assert "+0.1000" in output or "-0.1000" in output


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


def test_build_manifest_includes_metadata_and_wandb_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launch_runtime,
        "resolve_machine",
        lambda host: ("alpha", {"host": "root@example", "remote_dir": "/remote/shared", "gpus": 8, "hourly_rate": 12.0, "shutdown_after_idle_minutes": 7}, []),
    )
    monkeypatch.setattr(launch_runtime, "git_sha", lambda: "abc123")
    monkeypatch.setattr(launch_runtime, "sha256_file", lambda path: "deadbeef")
    manifest = launch_runtime.build_manifest(
        BASELINE_CONFIG,
        {"WANDB_PROJECT": "pgolf"},
        "meta-run",
        "alpha",
        8,
        None,
        metadata={"hypothesis_id": "h1", "group": "g1", "notes": "hello"},
        shutdown_enabled=True,
    )
    assert manifest["hypothesis_id"] == "h1"
    assert manifest["group"] == "g1"
    assert manifest["notes"] == "hello"
    assert manifest["shutdown_after_idle_minutes"] == 7
    assert manifest["_forwarded_env"]["PGOLF_WANDB_GROUP"] == "g1"
    assert manifest["_forwarded_env"]["PGOLF_WANDB_NOTES"] == "hello"
    assert manifest["_forwarded_env"]["PGOLF_WANDB_TAGS"] == "hypothesis:h1"
    assert manifest["_forwarded_env"]["DATA_PATH"] == "/remote/shared/data/datasets/fineweb10B_sp1024"
    assert manifest["_forwarded_env"]["TOKENIZER_PATH"] == "/remote/shared/data/tokenizers/fineweb_1024_bpe.model"
    assert manifest["_forwarded_env"]["DATASET_VARIANT"] == "sp1024"


def test_build_manifest_preserves_explicit_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launch_runtime,
        "resolve_machine",
        lambda host: ("alpha", {"host": "root@example", "remote_dir": "/remote/shared", "gpus": 8, "hourly_rate": 12.0}, []),
    )
    monkeypatch.setattr(launch_runtime, "git_sha", lambda: "abc123")
    monkeypatch.setattr(launch_runtime, "sha256_file", lambda path: "deadbeef")
    manifest = launch_runtime.build_manifest(
        BASELINE_CONFIG,
        {
            "DATA_PATH": "/custom/data/datasets/fineweb10B_sp768",
            "TOKENIZER_PATH": "/custom/data/tokenizers/fineweb_768_bpe.model",
        },
        "custom-paths",
        "alpha",
        8,
        None,
    )
    assert manifest["_forwarded_env"]["DATA_PATH"] == "/custom/data/datasets/fineweb10B_sp768"
    assert manifest["_forwarded_env"]["TOKENIZER_PATH"] == "/custom/data/tokenizers/fineweb_768_bpe.model"
    assert "DATASET_VARIANT" not in manifest["_forwarded_env"]


def test_build_manifest_defaults_to_machine_gpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launch_runtime,
        "resolve_machine",
        lambda host: ("alpha", {"host": "root@example", "remote_dir": "/remote/shared", "gpus": 1, "hourly_rate": 12.0}, []),
    )
    monkeypatch.setattr(launch_runtime, "git_sha", lambda: "abc123")
    monkeypatch.setattr(launch_runtime, "sha256_file", lambda path: "deadbeef")
    manifest = launch_runtime.build_manifest(
        BASELINE_CONFIG,
        {},
        "machine-gpus",
        "alpha",
        None,
        None,
    )
    assert manifest["gpus"] == 1


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


def test_schedule_shutdown_records_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launch_runtime, "RESULTS_DIR", tmp_path / "results")
    commands: list[str] = []

    def fake_ssh(host: str, command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(["ssh", host, command], 0, "", "")

    monkeypatch.setattr(launch_runtime, "ssh", fake_ssh)
    manifest = {"run_id": "shutdown-me", "host": "host-1", "shutdown_after_idle_minutes": 5, "start_time_epoch": 0.0, "hourly_rate": None}
    launch_runtime.schedule_shutdown([manifest])
    saved = launch_runtime.load_manifest("shutdown-me")
    assert commands == ["sudo shutdown -h +5"]
    assert saved is not None
    assert saved["shutdown_command"] == "sudo shutdown -h +5"


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
        if "RUNNING" in command or "EXIT:" in command:
            stdout = {
                "host-1": "run-a\tRUNNING\nrun-b\tEXIT:0\n",
                "host-2": "run-c\tUNKNOWN\n",
            }[host]
        else:
            stdout = {
                "host-1": "run-a\t10\tstep:1/10 train_loss:1.0 train_time:100ms step_avg:100ms\nrun-b\t12\tstep:10/10 val_loss:2.0 val_bpb:1.2 train_time:100ms step_avg:100ms\n",
                "host-2": "run-c\t13\tCUDA out of memory\n",
            }[host]
        return subprocess.CompletedProcess(["ssh", host, command], 0, stdout, "")

    monkeypatch.setattr(launch_runtime, "ssh", fake_ssh)
    assert launch.cmd_status(type("Args", (), {"host": None, "watch": False, "running": False, "failed": False})()) == 0
    capsys.readouterr()
    assert sorted(set(host for host, _ in ssh_calls)) == ["host-1", "host-2"]
    assert sum(host == "host-1" for host, _ in ssh_calls) == 2


def test_render_status_includes_rich_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    run_dir = results_dir / "run-1"
    run_dir.mkdir(parents=True)
    write_metrics(
        run_dir / "metrics.json",
        run_id="run-1",
        val_bpb=1.22,
        val_loss=2.07,
        artifact_slack_bytes=9_350_400,
        reloaded_postquant_val_bpb=1.32,
    )
    monkeypatch.setattr(launch, "RESULTS_DIR", results_dir)
    output = launch.render_status(
        [
            {
                "run_id": "run-1",
                "group": "grp",
                "host_name": "alpha",
                "status": "failed",
                "start_time_epoch": time.time() - 30,
                "end_time_epoch": time.time(),
                "estimated_cost": 1.23,
                "hypothesis_id": "h1",
                "failure_reason": "cuda_oom",
                "last_log_line": "step:10/20 train_loss:1.2340 train_time:100ms step_avg:100ms",
                "last_log_update_epoch": time.time() - 5,
            }
        ]
    )
    assert "Failure" in output
    assert "ReplayΔ" in output
    assert "qgap" in output
    assert "Tok/s" in output
    assert "cuda_oom" in output
    assert "grp" in output


def test_ensure_remote_data_uses_flock(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[str] = []

    def fake_ssh(host: str, command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        code = 1 if command.startswith("test -f ") else 0
        return subprocess.CompletedProcess(["ssh", host, command], code, "", "")

    monkeypatch.setattr(launch_runtime, "ssh", fake_ssh)
    manifest = {
        "host": "example",
        "resolved_env": {
            "DATA_PATH": "/shared/data/datasets/fineweb10B_sp1024",
            "TOKENIZER_PATH": "/shared/data/tokenizers/fineweb_1024_bpe.model",
            "DATASET_VARIANT": "sp1024",
        },
        "machine_remote_dir": "/shared",
    }
    launch_runtime.ensure_remote_data(manifest)
    assert commands[0].startswith("test -f ")
    assert "flock -x /tmp/pgolf_data.lock" in commands[1]
    assert config_utils.remote_python_command("/shared", "data/cached_challenge_fineweb.py --variant sp1024") in commands[1]


def test_ensure_remote_data_rejects_custom_missing_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launch_runtime,
        "ssh",
        lambda host, command, *, capture=False, check=True: subprocess.CompletedProcess(["ssh", host, command], 1, "", ""),
    )
    manifest = {
        "host": "example",
        "resolved_env": {
            "DATA_PATH": "/custom/datasets/experimental",
            "TOKENIZER_PATH": "/custom/tokenizers/experimental.model",
        },
        "machine_remote_dir": "/remote/shared",
    }
    with pytest.raises(RuntimeError, match="no DATASET_VARIANT was provided"):
        launch_runtime.ensure_remote_data(manifest)


def test_verify_remote_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[str] = []

    def fake_ssh(host: str, command: str, *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(["ssh", host, command], 0, "", "")

    monkeypatch.setattr(launch_runtime, "ssh", fake_ssh)
    launch_runtime.verify_remote_dependencies({"host": "example", "remote_repo_dir": "/remote/repo", "machine_remote_dir": "/shared"})
    assert config_utils.remote_python_command("/shared", "-c 'import torch; import sentencepiece; import numpy'") in commands[0]


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
        "machine_remote_dir": "/shared",
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
    assert config_utils.remote_python_command("/shared", "-m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py") in command
    assert manifest["pid"] == 4321


def test_watchdog_detects_invalid_loss() -> None:
    outcome = watchdog.evaluate_snapshot(
        {
            "lines": [
                "step:99/200 train_loss:1.0 train_time:100ms step_avg:100ms",
                "step:100/200 train_loss:nan train_time:100ms step_avg:100ms",
            ],
            "mtime_epoch": time.time(),
        },
        now_epoch=time.time(),
    )
    assert outcome["triggered"] is True
    assert outcome["failure_reason"] == "invalid_train_loss"


def test_watchdog_detects_stall_and_regression() -> None:
    now_epoch = time.time()
    stalled = watchdog.evaluate_snapshot(
        {
            "lines": ["step:150/200 train_loss:1.0 train_time:100ms step_avg:1000ms"],
            "mtime_epoch": now_epoch - 70,
        },
        now_epoch=now_epoch,
        train_log_every=1,
    )
    regressing = watchdog.evaluate_snapshot(
        {
            "lines": [
                "step:50/200 val_loss:2.0 val_bpb:1.10 train_time:100ms step_avg:100ms",
                "step:100/200 val_loss:2.0 val_bpb:1.11 train_time:100ms step_avg:100ms",
                "step:150/200 val_loss:2.0 val_bpb:1.12 train_time:100ms step_avg:100ms",
                "step:200/200 val_loss:2.0 val_bpb:1.13 train_time:100ms step_avg:100ms",
            ],
            "mtime_epoch": now_epoch,
        },
        now_epoch=now_epoch,
    )
    assert stalled["failure_reason"] == "stalled"
    assert regressing["failure_reason"] == "regressing_val_bpb"


def test_watchdog_calibration_policy_skips_val_regression_kill() -> None:
    now_epoch = time.time()
    outcome = watchdog.evaluate_snapshot(
        {
            "lines": [
                "step:50/200 val_loss:2.0 val_bpb:1.10 train_time:100ms step_avg:100ms",
                "step:100/200 val_loss:2.0 val_bpb:1.11 train_time:100ms step_avg:100ms",
                "step:150/200 val_loss:2.0 val_bpb:1.12 train_time:100ms step_avg:100ms",
                "step:200/200 val_loss:2.0 val_bpb:1.13 train_time:100ms step_avg:100ms",
            ],
            "mtime_epoch": now_epoch,
        },
        now_epoch=now_epoch,
        watchdog_policy="calibration",
    )
    assert outcome["triggered"] is False
    assert outcome["watchdog_policy"] == "calibration"


def test_watchdog_respects_train_log_cadence() -> None:
    now_epoch = time.time()
    outcome = watchdog.evaluate_snapshot(
        {
            "lines": ["step:110/20000 train_loss:4.4018 train_time:36416ms step_avg:331.06ms"],
            "mtime_epoch": now_epoch - 5,
        },
        now_epoch=now_epoch,
        train_log_every=10,
    )
    assert outcome["triggered"] is False


def test_watchdog_check_kills_run(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[dict[str, object]] = []
    terminated: list[str] = []
    monkeypatch.setattr(watchdog, "remote_log_snapshot", lambda manifest, lines=20: {"lines": ["CUDA out of memory"], "mtime_epoch": time.time(), "last_line": "CUDA out of memory"})
    monkeypatch.setattr(watchdog, "save_manifest", lambda manifest: saved.append(dict(manifest)))
    monkeypatch.setattr(watchdog, "terminate_remote_run", lambda manifest, failure_reason=None: terminated.append(str(failure_reason)))
    manifest = {"run_id": "watch-1", "status": "running"}
    outcome = watchdog.check_watchdog(manifest)
    assert outcome["triggered"] is True
    assert terminated == ["cuda_oom"]
    assert saved[-1]["failure_reason"] == "cuda_oom"


def test_watchdog_check_records_calibration_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[dict[str, object]] = []
    monkeypatch.setattr(
        watchdog,
        "remote_log_snapshot",
        lambda manifest, lines=20: {
            "lines": [
                "step:50/200 val_loss:2.0 val_bpb:1.10 train_time:100ms step_avg:100ms",
                "step:100/200 val_loss:2.0 val_bpb:1.11 train_time:100ms step_avg:100ms",
                "step:150/200 val_loss:2.0 val_bpb:1.12 train_time:100ms step_avg:100ms",
                "step:200/200 val_loss:2.0 val_bpb:1.13 train_time:100ms step_avg:100ms",
            ],
            "mtime_epoch": time.time(),
            "last_line": "step:200/200 val_loss:2.0 val_bpb:1.13 train_time:100ms step_avg:100ms",
        },
    )
    monkeypatch.setattr(watchdog, "save_manifest", lambda manifest: saved.append(dict(manifest)))
    monkeypatch.setattr(
        watchdog,
        "terminate_remote_run",
        lambda manifest, failure_reason=None: (_ for _ in ()).throw(AssertionError("calibration mode should not terminate")),
    )
    manifest = {
        "run_id": "watch-2",
        "status": "running",
        "resolved_env": {"WATCHDOG_POLICY": "calibration", "TRAIN_LOG_EVERY": "25"},
    }
    outcome = watchdog.check_watchdog(manifest)
    assert outcome["triggered"] is False
    assert saved[-1]["watchdog_policy"] == "calibration"


def test_preflight_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "resolve_machine", lambda host: ("alpha", {"host": "root@example", "remote_dir": "/shared", "gpus": 8}, []))
    python_cmd = config_utils.remote_python_command("/shared", "-c 'import torch; print(torch.cuda.device_count())'")
    deps_cmd = config_utils.remote_python_command("/shared", "-c 'import sentencepiece, numpy; print(\"ok\")'")
    disk_cmd = config_utils.remote_python_command("/shared", "-c 'import shutil; print(int(shutil.disk_usage(\"/tmp\").free / (1024**3)))'")

    def fake_ssh(host: str, command: str) -> subprocess.CompletedProcess[str]:
        mapping = {
            "echo ok": "ok\n",
            "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l": "8\n",
            python_cmd: "8\n",
            deps_cmd: "ok\n",
            disk_cmd: "200\n",
        }
        if "PGOLF_PYTHON_BIN" in command and preflight.COMPILE_TOOLCHAIN_CHECK in command:
            return subprocess.CompletedProcess(["ssh", host, command], 0, "ok\n", "")
        if command.startswith("test -d "):
            return subprocess.CompletedProcess(["ssh", host, command], 0, "", "")
        if command.startswith("mkdir -p /tmp/pgolf_preflight_test"):
            return subprocess.CompletedProcess(["ssh", host, command], 0, "", "")
        return subprocess.CompletedProcess(["ssh", host, command], 0, mapping[command], "")

    monkeypatch.setattr(preflight, "ssh_capture", fake_ssh)
    payload = preflight.run_preflight_target("alpha", print_output=False)
    assert payload["ok"] is True
    assert all(result["ok"] for result in payload["results"])


def test_preflight_uses_config_selected_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = write_yaml(
        tmp_path / "preflight_paths.yaml",
        """
        name: preflight_paths
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
          DATA_PATH: "/shared/data/datasets/fineweb10B_sp768"
          TOKENIZER_PATH: "/shared/data/tokenizers/fineweb_768_bpe.model"
        """,
    )
    monkeypatch.setattr(preflight, "resolve_machine", lambda host: ("alpha", {"host": "root@example", "remote_dir": "/shared", "gpus": 8}, []))
    python_cmd = config_utils.remote_python_command("/shared", "-c 'import torch; print(torch.cuda.device_count())'")
    deps_cmd = config_utils.remote_python_command("/shared", "-c 'import sentencepiece, numpy; print(\"ok\")'")
    disk_cmd = config_utils.remote_python_command("/shared", "-c 'import shutil; print(int(shutil.disk_usage(\"/tmp\").free / (1024**3)))'")

    def fake_ssh(host: str, command: str) -> subprocess.CompletedProcess[str]:
        mapping = {
            "echo ok": "ok\n",
            "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l": "8\n",
            python_cmd: "8\n",
            deps_cmd: "ok\n",
            disk_cmd: "200\n",
        }
        if "PGOLF_PYTHON_BIN" in command and preflight.COMPILE_TOOLCHAIN_CHECK in command:
            return subprocess.CompletedProcess(["ssh", host, command], 0, "ok\n", "")
        if command == "test -d '/shared/data/datasets/fineweb10B_sp768' && test -f '/shared/data/tokenizers/fineweb_768_bpe.model'":
            return subprocess.CompletedProcess(["ssh", host, command], 0, "", "")
        if command.startswith("mkdir -p /tmp/pgolf_preflight_test"):
            return subprocess.CompletedProcess(["ssh", host, command], 0, "", "")
        return subprocess.CompletedProcess(["ssh", host, command], 0, mapping[command], "")

    monkeypatch.setattr(preflight, "ssh_capture", fake_ssh)
    config_env, warnings, errors = preflight.load_preflight_env(str(config_path))
    assert errors == []
    payload = preflight.run_preflight_target("alpha", config_env=config_env, warnings=warnings, print_output=False)
    assert payload["ok"] is True


def test_cost_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(cost, "RESULTS_DIR", results_dir)
    now = time.time()
    for run_id, status, amount in [("a", "success", 1.5), ("b", "running", 2.0)]:
        run_dir = results_dir / run_id
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": status,
                    "estimated_cost": amount,
                    "host_name": "alpha",
                    "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                }
            ),
            encoding="utf-8",
        )
    output = cost.summarize_costs(cost.load_manifests(), budget=10.0)
    assert "Total spend" in output
    assert "$3.50" in output
    assert "Remaining budget" in output


def test_sweep_queue_create_and_requeue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep_queue, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(sweep_queue, "SWEEPS_DIR", tmp_path / "results" / "_sweeps")
    result = config_utils.validate_config(SWEEP_CONFIG)
    queue = sweep_queue.create_queue(str(SWEEP_CONFIG), result.runs[:2])
    sweep_id = queue["sweep_id"]
    claimed = sweep_queue.claim_next_run(sweep_id, "alpha")
    assert claimed is not None
    assert claimed["status"] == "running"
    payload = sweep_queue.load_queue(sweep_id)
    payload["runs"][0]["status"] = "failed"
    sweep_queue.save_queue(payload)
    assert sweep_queue.requeue_runs(sweep_id, failed=True, lost=False) == 1
    reloaded = sweep_queue.load_queue(sweep_id)
    assert reloaded["runs"][0]["status"] == "pending"


def test_gc_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pgolf_gc,
        "list_remote_dirs",
        lambda host: (
            "root@example",
            [
                {"path": "/tmp/pgolf_old-run", "age_seconds": 30 * 3600, "mtime": time.time() - 30 * 3600},
                {"path": "/tmp/pgolf_fresh-run", "age_seconds": 2 * 3600, "mtime": time.time() - 2 * 3600},
            ],
        ),
    )
    monkeypatch.setattr(pgolf_gc, "is_collected", lambda run_id: run_id == "old-run")
    payload = pgolf_gc.run_gc("alpha", older_than="24h", dry_run=True)
    assert len(payload["deletable"]) == 1
    assert payload["deletable"][0]["run_id"] == "old-run"


def test_sitecustomize_adds_wandb_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_calls: list[dict[str, object]] = []
    fake_wandb = types.SimpleNamespace(init=lambda *args, **kwargs: fake_calls.append(kwargs) or kwargs)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("PGOLF_WANDB_GROUP", "group-a")
    monkeypatch.setenv("PGOLF_WANDB_NOTES", "note-a")
    monkeypatch.setenv("PGOLF_WANDB_TAGS", "hypothesis:h1,tag2")
    spec = importlib.util.spec_from_file_location("pgolf_sitecustomize", SCRIPTS_DIR / "sitecustomize.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake_wandb.init(project="demo")
    assert fake_calls[-1]["group"] == "group-a"
    assert fake_calls[-1]["notes"] == "note-a"
    assert fake_calls[-1]["tags"] == ["hypothesis:h1", "tag2"]
