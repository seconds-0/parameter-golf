#!/usr/bin/env python3
"""Run an exact exporter-only roundtrip evaluation from a saved final_model.pt checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import random
import socket
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config_utils import CONFIG_METADATA_FIELDS, REPO_DIR, extract_allowlist, resolve_path, validate_config, with_default_paths


@dataclass(frozen=True)
class ReplayInputs:
    env: dict[str, str]
    metadata: dict[str, str]
    trainer_path: Path
    env_source: str


def default_artifact_out(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.export_eval.int8.ptz")


def default_metrics_out(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.export_eval.json")


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Invalid --set-env override {value!r}; expected KEY=VALUE")
        overrides[key.strip()] = raw
    return overrides


def resolve_config_env(config_path_arg: str) -> tuple[dict[str, str], dict[str, str]]:
    config_path = resolve_path(config_path_arg)
    result = validate_config(config_path)
    if result.errors:
        raise RuntimeError("; ".join(result.errors))
    if len(result.runs) != 1:
        raise RuntimeError(f"Config {config_path} expands to {len(result.runs)} runs; export_eval requires exactly 1 run")
    run = result.runs[0]
    return with_default_paths(run.env, REPO_DIR), run.metadata


def resolve_replay_inputs(checkpoint_path: Path, config_path_arg: str | None) -> ReplayInputs:
    trainer_path = checkpoint_path.parent / "train_gpt.py"
    if not trainer_path.exists():
        trainer_path = REPO_DIR / "train_gpt.py"

    if config_path_arg:
        env, metadata = resolve_config_env(config_path_arg)
        return ReplayInputs(env=env, metadata=metadata, trainer_path=trainer_path, env_source="config")

    manifest_path = checkpoint_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_env = manifest.get("resolved_env")
        if isinstance(raw_env, dict):
            metadata = {
                key: value
                for key in CONFIG_METADATA_FIELDS
                if isinstance((value := manifest.get(key)), str) and value.strip()
            }
            return ReplayInputs(
                env={str(key): str(value) for key, value in raw_env.items()},
                metadata=metadata,
                trainer_path=trainer_path,
                env_source="manifest",
            )

    raise RuntimeError("export_eval requires --config unless checkpoint sits next to a manifest.json with resolved_env")


def load_trainer_with_env(trainer_path: Path, env: dict[str, str]) -> Any:
    allowlist = extract_allowlist()
    original_env = {key: os.environ.get(key) for key in allowlist}
    for key in allowlist:
        os.environ.pop(key, None)
    for key, value in env.items():
        os.environ[key] = value

    spec = importlib.util.spec_from_file_location(f"pgolf_export_eval_{trainer_path.stem}", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer module from {trainer_path}")
    trainer = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(trainer)
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return trainer


def maybe_call(trainer: Any, name: str, *args: Any) -> None:
    fn = getattr(trainer, name, None)
    if callable(fn):
        fn(*args)


def compat_raw_checkpoint_state_dict(base_model: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
    patched = dict(state_dict)
    model_state = base_model.state_dict()
    for key, value in model_state.items():
        if key.endswith("rotary.inv_freq") and key not in patched:
            patched[key] = value
    return patched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate exporter settings from a saved final_model.pt checkpoint")
    parser.add_argument("checkpoint", help="Path to final_model.pt")
    parser.add_argument(
        "--config",
        help="Single-run config used to construct the model/eval setup; optional when checkpoint sits next to manifest.json",
    )
    parser.add_argument("--artifact-out", help="Where to write the int8+zlib artifact (default: <checkpoint>.int8.ptz)")
    parser.add_argument("--metrics-out", help="Where to write metrics JSON (default: <checkpoint>.export_eval.json)")
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override exporter env vars such as INT8_CLIP_PERCENTILE or INT8_KEEP_FLOAT_MAX_NUMEL",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    try:
        export_overrides = parse_env_overrides(args.set_env)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        replay = resolve_replay_inputs(checkpoint_path, args.config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    trainer = load_trainer_with_env(replay.trainer_path, {**replay.env, **export_overrides})

    if not trainer.torch.cuda.is_available():
        raise SystemExit("CUDA is required for export_eval.py")

    trainer.torch.backends.cuda.matmul.allow_tf32 = True
    trainer.torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    runtime_args = trainer.Hyperparameters()
    maybe_call(trainer, "hyperparams_fingerprint", runtime_args, "replay_hparams")
    random.seed(runtime_args.seed)
    np.random.seed(runtime_args.seed)
    trainer.torch.manual_seed(runtime_args.seed)
    trainer.torch.cuda.manual_seed_all(runtime_args.seed)

    device = trainer.torch.device("cuda", 0)
    trainer.torch.cuda.set_device(device)
    trainer.torch.cuda.reset_peak_memory_stats(device)
    grad_accum_steps = 8
    world_size = 1
    rank = 0

    sp = trainer.spm.SentencePieceProcessor(model_file=runtime_args.tokenizer_path)
    if int(sp.vocab_size()) != runtime_args.vocab_size:
        raise SystemExit(
            f"VOCAB_SIZE={runtime_args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    val_tokens = trainer.load_validation_tokens(runtime_args.val_files, runtime_args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = trainer.build_sentencepiece_luts(
        sp, runtime_args.vocab_size, device
    )

    base_model = trainer.GPT(
        vocab_size=runtime_args.vocab_size,
        num_layers=runtime_args.num_layers,
        model_dim=runtime_args.model_dim,
        num_heads=runtime_args.num_heads,
        num_kv_heads=runtime_args.num_kv_heads,
        mlp_mult=runtime_args.mlp_mult,
        tie_embeddings=runtime_args.tie_embeddings,
        tied_embed_init_std=runtime_args.tied_embed_init_std,
        logit_softcap=runtime_args.logit_softcap,
        logit_softcap_pos=runtime_args.logit_softcap_pos,
        logit_softcap_neg=runtime_args.logit_softcap_neg,
        rope_base=runtime_args.rope_base,
        qk_gain_init=runtime_args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, trainer.CastedLinear):
            module.float()
    trainer.restore_low_dim_params_to_fp32(base_model)

    state_dict = trainer.torch.load(checkpoint_path, map_location="cpu")
    state_dict = compat_raw_checkpoint_state_dict(base_model, state_dict)
    base_model.load_state_dict(state_dict, strict=True)
    maybe_call(trainer, "model_fingerprint", base_model, "replay_loaded")
    maybe_call(
        trainer,
        "diagnostic_forward",
        base_model,
        runtime_args.train_seq_len,
        runtime_args.vocab_size,
        device,
        "replay_diag_fwd",
    )
    maybe_call(
        trainer,
        "diagnostic_block0_forward",
        base_model,
        runtime_args.train_seq_len,
        runtime_args.vocab_size,
        device,
        "replay_loaded_block0",
    )
    maybe_call(trainer, "reset_rotary_caches", base_model)
    maybe_call(
        trainer,
        "diagnostic_block0_forward",
        base_model,
        runtime_args.train_seq_len,
        runtime_args.vocab_size,
        device,
        "replay_loaded_block0_cache_cleared",
    )
    maybe_call(trainer, "reset_rotary_caches", base_model)
    maybe_call(trainer, "prewarm_rotary_caches", base_model, runtime_args.train_seq_len, device, base_model.tok_emb.weight.dtype)
    maybe_call(
        trainer,
        "diagnostic_block0_forward",
        base_model,
        runtime_args.train_seq_len,
        runtime_args.vocab_size,
        device,
        "replay_loaded_block0_cache_prewarmed",
    )

    pre_val_loss, pre_val_bpb = trainer.eval_val(
        runtime_args,
        base_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )

    quant_obj, quant_stats = trainer.quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    trainer.torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    artifact_path = resolve_path(args.artifact_out) if args.artifact_out else default_artifact_out(checkpoint_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(quant_blob)

    quant_state = trainer.torch.load(io.BytesIO(zlib.decompress(artifact_path.read_bytes())), map_location="cpu")
    base_model.load_state_dict(trainer.dequantize_state_dict_int8(quant_state), strict=True)
    post_val_loss, post_val_bpb = trainer.eval_val(
        runtime_args,
        base_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )

    code_bytes = len(Path(trainer.__file__).read_text(encoding="utf-8").encode("utf-8"))
    quant_file_bytes = artifact_path.stat().st_size
    total_submission_bytes = quant_file_bytes + code_bytes
    qgap_loss = post_val_loss - pre_val_loss
    qgap_bpb = post_val_bpb - pre_val_bpb

    metrics = {
        "run_id": checkpoint_path.stem,
        "status": "success",
        "host": socket.gethostname(),
        "group": replay.metadata.get("group"),
        "hypothesis_id": replay.metadata.get("hypothesis_id"),
        "notes": replay.metadata.get("notes"),
        "source_checkpoint": str(checkpoint_path),
        "env_source": replay.env_source,
        "trainer_path": str(replay.trainer_path),
        "config": {
            "world_size": world_size,
            "grad_accum_steps": grad_accum_steps,
            "model_params": sum(param.numel() for param in base_model.parameters()),
            "train_batch_tokens": runtime_args.train_batch_tokens,
            "train_seq_len": runtime_args.train_seq_len,
            "max_wallclock_seconds": runtime_args.max_wallclock_seconds,
            "seed": runtime_args.seed,
            "int8_clip_percentile": float(trainer.INT8_CLIP_PERCENTILE),
            "int8_keep_float_max_numel": int(trainer.INT8_KEEP_FLOAT_MAX_NUMEL),
            "control_tensor_name_patterns": list(trainer.CONTROL_TENSOR_NAME_PATTERNS),
            "int8_keep_float_fp32_name_patterns": list(trainer.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS),
        },
        "train_steps": [],
        "val_steps": [],
        "final": {
            "prequant_val_loss": pre_val_loss,
            "prequant_val_bpb": pre_val_bpb,
            "postquant_val_loss": post_val_loss,
            "postquant_val_bpb": post_val_bpb,
            "val_loss": post_val_loss,
            "val_bpb": post_val_bpb,
            "qgap_loss": qgap_loss,
            "qgap_bpb": qgap_bpb,
            "model_bytes_raw": checkpoint_path.stat().st_size,
            "model_bytes_int8_zlib": quant_file_bytes,
            "payload_bytes": quant_stats["int8_payload_bytes"],
            "payload_raw_torch_bytes": len(quant_raw),
            "payload_ratio": quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1),
            "code_bytes": code_bytes,
            "total_submission_bytes": total_submission_bytes,
            "artifact_slack_bytes": 16_000_000 - total_submission_bytes,
            "peak_memory_allocated_mib": trainer.torch.cuda.max_memory_allocated() // 1024 // 1024,
            "peak_memory_reserved_mib": trainer.torch.cuda.max_memory_reserved() // 1024 // 1024,
        },
    }

    metrics_path = resolve_path(args.metrics_out) if args.metrics_out else default_metrics_out(checkpoint_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {metrics_path}")
    print(
        f"postquant_val_bpb={post_val_bpb:.8f} prequant_val_bpb={pre_val_bpb:.8f} "
        f"qgap_bpb={qgap_bpb:.8f} total_submission_bytes={total_submission_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
