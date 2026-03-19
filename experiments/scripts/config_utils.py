#!/usr/bin/env python3
"""Shared config helpers for Parameter Golf experiment scripts."""

from __future__ import annotations

import hashlib
import itertools
import random
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_DIR / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
SCRIPTS_DIR = EXPERIMENTS_DIR / "scripts"
MACHINES_FILE = EXPERIMENTS_DIR / "machines.yaml"
TRAINER_PATH = REPO_DIR / "train_gpt.py"
TRAINER_MLX_PATH = REPO_DIR / "train_gpt_mlx.py"
ENV_NAME_RE = re.compile(r'os\.environ\.get\("([A-Z0-9_]+)"')
POSITIVE_INT_KEYS = {
    "ITERATIONS", "MLP_MULT", "MODEL_DIM", "MUON_BACKEND_STEPS", "NUM_HEADS",
    "NUM_KV_HEADS", "NUM_LAYERS", "TRAIN_BATCH_TOKENS", "TRAIN_SEQ_LEN",
    "VAL_BATCH_SIZE", "VOCAB_SIZE",
}
POSITIVE_FLOAT_KEYS = {
    "ADAM_EPS", "BETA1", "BETA2", "EMBED_LR", "HEAD_LR", "LOGIT_SOFTCAP",
    "MATRIX_LR", "MAX_WALLCLOCK_SECONDS", "MUON_MOMENTUM",
    "MUON_MOMENTUM_WARMUP_START", "QK_GAIN_INIT", "ROPE_BASE", "SCALAR_LR",
    "TIED_EMBED_INIT_STD", "TIED_EMBED_LR",
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    env: dict[str, str]
    combo: dict[str, str]


@dataclass(frozen=True)
class ValidationResult:
    config: dict[str, Any]
    runs: list[RunSpec]
    warnings: list[str]
    errors: list[str]
    allowlist: set[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_path(path_str: str, base_dir: Path | None = None) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    candidate = (CONFIGS_DIR / path).resolve()
    if candidate.exists():
        return candidate
    return (Path.cwd() / path).resolve()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def slugify(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return clean.strip("-") or "run"


def unique_run_id(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slugify(label)}-{stamp}-{uuid.uuid4().hex[:8]}"


def load_yaml(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}, [f"Config not found: {path}"]
    except yaml.YAMLError as exc:
        return {}, [f"Invalid YAML in {path}: {exc}"]
    if not isinstance(data, dict):
        return {}, [f"Top-level YAML object must be a mapping in {path}"]
    return data, []


def merge_dicts(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, value in child.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    cfg, errors = load_yaml(config_path)
    warnings: list[str] = []
    if errors:
        return {}, warnings, errors
    parent_ref = cfg.get("extends")
    if not parent_ref:
        return cfg, warnings, []
    parent_path = resolve_path(str(parent_ref), config_path.parent)
    parent_cfg, parent_errors = load_yaml(parent_path)
    if parent_errors:
        return {}, warnings, parent_errors
    if "extends" in parent_cfg:
        return {}, warnings, [f"Config extends chain is not supported: {parent_path} also has extends"]
    merged = merge_dicts(parent_cfg, {k: v for k, v in cfg.items() if k != "extends"})
    return merged, warnings, []


def extract_allowlist() -> set[str]:
    names: set[str] = set()
    for trainer_path in (TRAINER_PATH, TRAINER_MLX_PATH):
        if trainer_path.exists():
            names.update(ENV_NAME_RE.findall(trainer_path.read_text(encoding="utf-8")))
    return names


def stringify_env(env: Any) -> tuple[dict[str, str], list[str]]:
    if env is None:
        return {}, []
    if not isinstance(env, dict):
        return {}, [f"Environment block must be a mapping, got {type(env).__name__}"]
    return {str(k): str(v) for k, v in env.items()}, []


def _base_env(cfg: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    base_env, base_errors = stringify_env(cfg.get("base_env"))
    env, env_errors = stringify_env(cfg.get("env"))
    return {**base_env, **env}, [*base_errors, *env_errors]


def _combo_label(cfg: dict[str, Any], combo: dict[str, str]) -> str:
    template = cfg.get("naming")
    if isinstance(template, str):
        try:
            return template.format_map(combo)
        except KeyError as exc:
            return f"invalid-naming-missing-{exc.args[0]}"
    if combo:
        return "_".join(f"{key}{value}" for key, value in sorted(combo.items()))
    return str(cfg.get("name") or "run")


def expand_runs(cfg: dict[str, Any]) -> tuple[list[RunSpec], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    base_env, env_errors = _base_env(cfg)
    errors.extend(env_errors)
    if errors:
        return [], warnings, errors
    sweep = cfg.get("sweep")
    if not sweep:
        label = base_env.get("RUN_ID") or str(cfg.get("name") or "run")
        return [RunSpec(label=label, env=base_env, combo={})], warnings, []
    if not isinstance(sweep, dict):
        return [], warnings, ["sweep must be a mapping"]
    params = sweep.get("params")
    if not isinstance(params, dict) or not params:
        return [], warnings, ["sweep.params must be a non-empty mapping"]
    keys = sorted(str(key) for key in params)
    value_lists: list[list[str]] = []
    for key in keys:
        values = params[key]
        if not isinstance(values, list) or not values:
            errors.append(f"sweep.params.{key} must be a non-empty list")
            continue
        value_lists.append([str(value) for value in values])
    if errors:
        return [], warnings, errors
    combos = [dict(zip(keys, values, strict=True)) for values in itertools.product(*value_lists)]
    sweep_type = str(sweep.get("type", "grid"))
    if sweep_type == "random":
        count = sweep.get("count")
        if not isinstance(count, int) or count <= 0:
            errors.append("random sweep requires positive integer sweep.count")
            return [], warnings, errors
        rng = random.Random(sweep.get("seed", 0))
        selected = rng.sample(combos, count) if count <= len(combos) else rng.choices(combos, k=count)
        combos = selected
    elif sweep_type != "grid":
        errors.append(f"Unsupported sweep.type: {sweep_type}")
        return [], warnings, errors
    runs = []
    for combo in combos:
        env = {**base_env, **combo}
        runs.append(RunSpec(label=_combo_label(cfg, combo), env=env, combo=combo))
    return runs, warnings, []


def _parse_int(name: str, value: str, errors: list[str]) -> int | None:
    try:
        return int(value)
    except ValueError:
        errors.append(f"{name} must be an integer, got {value!r}")
        return None


def _parse_float(name: str, value: str, errors: list[str]) -> float | None:
    try:
        return float(value)
    except ValueError:
        errors.append(f"{name} must be a float, got {value!r}")
        return None


def validate_env(env: dict[str, str], allowlist: set[str]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    for key in sorted(env):
        if key not in allowlist:
            warnings.append(f"Unknown env var {key} (not found in train_gpt.py/train_gpt_mlx.py)")
    ints = {key: _parse_int(key, env[key], errors) for key in POSITIVE_INT_KEYS if key in env}
    floats = {key: _parse_float(key, env[key], errors) for key in POSITIVE_FLOAT_KEYS if key in env}
    for key, value in ints.items():
        if value is not None and value <= 0:
            errors.append(f"{key} must be > 0, got {value}")
    for key, value in floats.items():
        if value is not None and value <= 0:
            errors.append(f"{key} must be > 0, got {value}")
    vocab = ints.get("VOCAB_SIZE")
    if vocab is not None and not 256 <= vocab <= 65536:
        errors.append(f"VOCAB_SIZE must be in [256, 65536], got {vocab}")
    model_dim, num_heads = ints.get("MODEL_DIM"), ints.get("NUM_HEADS")
    if model_dim is not None and num_heads is not None and model_dim % num_heads != 0:
        errors.append(f"MODEL_DIM must be divisible by NUM_HEADS, got {model_dim} and {num_heads}")
    kv_heads = ints.get("NUM_KV_HEADS")
    if num_heads is not None and kv_heads is not None and num_heads % kv_heads != 0:
        errors.append(f"NUM_HEADS must be divisible by NUM_KV_HEADS, got {num_heads} and {kv_heads}")
    batch_tokens, seq_len = ints.get("TRAIN_BATCH_TOKENS"), ints.get("TRAIN_SEQ_LEN")
    if batch_tokens is not None and seq_len is not None and batch_tokens % seq_len != 0:
        errors.append(f"TRAIN_BATCH_TOKENS must be divisible by TRAIN_SEQ_LEN, got {batch_tokens} and {seq_len}")
    return warnings, errors


def validate_config(config_path: Path) -> ValidationResult:
    cfg, warnings, errors = load_config(config_path)
    if errors:
        return ValidationResult(config={}, runs=[], warnings=warnings, errors=errors, allowlist=set())
    runs, run_warnings, run_errors = expand_runs(cfg)
    allowlist = extract_allowlist()
    warnings.extend(run_warnings)
    errors.extend(run_errors)
    for index, run in enumerate(runs, start=1):
        env_warnings, env_errors = validate_env(run.env, allowlist)
        warnings.extend([f"run {index}: {message}" for message in env_warnings])
        errors.extend([f"run {index}: {message}" for message in env_errors])
    return ValidationResult(config=cfg, runs=runs, warnings=warnings, errors=errors, allowlist=allowlist)


def load_machines() -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not MACHINES_FILE.exists():
        return {}, [f"Machine inventory not found: {MACHINES_FILE}"]
    data, errors = load_yaml(MACHINES_FILE)
    machines = data.get("machines", {})
    if not isinstance(machines, dict):
        return {}, ["machines.yaml must contain a top-level 'machines' mapping"]
    return {str(name): info for name, info in machines.items() if isinstance(info, dict)}, errors


def resolve_machine(host_arg: str) -> tuple[str, dict[str, Any] | None, list[str]]:
    machines, errors = load_machines()
    if errors:
        return host_arg, None, errors
    if host_arg in machines:
        return host_arg, machines[host_arg], []
    for name, info in machines.items():
        if str(info.get("host")) == host_arg:
            return name, info, []
    return host_arg, None, [f"Unknown host {host_arg!r}; add it to {MACHINES_FILE}"]
