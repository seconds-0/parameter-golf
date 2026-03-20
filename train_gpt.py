"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
"""
from __future__ import annotations

import copy
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ema_export import build_export_state_dict, init_ema_parameters, update_ema_parameters
from logit_softcap import apply_logit_softcap
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from train_data import DistributedTokenLoader, build_sentencepiece_luts, load_validation_tokens
from train_schedule import beta2_for_schedule, lr_schedule_multiplier

# Optional wandb integration: set WANDB_PROJECT env var to enable
_WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
if _WANDB_PROJECT:
    import wandb
# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    doc_aligned_batching = bool(int(os.environ.get("DOC_ALIGNED_BATCHING", "0")))
    ema_export = bool(int(os.environ.get("EMA_EXPORT", "0")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.999))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    logit_softcap_pos = float(os.environ.get("LOGIT_SOFTCAP_POS", logit_softcap))
    logit_softcap_neg = float(os.environ.get("LOGIT_SOFTCAP_NEG", logit_softcap))
    value_embed_gate = int(os.environ.get("VALUE_EMBED_GATE", "0"))
    aux_loss_weight_decay = float(os.environ.get("AUX_LOSS_WEIGHT_DECAY", "0.0"))
    aux_loss_range_reg = float(os.environ.get("AUX_LOSS_RANGE_REG", "0.0"))
    aux_loss_clamp_reg = float(os.environ.get("AUX_LOSS_CLAMP_REG", "0.0"))
    cautious_weight_decay = float(os.environ.get("CAUTIOUS_WEIGHT_DECAY", "0.0"))
    batch_schedule = os.environ.get("BATCH_SCHEDULE", "")

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    cooldown_beta2 = float(os.environ.get("COOLDOWN_BETA2", 0.98))
    enable_cooldown_beta2 = bool(int(os.environ.get("ENABLE_COOLDOWN_BETA2", "0")))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    lr_schedule = os.environ.get("LR_SCHEDULE", "baseline")
    wsd_warmup_frac = float(os.environ.get("WSD_WARMUP_FRAC", 0.01))
    wsd_stable_frac = float(os.environ.get("WSD_STABLE_FRAC", 0.75))
    wsd_decay_style = os.environ.get("WSD_DECAY_STYLE", "cosine")

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,ve_gate_w",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = int(os.environ.get("INT8_KEEP_FLOAT_MAX_NUMEL", 65_536))
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = float(os.environ.get("INT8_CLIP_PERCENTILE", 99.99984))
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


LOSS_IGNORE_INDEX = -100

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = self._build_inv_freq()
        # Persist the RoPE frequency basis so fresh-process checkpoint loads do not
        # depend on reconstructing forward-relevant buffers from ambient config.
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _build_inv_freq(self, device: torch.device | None = None) -> Tensor:
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _apply(self, fn):
        super()._apply(fn)
        # Keep the RoPE basis in fp32 even when the model body runs in bf16 so
        # cache rebuilds are numerically stable across live and replay paths.
        self.inv_freq = self._build_inv_freq(self.inv_freq.device)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        return self

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != torch.float32
            or self._sin_cached.dtype != torch.float32
        ):
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        value_embed_gate: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.ve_gate_w = nn.Parameter(torch.zeros(1, dtype=torch.float32)) if value_embed_gate else None

    def forward(self, x: Tensor, tok_emb: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.ve_gate_w is not None and tok_emb is not None:
            kv_dim = self.num_kv_heads * self.head_dim
            ve = tok_emb[..., :kv_dim].reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v + self.ve_gate_w.to(dtype=v.dtype) * ve
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        value_embed_gate: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, value_embed_gate)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), tok_emb=x0)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        logit_softcap_pos: float | None = None,
        logit_softcap_neg: float | None = None,
        value_embed_gate: bool = False,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        logit_softcap_pos = logit_softcap if logit_softcap_pos is None else logit_softcap_pos
        logit_softcap_neg = logit_softcap if logit_softcap_neg is None else logit_softcap_neg
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.logit_softcap_pos = logit_softcap_pos
        self.logit_softcap_neg = logit_softcap_neg
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    value_embed_gate,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = apply_logit_softcap(logits_proj, self.logit_softcap_pos, self.logit_softcap_neg)
        return F.cross_entropy(logits.float(), targets, reduction="mean", ignore_index=LOSS_IGNORE_INDEX)


# -----------------------------
# REPLAY DIAGNOSTICS
# -----------------------------
# These functions capture forward-pass-affecting state for comparing
# in-process vs fresh-process model reconstruction.  Output as structured
# DIAG: lines that can be diffed between training and export_eval logs.

import json as _json


def hyperparams_fingerprint(args: Hyperparameters, label: str = "hparams") -> None:
    fields = [
        "vocab_size", "num_layers", "model_dim", "num_heads", "num_kv_heads",
        "mlp_mult", "tie_embeddings", "logit_softcap", "rope_base",
        "qk_gain_init", "tied_embed_init_std", "train_seq_len", "val_batch_size",
        "logit_softcap_pos", "logit_softcap_neg",
    ]
    info: dict[str, object] = {f: getattr(args, f) for f in fields}
    info["CONTROL_TENSOR_NAME_PATTERNS"] = list(CONTROL_TENSOR_NAME_PATTERNS)
    info["INT8_KEEP_FLOAT_MAX_NUMEL"] = INT8_KEEP_FLOAT_MAX_NUMEL
    info["INT8_CLIP_PERCENTILE"] = INT8_CLIP_PERCENTILE
    print(f"DIAG:{label}:{_json.dumps(info, default=str)}")


def _tensor_diag(tensor: Tensor) -> str:
    tensor_float = tensor.detach().float()
    return (
        f"{tensor_float.mean().item():.8f}|{tensor_float.std().item():.8f}"
        f"|{tensor.dtype}|{list(tensor.shape)}"
    )


def _cache_tensor_diag(tensor: Tensor | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    tensor_float = tensor.detach().float()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "mean": round(tensor_float.mean().item(), 8),
        "std": round(tensor_float.std().item(), 8),
    }


def rotary_cache_state(rotary: Rotary) -> dict[str, object]:
    return {
        "seq_len_cached": rotary._seq_len_cached,
        "cos_cached": _cache_tensor_diag(rotary._cos_cached),
        "sin_cached": _cache_tensor_diag(rotary._sin_cached),
    }


def reset_rotary_caches(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, Rotary):
            module._seq_len_cached = 0
            module._cos_cached = None
            module._sin_cached = None


def prewarm_rotary_caches(model: nn.Module, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
    for module in model.modules():
        if isinstance(module, Rotary):
            module(seq_len, device, dtype)


def model_fingerprint(model: nn.Module, label: str = "fingerprint") -> None:
    info: dict[str, object] = {
        "logit_softcap": model.logit_softcap,
        "logit_softcap_pos": model.logit_softcap_pos,
        "logit_softcap_neg": model.logit_softcap_neg,
        "tie_embeddings": model.tie_embeddings,
        "num_encoder_layers": model.num_encoder_layers,
        "num_decoder_layers": model.num_decoder_layers,
        "num_skip_weights": model.num_skip_weights,
        "has_lm_head": model.lm_head is not None,
    }
    for i, block in enumerate(model.blocks):
        a = block.attn
        info[f"block{i}.heads"] = f"{a.num_heads}/{a.num_kv_heads}/{a.head_dim}"
        info[f"block{i}.rotary_cache"] = rotary_cache_state(a.rotary)
    params: dict[str, str] = {}
    for name, p in model.named_parameters():
        params[name] = (
            f"dtype={p.dtype}|shape={list(p.shape)}|dev={p.device}"
            f"|sum_abs={p.detach().float().abs().sum().item():.6f}"
            f"|mean={p.detach().float().mean().item():.8f}"
        )
    info["params"] = params
    bufs: dict[str, str] = {}
    for name, b in model.named_buffers():
        bufs[name] = (
            f"dtype={b.dtype}|shape={list(b.shape)}|dev={b.device}"
            f"|sum={b.float().sum().item():.8f}"
        )
    info["buffers"] = bufs
    print(f"DIAG:{label}:{_json.dumps(info, default=str)}")


@torch.no_grad()
def diagnostic_forward(
    model: nn.Module, seq_len: int, vocab_size: int,
    device: torch.device, label: str = "diag_fwd",
) -> None:
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    y = torch.randint(0, vocab_size, (1, seq_len), device=device)
    was_training = model.training
    model.eval()
    stats: dict[str, str] = {}
    emb = model.tok_emb(x)
    h = F.rms_norm(emb, (emb.size(-1),))
    stats["emb"] = f"{h.float().mean().item():.8f}|{h.float().std().item():.8f}"
    x0 = h.clone()
    skips: list[Tensor] = []
    for i in range(model.num_encoder_layers):
        h = model.blocks[i](h, x0)
        skips.append(h)
        stats[f"enc{i}"] = f"{h.float().mean().item():.8f}|{h.float().std().item():.8f}"
    for i in range(model.num_decoder_layers):
        if skips:
            h = h + model.skip_weights[i].to(dtype=h.dtype)[None, None, :] * skips.pop()
        h = model.blocks[model.num_encoder_layers + i](h, x0)
        stats[f"dec{i}"] = f"{h.float().mean().item():.8f}|{h.float().std().item():.8f}"
    final = model.final_norm(h).reshape(-1, h.size(-1))
    if model.tie_embeddings:
        logits_proj = F.linear(final, model.tok_emb.weight)
    else:
        logits_proj = model.lm_head(final)
    logits = apply_logit_softcap(logits_proj, model.logit_softcap_pos, model.logit_softcap_neg)
    loss = F.cross_entropy(logits.float(), y.reshape(-1), reduction="mean")
    stats["logits"] = f"{logits.float().mean().item():.8f}|{logits.float().std().item():.8f}|loss={loss.item():.8f}"
    if was_training:
        model.train()
    print(f"DIAG:{label}:{_json.dumps(stats)}")


@torch.no_grad()
def diagnostic_block0_forward(
    model: nn.Module, seq_len: int, vocab_size: int,
    device: torch.device, label: str = "diag_block0",
) -> None:
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    was_training = model.training
    model.eval()

    emb = model.tok_emb(x)
    h = F.rms_norm(emb, (emb.size(-1),))
    x0 = h.clone()
    block = model.blocks[0]
    mix = block.resid_mix.to(dtype=h.dtype)
    h = mix[0][None, None, :] * h + mix[1][None, None, :] * x0
    attn_norm = block.attn_norm(h)

    attn = block.attn
    bsz, block_seq_len, dim = attn_norm.shape
    q = attn.c_q(attn_norm).reshape(bsz, block_seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.c_k(attn_norm).reshape(bsz, block_seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.c_v(attn_norm).reshape(bsz, block_seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))

    stats: dict[str, object] = {
        "emb": _tensor_diag(F.rms_norm(emb, (emb.size(-1),))),
        "block0_input": _tensor_diag(h),
        "attn_norm": _tensor_diag(attn_norm),
        "q_pre_rope": _tensor_diag(q),
        "k_pre_rope": _tensor_diag(k),
        "v": _tensor_diag(v),
        "rotary_cache_before": rotary_cache_state(attn.rotary),
    }

    cos, sin = attn.rotary(block_seq_len, attn_norm.device, q.dtype)
    stats["cos"] = _tensor_diag(cos)
    stats["sin"] = _tensor_diag(sin)
    stats["rotary_cache_after"] = rotary_cache_state(attn.rotary)

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    stats["q_post_rope"] = _tensor_diag(q)
    stats["k_post_rope"] = _tensor_diag(k)

    q = q * attn.q_gain.to(dtype=q.dtype)[None, :, None, None]
    y = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        enable_gqa=(attn.num_kv_heads != attn.num_heads),
    )
    y = y.transpose(1, 2).contiguous().reshape(bsz, block_seq_len, dim)
    attn_out = attn.proj(y)
    stats["attn_out"] = _tensor_diag(attn_out)

    h = h + block.attn_scale.to(dtype=h.dtype)[None, None, :] * attn_out
    mlp_norm = block.mlp_norm(h)
    mlp_out = block.mlp(mlp_norm)
    stats["mlp_norm"] = _tensor_diag(mlp_norm)
    stats["mlp_out"] = _tensor_diag(mlp_out)
    h = h + block.mlp_scale.to(dtype=h.dtype)[None, None, :] * mlp_out
    stats["enc0"] = _tensor_diag(h)

    if was_training:
        model.train()
    print(f"DIAG:{label}:{_json.dumps(stats, default=str)}")


# -----------------------------
# BATCH SCHEDULE
# -----------------------------

def parse_batch_schedule(schedule_str: str) -> list[tuple[float, int]]:
    """Parse a batch schedule string like '0.3:131072,1.0:524288' into sorted stages."""
    if not schedule_str.strip():
        return []
    stages: list[tuple[float, int]] = []
    for part in schedule_str.split(","):
        frac_str, tokens_str = part.strip().split(":")
        frac, tokens = float(frac_str), int(tokens_str)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"batch schedule fraction must be in (0, 1], got {frac}")
        if tokens <= 0:
            raise ValueError(f"batch schedule tokens must be positive, got {tokens}")
        stages.append((frac, tokens))
    stages.sort(key=lambda s: s[0])
    return stages


def scheduled_batch_tokens(step: int, iterations: int, stages: list[tuple[float, int]], default: int) -> int:
    """Return the batch token count for the current step based on the schedule.

    Each stage is (frac, tokens) where frac is the upper fraction boundary.
    Example: [(0.3, 131072), (1.0, 524288)] means use 131072 for steps 0-30%,
    then 524288 for steps 30%-100%.
    """
    if not stages:
        return default
    frac = step / max(iterations, 1)
    for stage_frac, stage_tokens in stages:
        if frac < stage_frac:
            return stage_tokens
    return stages[-1][1]


# -----------------------------
# AUXILIARY REGULARIZATION LOSSES
# -----------------------------

def compute_aux_losses(model: nn.Module, args: Hyperparameters) -> Tensor:
    """Compute auxiliary regularization losses on 2D weight matrices.

    Called once per step (outside the micro-step loop) since these penalties
    depend on the current weights, not on the data batch.
    """
    aux = torch.zeros((), device=next(model.parameters()).device)
    if args.aux_loss_weight_decay == 0 and args.aux_loss_range_reg == 0 and args.aux_loss_clamp_reg == 0:
        return aux
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        if any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            continue
        p_float = p.float()
        if args.aux_loss_weight_decay > 0:
            aux = aux + args.aux_loss_weight_decay * p_float.pow(2).mean()
        if args.aux_loss_range_reg > 0:
            row_ranges = p_float.max(dim=-1).values - p_float.min(dim=-1).values
            aux = aux + args.aux_loss_range_reg * row_ranges.mean()
        if args.aux_loss_clamp_reg > 0:
            with torch.no_grad():
                thresholds = torch.quantile(p_float.abs(), INT8_CLIP_Q, dim=-1, keepdim=True)
            excess = torch.clamp(p_float.abs() - thresholds, min=0)
            aux = aux + args.aux_loss_clamp_reg * excess.pow(2).mean()
    return aux


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    hyperparams_fingerprint(args, "train_hparams")
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    doc_boundary_token_id = int(sp.bos_id())
    if args.doc_aligned_batching and doc_boundary_token_id < 0:
        raise ValueError("DOC_ALIGNED_BATCHING=1 requires a tokenizer with bos_id")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(
        f"doc_aligned_batching:{args.doc_aligned_batching} boundary_token_id:{doc_boundary_token_id} "
        f"ema_export:{args.ema_export} ema_decay:{args.ema_decay:.6f}"
    )

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        logit_softcap_pos=args.logit_softcap_pos,
        logit_softcap_neg=args.logit_softcap_neg,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        value_embed_gate=bool(args.value_embed_gate),
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(
        f"logit_softcap:{args.logit_softcap:.6f} "
        f"logit_softcap_pos:{args.logit_softcap_pos:.6f} "
        f"logit_softcap_neg:{args.logit_softcap_neg:.6f}"
    )
    log0(
        f"seed:{args.seed} lr_schedule:{args.lr_schedule} "
        f"wsd_warmup_frac:{args.wsd_warmup_frac:.6f} "
        f"wsd_stable_frac:{args.wsd_stable_frac:.6f} "
        f"wsd_decay_style:{args.wsd_decay_style} "
        f"enable_cooldown_beta2:{args.enable_cooldown_beta2} "
        f"cooldown_beta2:{args.cooldown_beta2:.6f}"
    )

    if _WANDB_PROJECT and master_process:
        wandb.init(project=_WANDB_PROJECT, name=args.run_id, config={
            k: getattr(args, k) for k in vars(args) if not k.startswith("_")
        })

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(
        args.train_files,
        rank,
        world_size,
        device,
        doc_aligned_batching=args.doc_aligned_batching,
        boundary_token_id=doc_boundary_token_id,
        pad_token_id=doc_boundary_token_id,
    )

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    any_aux_loss = args.aux_loss_weight_decay > 0 or args.aux_loss_range_reg > 0 or args.aux_loss_clamp_reg > 0
    if any_aux_loss:
        log0(
            f"aux_losses:weight_decay={args.aux_loss_weight_decay} "
            f"range_reg={args.aux_loss_range_reg} clamp_reg={args.aux_loss_clamp_reg}"
        )
    batch_stages = parse_batch_schedule(args.batch_schedule)
    if batch_stages:
        log0(f"batch_schedule:{args.batch_schedule} stages:{batch_stages}")
    if args.cautious_weight_decay > 0:
        log0(f"cautious_weight_decay:{args.cautious_weight_decay}")
    if args.cooldown_beta2 > 0:
        log0(f"cooldown_beta2:{args.cooldown_beta2}")

    def lr_mul(step: int, elapsed_ms: float) -> float:
        return lr_schedule_multiplier(
            schedule=args.lr_schedule,
            step=step,
            iterations=args.iterations,
            warmdown_iters=args.warmdown_iters,
            elapsed_ms=elapsed_ms,
            max_wallclock_ms=max_wallclock_ms,
            wsd_warmup_frac=args.wsd_warmup_frac,
            wsd_stable_frac=args.wsd_stable_frac,
            wsd_decay_style=args.wsd_decay_style,
        )

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files,
            rank,
            world_size,
            device,
            doc_aligned_batching=args.doc_aligned_batching,
            boundary_token_id=doc_boundary_token_id,
            pad_token_id=doc_boundary_token_id,
        )

    ema_params = init_ema_parameters(base_model) if args.ema_export else None

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    train_tokens_seen = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if _WANDB_PROJECT and master_process:
                wandb.log({"val_loss": val_loss, "val_bpb": val_bpb}, step=step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        effective_batch_tokens = scheduled_batch_tokens(step, args.iterations, batch_stages, args.train_batch_tokens)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(effective_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            if any_aux_loss:
                # Recompute per micro-step so each backward sees a fresh graph.
                loss = loss + compute_aux_losses(base_model, args)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        current_beta2 = beta2_for_schedule(
            base_beta2=args.beta2,
            cooldown_beta2=args.cooldown_beta2,
            enable_cooldown_beta2=args.enable_cooldown_beta2,
            schedule=args.lr_schedule,
            step=step,
            iterations=args.iterations,
            elapsed_ms=elapsed_ms,
            max_wallclock_ms=max_wallclock_ms,
            wsd_warmup_frac=args.wsd_warmup_frac,
            wsd_stable_frac=args.wsd_stable_frac,
        )
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
                if "betas" in group:
                    group["betas"] = (group["betas"][0], current_beta2)

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        if args.cautious_weight_decay > 0:
            with torch.no_grad():
                for name, p in base_model.named_parameters():
                    if p.ndim != 2 or p.grad is None:
                        continue
                    if any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
                        continue
                    # Decay only where gradient and weight agree in sign
                    # (gradient is already pushing toward zero)
                    mask = (p.data.sign() == p.grad.sign()).to(dtype=p.dtype)
                    p.data.mul_(1.0 - scale * args.cautious_weight_decay * mask)
        if ema_params is not None:
            update_ema_parameters(ema_params, base_model, args.ema_decay)
        if ema_params is not None:
            update_ema_parameters(ema_params, base_model, args.ema_decay)
        zero_grad_all()

        step += 1
        train_tokens_seen += effective_batch_tokens
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            tok_s = train_tokens_seen / max(approx_training_time_ms / 1000.0, 1e-9)
            supervised_tokens_seen = train_loader.supervised_tokens_seen
            ignored_target_tokens_seen = train_loader.ignored_target_tokens_seen
            supervised_target_fraction = train_loader.supervised_target_fraction
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"tok_s:{tok_s:.2f} train_tokens_seen:{train_tokens_seen} "
                f"train_supervised_tokens_seen:{supervised_tokens_seen} "
                f"ignored_target_tokens_seen:{ignored_target_tokens_seen} "
                f"supervised_target_fraction:{supervised_target_fraction:.6f}"
            )
            if _WANDB_PROJECT and master_process:
                wandb.log(
                    {
                        "train_loss": train_loss.item(),
                        "step_avg_ms": approx_training_time_ms / step,
                        "tok_s": tok_s,
                        "train_tokens_seen": train_tokens_seen,
                        "train_supervised_tokens_seen": supervised_tokens_seen,
                        "ignored_target_tokens_seen": ignored_target_tokens_seen,
                        "supervised_target_fraction": supervised_target_fraction,
                    },
                    step=step,
                )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    train_supervised_tokens_seen = train_loader.supervised_tokens_seen
    ignored_target_tokens_seen = train_loader.ignored_target_tokens_seen
    supervised_target_fraction = train_loader.supervised_target_fraction
    train_supervised_tokens_seen = train_loader.supervised_tokens_seen
    ignored_target_tokens_seen = train_loader.ignored_target_tokens_seen
    supervised_target_fraction = train_loader.supervised_target_fraction

    def exact_eval(eval_model: nn.Module) -> tuple[float, float, int]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        val_loss, val_bpb = eval_val(
            args,
            eval_model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        return val_loss, val_bpb, int(1000.0 * (time.perf_counter() - start))

    def log_exact_eval(
        label: str,
        eval_model: nn.Module,
        *,
        reference: tuple[float, float] | None = None,
        include_train_tokens_seen: bool = False,
    ) -> tuple[float, float]:
        val_loss, val_bpb, eval_time_ms = exact_eval(eval_model)
        delta_suffix = ""
        if reference is not None:
            delta_suffix = (
                f" delta_loss:{val_loss - reference[0]:.8f} "
                f"delta_bpb:{val_bpb - reference[1]:.8f}"
            )
        train_tokens_suffix = (
            f" train_tokens_seen:{train_tokens_seen}"
            f" train_supervised_tokens_seen:{train_supervised_tokens_seen}"
            f" ignored_target_tokens_seen:{ignored_target_tokens_seen}"
            f" supervised_target_fraction:{supervised_target_fraction:.6f}"
            if include_train_tokens_seen
            else ""
        )
        log0(
            f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}"
            f"{delta_suffix}{train_tokens_suffix} eval_time:{eval_time_ms}ms"
        )
        return val_loss, val_bpb

    def log_checkpoint_save_verify(path: str, state_dict: dict[str, Tensor]) -> None:
        if not master_process:
            return
        reloaded_state = torch.load(path, map_location="cpu")
        max_abs_diff = 0.0
        tensors_mismatched = 0
        for name, tensor in state_dict.items():
            reloaded_tensor = reloaded_state.get(name)
            if reloaded_tensor is None or tensor.shape != reloaded_tensor.shape:
                tensors_mismatched += 1
                max_abs_diff = float("inf")
                continue
            if tensor.numel() == 0:
                diff = 0.0
            else:
                diff = (tensor.detach().float().cpu() - reloaded_tensor.detach().float()).abs().max().item()
            if diff > 0.0:
                tensors_mismatched += 1
            max_abs_diff = max(max_abs_diff, diff)
        log0(
            f"checkpoint_save_verify max_abs_diff:{max_abs_diff:.8e} "
            f"tensors_mismatched:{tensors_mismatched}"
        )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if ema_params is not None:
        log_exact_eval(
            "live_prequant_exact",
            model,
            include_train_tokens_seen=True,
        )
        base_model.load_state_dict(build_export_state_dict(base_model, ema_params), strict=True)
        log0(f"ema_export_applied decay:{args.ema_decay:.6f} tracked_tensors:{len(ema_params)}")

    pre_val_loss, pre_val_bpb = log_exact_eval(
        "final_prequant_exact",
        model,
        include_train_tokens_seen=True,
    )
    if model is not base_model:
        log_exact_eval(
            "uncompiled_check",
            base_model,
            reference=(pre_val_loss, pre_val_bpb),
        )
    diagnostic_block0_forward(base_model, args.train_seq_len, args.vocab_size, device, "train_live_block0")

    if master_process:
        raw_state_dict = build_export_state_dict(base_model, ema_params)
        torch.save(raw_state_dict, "final_model.pt")
        log_checkpoint_save_verify("final_model.pt", raw_state_dict)
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    raw_state_dict = torch.load("final_model.pt", map_location="cpu")
    base_model.load_state_dict(raw_state_dict, strict=True)
    model_fingerprint(base_model, "train_reloaded")
    diagnostic_forward(base_model, args.train_seq_len, args.vocab_size, device, "train_diag_fwd")
    diagnostic_block0_forward(base_model, args.train_seq_len, args.vocab_size, device, "train_reloaded_block0")
    reset_rotary_caches(base_model)
    diagnostic_block0_forward(
        base_model,
        args.train_seq_len,
        args.vocab_size,
        device,
        "train_reloaded_block0_cache_cleared",
    )
    reset_rotary_caches(base_model)
    prewarm_rotary_caches(base_model, args.train_seq_len, device, base_model.tok_emb.weight.dtype)
    diagnostic_block0_forward(
        base_model,
        args.train_seq_len,
        args.vocab_size,
        device,
        "train_reloaded_block0_cache_prewarmed",
    )
    reset_rotary_caches(base_model)
    log_exact_eval(
        "reloaded_prequant_exact",
        base_model,
        reference=(pre_val_loss, pre_val_bpb),
    )

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    q_val_loss, q_val_bpb, q_eval_time_ms = exact_eval(model)
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{q_eval_time_ms:.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    diagnostic_block0_forward(base_model, args.train_seq_len, args.vocab_size, device, "train_reloaded_int8_block0")
    reset_rotary_caches(base_model)
    log_exact_eval(
        "reloaded_int8_zlib_roundtrip_exact",
        base_model,
        reference=(q_val_loss, q_val_bpb),
    )
    log0(
        f"quantization_delta_exact val_loss:{q_val_loss - pre_val_loss:.8f} "
        f"val_bpb:{q_val_bpb - pre_val_bpb:.8f} train_tokens_seen:{train_tokens_seen} "
        f"train_supervised_tokens_seen:{train_supervised_tokens_seen} "
        f"ignored_target_tokens_seen:{ignored_target_tokens_seen} "
        f"supervised_target_fraction:{supervised_target_fraction:.6f}"
    )

    if _WANDB_PROJECT and master_process:
        wandb.log(
            {
                "prequant_val_loss": pre_val_loss,
                "prequant_val_bpb": pre_val_bpb,
                "final_val_loss": q_val_loss,
                "final_val_bpb": q_val_bpb,
                "qgap_loss": q_val_loss - pre_val_loss,
                "qgap_bpb": q_val_bpb - pre_val_bpb,
                "train_tokens_seen": train_tokens_seen,
                "train_supervised_tokens_seen": train_supervised_tokens_seen,
                "ignored_target_tokens_seen": ignored_target_tokens_seen,
                "supervised_target_fraction": supervised_target_fraction,
                "model_bytes_int8_zlib": quant_file_bytes,
                "total_submission_bytes": quant_file_bytes + code_bytes,
            }
        )
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
