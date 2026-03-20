from __future__ import annotations

import torch
from torch import Tensor


def apply_logit_softcap(logits: Tensor, cap_pos: float, cap_neg: float) -> Tensor:
    if cap_pos <= 0.0 or cap_neg <= 0.0:
        raise ValueError(f"logit softcaps must be positive, got cap_pos={cap_pos}, cap_neg={cap_neg}")
    if cap_pos == cap_neg:
        return cap_pos * torch.tanh(logits / cap_pos)
    cap = torch.where(
        logits >= 0,
        torch.full((), cap_pos, dtype=logits.dtype, device=logits.device),
        torch.full((), cap_neg, dtype=logits.dtype, device=logits.device),
    )
    return cap * torch.tanh(logits / cap)
