from __future__ import annotations

import torch
from torch import Tensor


def apply_logit_softcap(logits: Tensor, cap_pos: float, cap_neg: float) -> Tensor:
    if cap_pos <= 0.0 or cap_neg <= 0.0:
        raise ValueError(f"logit softcaps must be positive, got cap_pos={cap_pos}, cap_neg={cap_neg}")
    pos = cap_pos * torch.tanh(torch.clamp_min(logits, 0.0) / cap_pos)
    neg = cap_neg * torch.tanh(torch.clamp_max(logits, 0.0) / cap_neg)
    return pos + neg
