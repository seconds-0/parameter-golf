from __future__ import annotations

from torch import Tensor, nn


def init_ema_parameters(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: param.detach().float().clone()
        for name, param in model.named_parameters()
        if param.is_floating_point()
    }


def update_ema_parameters(ema_params: dict[str, Tensor], model: nn.Module, decay: float) -> None:
    one_minus_decay = 1.0 - decay
    for name, param in model.named_parameters():
        ema_tensor = ema_params.get(name)
        if ema_tensor is None:
            continue
        ema_tensor.mul_(decay).add_(param.detach().float(), alpha=one_minus_decay)


def build_export_state_dict(model: nn.Module, ema_params: dict[str, Tensor] | None) -> dict[str, Tensor]:
    export_state = {}
    live_state = model.state_dict()
    if not ema_params:
        return live_state
    for name, tensor in live_state.items():
        ema_tensor = ema_params.get(name)
        if ema_tensor is None:
            export_state[name] = tensor.detach().clone()
            continue
        export_state[name] = ema_tensor.to(device=tensor.device, dtype=tensor.dtype)
    return export_state
