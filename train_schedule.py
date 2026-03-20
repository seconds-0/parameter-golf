from __future__ import annotations

import math


def lr_schedule_multiplier(
    *,
    schedule: str,
    step: int,
    iterations: int,
    warmdown_iters: int,
    elapsed_ms: float,
    max_wallclock_ms: float | None,
    wsd_warmup_frac: float,
    wsd_stable_frac: float,
    wsd_decay_style: str,
) -> float:
    if schedule == "baseline":
        return baseline_lr_multiplier(
            step=step,
            iterations=iterations,
            warmdown_iters=warmdown_iters,
            elapsed_ms=elapsed_ms,
            max_wallclock_ms=max_wallclock_ms,
        )
    if schedule == "wsd":
        return wsd_lr_multiplier(
            step=step,
            iterations=iterations,
            elapsed_ms=elapsed_ms,
            max_wallclock_ms=max_wallclock_ms,
            warmup_frac=wsd_warmup_frac,
            stable_frac=wsd_stable_frac,
            decay_style=wsd_decay_style,
        )
    raise ValueError(f"unknown LR_SCHEDULE={schedule!r}")


def beta2_for_schedule(
    *,
    base_beta2: float,
    cooldown_beta2: float,
    enable_cooldown_beta2: bool,
    schedule: str,
    step: int,
    iterations: int,
    elapsed_ms: float,
    max_wallclock_ms: float | None,
    wsd_warmup_frac: float,
    wsd_stable_frac: float,
) -> float:
    if not enable_cooldown_beta2:
        return base_beta2
    if schedule != "wsd":
        return base_beta2
    progress = training_progress(
        step=step,
        iterations=iterations,
        elapsed_ms=elapsed_ms,
        max_wallclock_ms=max_wallclock_ms,
    )
    return cooldown_beta2 if progress >= wsd_warmup_frac + wsd_stable_frac else base_beta2


def baseline_lr_multiplier(
    *,
    step: int,
    iterations: int,
    warmdown_iters: int,
    elapsed_ms: float,
    max_wallclock_ms: float | None,
) -> float:
    if warmdown_iters <= 0:
        return 1.0
    if max_wallclock_ms is None:
        warmdown_start = max(iterations - warmdown_iters, 0)
        if warmdown_start <= step < iterations:
            return max((iterations - step) / max(warmdown_iters, 1), 0.0)
        return 1.0
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = warmdown_iters * step_ms
    remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
    return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


def wsd_lr_multiplier(
    *,
    step: int,
    iterations: int,
    elapsed_ms: float,
    max_wallclock_ms: float | None,
    warmup_frac: float,
    stable_frac: float,
    decay_style: str,
) -> float:
    warmup_frac = max(warmup_frac, 0.0)
    stable_frac = max(stable_frac, 0.0)
    if warmup_frac + stable_frac >= 1.0:
        raise ValueError("WSD warmup and stable fractions must sum to less than 1.0")

    progress = training_progress(
        step=step,
        iterations=iterations,
        elapsed_ms=elapsed_ms,
        max_wallclock_ms=max_wallclock_ms,
    )
    warmup_end = warmup_frac
    stable_end = warmup_frac + stable_frac

    if warmup_end > 0.0 and progress < warmup_end:
        return progress / warmup_end
    if progress < stable_end:
        return 1.0

    decay_frac = (progress - stable_end) / max(1.0 - stable_end, 1e-9)
    decay_frac = min(max(decay_frac, 0.0), 1.0)
    if decay_style == "linear":
        return 1.0 - decay_frac
    if decay_style == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * decay_frac))
    raise ValueError(f"unknown WSD_DECAY_STYLE={decay_style!r}")


def training_progress(
    *,
    step: int,
    iterations: int,
    elapsed_ms: float,
    max_wallclock_ms: float | None,
) -> float:
    if max_wallclock_ms is not None and max_wallclock_ms > 0:
        return min(max(elapsed_ms / max_wallclock_ms, 0.0), 1.0)
    if iterations <= 0:
        return 1.0
    return min(max(step / iterations, 0.0), 1.0)
