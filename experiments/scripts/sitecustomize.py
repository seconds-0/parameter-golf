"""Runtime patching for remote experiment processes."""

from __future__ import annotations

import os


def _patch_wandb_init() -> None:
    try:
        import wandb
    except Exception:
        return
    original = getattr(wandb, "init", None)
    if not callable(original) or getattr(original, "_pgolf_wrapped", False):
        return

    def wrapped_init(*args, **kwargs):
        group = os.environ.get("PGOLF_WANDB_GROUP")
        notes = os.environ.get("PGOLF_WANDB_NOTES")
        raw_tags = os.environ.get("PGOLF_WANDB_TAGS", "")
        tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
        if group and "group" not in kwargs:
            kwargs["group"] = group
        if notes and "notes" not in kwargs:
            kwargs["notes"] = notes
        if tags:
            existing = list(kwargs.get("tags") or [])
            kwargs["tags"] = existing + [tag for tag in tags if tag not in existing]
        return original(*args, **kwargs)

    setattr(wrapped_init, "_pgolf_wrapped", True)
    wandb.init = wrapped_init


_patch_wandb_init()
