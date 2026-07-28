"""Environment / credential loading for training runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from alien_ink.log import detail, get_logger

log = get_logger("env")


@dataclass(frozen=True)
class EnvConfig:
    hf_token: str | None
    wandb_api_key: str | None
    wandb_entity: str | None
    wandb_project: str | None


def in_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def _colab_secret(name: str) -> str | None:
    """Read a Colab notebook secret, or None if missing / inaccessible."""
    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        value = userdata.get(name)
    except Exception:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _apply_colab_secrets() -> bool:
    """Copy Colab Secrets into ``os.environ`` for any unset credential keys.

    Expects notebook secrets named ``HF_TOKEN`` and ``WANDB_API_KEY`` (grant
    notebook access in the Colab Secrets panel). Returns True when Colab is
    detected.
    """
    if not in_colab():
        return False

    if not os.getenv("HF_TOKEN") and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
        token = _colab_secret("HF_TOKEN") or _colab_secret("HUGGING_FACE_HUB_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    if not os.getenv("WANDB_API_KEY"):
        key = _colab_secret("WANDB_API_KEY")
        if key:
            os.environ["WANDB_API_KEY"] = key

    return True


def load_env(
    *env_files: Path,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    verbose: bool = True,
) -> EnvConfig:
    """Load dotenv files and normalize HF / W&B API credentials.

    In Google Colab, also reads ``HF_TOKEN`` / ``WANDB_API_KEY`` from the
    notebook Secrets panel when those vars are not already set.

    W&B entity / project come from kwargs only — never from the process
    environment or ``.env``. Pass them explicitly when starting a W&B run.
    """
    if env_files:
        load_dotenv(env_files[0])
        for path in env_files[1:]:
            load_dotenv(path, override=True)

    colab = _apply_colab_secrets()

    # Entity / project / run name are kwargs only — drop any values dotenv
    # (or the parent shell) may have set so the W&B SDK cannot pick them up.
    os.environ.pop("WANDB_ENTITY", None)
    os.environ.pop("WANDB_PROJECT", None)
    os.environ.pop("WANDB_NAME", None)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    if verbose:
        shown = ", ".join(str(p) for p in env_files) if env_files else "(none)"
        detail(f"env file(s):   {shown}", logger=log)
        detail(f"colab secrets: {'yes' if colab else 'no'}", logger=log)
        detail(f"HF_TOKEN:      {'set' if hf_token else 'missing'}", logger=log)
        detail(
            f"WANDB_API_KEY: {'set' if wandb_api_key else 'missing'}",
            logger=log,
        )
        detail(f"wandb entity:  {wandb_entity or '(unset)'}", logger=log)
        detail(f"wandb project: {wandb_project or '(unset)'}", logger=log)

    return EnvConfig(
        hf_token=hf_token,
        wandb_api_key=wandb_api_key,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )
