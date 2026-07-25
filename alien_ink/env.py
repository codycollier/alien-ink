"""Environment / credential loading for experiment runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# W&B team (entity) + project defaults used everywhere.
DEFAULT_WANDB_ENTITY = "logbook"
DEFAULT_WANDB_PROJECT = "ink-explore"


@dataclass(frozen=True)
class EnvConfig:
    hf_token: str | None
    wandb_api_key: str | None
    wandb_entity: str
    wandb_project: str


def load_env(
    *env_files: Path,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_entity_fallback: str = DEFAULT_WANDB_ENTITY,
    wandb_project_fallback: str = DEFAULT_WANDB_PROJECT,
    verbose: bool = True,
) -> EnvConfig:
    """Load dotenv files and normalize HF / W&B API credentials.

    W&B entity / project come from kwargs or the code fallbacks only — never
    from the process environment or ``.env``. Pass entity / project / run name
    via CLI flags or function kwargs instead.
    """
    if env_files:
        load_dotenv(env_files[0])
        for path in env_files[1:]:
            load_dotenv(path, override=True)

    # Entity / project / run name are CLI/kwargs only — drop any values dotenv
    # (or the parent shell) may have set so the W&B SDK cannot pick them up.
    os.environ.pop("WANDB_ENTITY", None)
    os.environ.pop("WANDB_PROJECT", None)
    os.environ.pop("WANDB_NAME", None)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    resolved_entity = wandb_entity or wandb_entity_fallback
    resolved_project = wandb_project or wandb_project_fallback

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    if verbose:
        shown = ", ".join(str(p) for p in env_files) if env_files else "(none)"
        print(f"   env file(s):   {shown}")
        print(f"   HF_TOKEN:      {'set' if hf_token else 'missing'}")
        print(f"   WANDB_API_KEY: {'set' if wandb_api_key else 'missing'}")
        print(f"   wandb entity:  {resolved_entity}")
        print(f"   wandb project: {resolved_project}")

    return EnvConfig(
        hf_token=hf_token,
        wandb_api_key=wandb_api_key,
        wandb_entity=resolved_entity,
        wandb_project=resolved_project,
    )
