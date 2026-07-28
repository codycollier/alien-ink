"""Tests for credential / env loading."""

from __future__ import annotations

import os

from alien_ink import env as env_mod
from alien_ink.env import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT


def test_load_env_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    cfg = env_mod.load_env(tmp_path / ".env", verbose=False)

    assert cfg.hf_token is None
    assert cfg.wandb_api_key is None
    assert cfg.wandb_entity == DEFAULT_WANDB_ENTITY
    assert cfg.wandb_project == DEFAULT_WANDB_PROJECT


def test_load_env_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "HF_TOKEN=hf_from_file\nWANDB_API_KEY=wandb_from_file\n",
        encoding="utf-8",
    )

    cfg = env_mod.load_env(env_file, verbose=False)

    assert cfg.hf_token == "hf_from_file"
    assert cfg.wandb_api_key == "wandb_from_file"
    assert os.environ["HF_TOKEN"] == "hf_from_file"
    assert os.environ["WANDB_API_KEY"] == "wandb_from_file"


def test_load_env_kwargs_override_entity_project(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_ENTITY", "should-be-cleared")
    monkeypatch.setenv("WANDB_PROJECT", "should-be-cleared")

    cfg = env_mod.load_env(
        tmp_path / ".env",
        wandb_entity="custom-entity",
        wandb_project="custom-project",
        verbose=False,
    )

    assert cfg.wandb_entity == "custom-entity"
    assert cfg.wandb_project == "custom-project"
    assert "WANDB_ENTITY" not in os.environ
    assert "WANDB_PROJECT" not in os.environ
