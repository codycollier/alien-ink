"""Tests for credential / env loading."""

from __future__ import annotations

import os
import sys
import types

from alien_ink import env as env_mod


def _install_fake_colab(monkeypatch, secrets: dict[str, str]):
    """Register a minimal ``google.colab.userdata`` stub in ``sys.modules``."""

    def get(name: str) -> str:
        if name not in secrets:
            raise KeyError(name)
        return secrets[name]

    userdata = types.ModuleType("google.colab.userdata")
    userdata.get = get  # type: ignore[attr-defined]

    colab = types.ModuleType("google.colab")
    colab.userdata = userdata  # type: ignore[attr-defined]

    google = types.ModuleType("google")
    google.colab = colab  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)


def test_in_colab_false_by_default():
    assert env_mod.in_colab() is False


def test_in_colab_true_when_google_colab_importable(monkeypatch):
    _install_fake_colab(monkeypatch, {})
    assert env_mod.in_colab() is True


def test_load_env_reads_colab_secrets(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    _install_fake_colab(
        monkeypatch,
        {"HF_TOKEN": "hf_from_colab", "WANDB_API_KEY": "wandb_from_colab"},
    )

    cfg = env_mod.load_env(tmp_path / ".env", verbose=False)

    assert cfg.hf_token == "hf_from_colab"
    assert cfg.wandb_api_key == "wandb_from_colab"
    assert os.environ["HF_TOKEN"] == "hf_from_colab"
    assert os.environ["WANDB_API_KEY"] == "wandb_from_colab"


def test_load_env_existing_environ_wins_over_colab(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    monkeypatch.setenv("WANDB_API_KEY", "wandb_from_env")
    _install_fake_colab(
        monkeypatch,
        {"HF_TOKEN": "hf_from_colab", "WANDB_API_KEY": "wandb_from_colab"},
    )

    cfg = env_mod.load_env(tmp_path / ".env", verbose=False)

    assert cfg.hf_token == "hf_from_env"
    assert cfg.wandb_api_key == "wandb_from_env"


def test_load_env_colab_secret_errors_are_ignored(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    _install_fake_colab(monkeypatch, {})  # get() raises for any name

    cfg = env_mod.load_env(tmp_path / ".env", verbose=False)

    assert cfg.hf_token is None
    assert cfg.wandb_api_key is None


def test_load_env_reads_dotenv(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "HF_TOKEN=hf_from_file\nWANDB_API_KEY=wandb_from_file\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    cfg = env_mod.load_env(
        env_path,
        wandb_entity="my-entity",
        wandb_project="my-project",
        verbose=False,
    )

    assert cfg.hf_token == "hf_from_file"
    assert cfg.wandb_api_key == "wandb_from_file"
    assert cfg.wandb_entity == "my-entity"
    assert cfg.wandb_project == "my-project"
    assert os.environ["HF_TOKEN"] == "hf_from_file"
    assert os.environ["WANDB_API_KEY"] == "wandb_from_file"


def test_load_env_strips_wandb_env_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_ENTITY", "from-shell")
    monkeypatch.setenv("WANDB_PROJECT", "from-shell")
    monkeypatch.setenv("WANDB_NAME", "from-shell")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    cfg = env_mod.load_env(tmp_path / ".env", verbose=False)

    assert "WANDB_ENTITY" not in os.environ
    assert "WANDB_PROJECT" not in os.environ
    assert "WANDB_NAME" not in os.environ
    assert cfg.wandb_entity is None
    assert cfg.wandb_project is None


def test_load_env_without_kwargs_leaves_wandb_identity_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    cfg = env_mod.load_env(tmp_path / ".env", verbose=False)

    assert cfg.hf_token is None
    assert cfg.wandb_api_key is None
    assert cfg.wandb_entity is None
    assert cfg.wandb_project is None
