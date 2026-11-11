"""Full-parameter supervised fine-tuning of pretrained causal LMs.

Full-parameter training of an off-the-shelf Hugging Face checkpoint (or a
local Alien Ink output dir). SFT accepts either packed Hub text or supervised
prompt/completion JSON with an explicit prompt-loss mask. No adapters or
quantization: optimizer state and gradient flow stay visible and debuggable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from alien_ink.com.log import detail, get_logger
from alien_ink.hf.completion import CompletionDataConfig, prepare_completion_datasets
from alien_ink.hf.ds import PretrainDataConfig, prepare_lm_datasets
from alien_ink.hf.metrics import count_model_params
from alien_ink.hf.model import (
    PretrainedLmConfig,
    load_hub_model_and_tokenizer,
    model_max_positions,
)
from alien_ink.hf.runtime import (
    initialize_run,
    log_training_banner,
    resolve_trainer_config,
    run_training,
)
from alien_ink.hf.trainer import CausalLmTrainerConfig

log = get_logger("hf.sft")

__all__ = ["SftConfig", "finetune"]


@dataclass(frozen=True)
class SftConfig:
    """Composable config for full-parameter fine-tuning of a pretrained LM.

    ``data`` is either packed Hub text or supervised prompt/completion JSON;
    ``model`` names the pretrained checkpoint to start from.
    """

    data: PretrainDataConfig | CompletionDataConfig
    model: PretrainedLmConfig = field(
        default_factory=lambda: PretrainedLmConfig(model_name="EleutherAI/pythia-160m")
    )
    trainer: CausalLmTrainerConfig = field(
        default_factory=lambda: CausalLmTrainerConfig(
            output_dir=Path.cwd() / "output" / "train" / "sft",
            run_name="sft",
        )
    )

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.trainer.validate()


def log_sft_banner(
    *,
    title: str,
    run_label: str,
    zdeck_name: str | None,
    config: SftConfig,
):
    """Log stars header, device introspection, then run summary.

    Returns ``(AcceleratorInfo, tokens_per_step)``.
    """
    return log_training_banner(
        title=title,
        run_label=run_label,
        zdeck_name=zdeck_name,
        trainer=config.trainer,
        block_size=config.data.max_length
        if isinstance(config.data, CompletionDataConfig)
        else config.data.block_size,
        model_label=f"base model: {config.model.model_name}",
        action="Fine-tuning",
    )


def finetune(
    config: SftConfig,
    *,
    run_label: str = "regular",
    zdeck_name: str | None = None,
    title: str = "Causal LM fine-tune",
    env_files: tuple[Path, ...] | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    use_wandb: bool | None = None,
    resume_from_checkpoint: str | Path | bool | None = None,
    extra_configs: Mapping[str, Any] | None = None,
):
    """End-to-end SFT: env → pretrained model → data → trainer → train/save.

    Mirrors :func:`~alien_ink.hf.pretrain.pretrain` semantics: returns
    ``(trainer, run_summary)``, requires explicit W&B identity when enabled,
    and always writes ``run_config.json`` / ``run_summary.json`` under
    ``trainer.output_dir``.
    """
    trainer_config, want_wandb = resolve_trainer_config(
        config.trainer,
        use_wandb=use_wandb,
        wandb_name=wandb_name,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    config = replace(config, trainer=trainer_config)
    config.validate()
    accel, tokens_per_step = log_sft_banner(
        title=title, run_label=run_label, zdeck_name=zdeck_name, config=config
    )

    env = initialize_run(
        config.trainer,
        env_files=env_files,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        enabled=want_wandb,
    )
    model, tokenizer = load_hub_model_and_tokenizer(config.model)
    max_positions = model_max_positions(model)
    data_max_length = (
        config.data.max_length
        if isinstance(config.data, CompletionDataConfig)
        else config.data.block_size
    )
    if max_positions and data_max_length > max_positions:
        raise ValueError(
            f"data sequence length ({data_max_length}) cannot exceed the "
            f"pretrained model's context window ({max_positions})"
        )
    model_size = count_model_params(model, vocab_size=tokenizer.vocab_size)
    trainable_pct = (
        100.0 * model_size.trainable_params / model_size.total_params
        if model_size.total_params
        else 0.0
    )
    detail(
        f"trainable params: {model_size.trainable_params:,} / "
        f"{model_size.total_params:,} ({trainable_pct:.1f}%)",
        logger=log,
    )

    if isinstance(config.data, CompletionDataConfig):
        train_dataset, eval_dataset = prepare_completion_datasets(
            config.data, tokenizer
        )
    else:
        train_dataset, eval_dataset = prepare_lm_datasets(config.data, tokenizer)

    log_configs: dict[str, Any] = {
        "data": config.data,
        "model": config.model,
        "trainer": config.trainer,
    }
    if extra_configs:
        log_configs.update(dict(extra_configs))

    return run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config.trainer,
        log_configs=log_configs,
        run_label=run_label,
        title=title,
        env=env,
        accel=accel,
        tokens_per_step=tokens_per_step,
        model_size=model_size,
        want_wandb=want_wandb,
    )
