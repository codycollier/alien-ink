"""Full-parameter supervised fine-tuning of pretrained causal LMs.

First fine-tuning stage from the model learning plan (Track B0): ordinary
full-parameter training of an off-the-shelf Hugging Face checkpoint (or a
local Alien Ink output dir) on packed causal-LM text blocks. No adapters, no
quantization — optimizer state and gradient flow stay visible and debuggable.

The data path is shared with pretraining (:func:`prepare_lm_datasets`); only
model construction differs: :func:`~alien_ink.hf.model.load_hub_model_and_tokenizer`
loads weights generically via ``AutoModelForCausalLM`` instead of initializing
from scratch. Prefer defining a :class:`~alien_ink.hf.manifest.Manifest` with
``stage="sft"`` in zdeck programs; this module is the runtime it materializes
into.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from transformers import set_seed

from alien_ink.com.device import collect_accelerator_info, introspect
from alien_ink.com.env import load_env
from alien_ink.com.log import banner, blank, detail, get_logger, header, step
from alien_ink.com.wb import build_run_config, require_wandb_identity, set_wandb_dir, wandb_run
from alien_ink.hf.ds import PretrainDataConfig
from alien_ink.hf.metrics import (
    build_run_config_payload,
    collect_software_versions,
    count_model_params,
    save_run_config,
)
from alien_ink.hf.model import (
    PretrainedLmConfig,
    load_hub_model_and_tokenizer,
    model_max_positions,
)
from alien_ink.hf.pretrain import prepare_lm_datasets, resolve_use_wandb
from alien_ink.hf.trainer import (
    CausalLmTrainerConfig,
    build_causal_lm_trainer,
    reporting_disabled,
    tokens_per_optimizer_step,
    train_and_save,
)

log = get_logger("hf.sft")

__all__ = ["SftConfig", "finetune"]


@dataclass(frozen=True)
class SftConfig:
    """Composable config for full-parameter fine-tuning of a pretrained LM.

    ``data`` is a single Hub text corpus packed into causal-LM blocks exactly
    like pretraining; ``model`` names the pretrained checkpoint to start from.
    """

    data: PretrainDataConfig
    model: PretrainedLmConfig = field(
        default_factory=lambda: PretrainedLmConfig(model_name="EleutherAI/pythia-160m")
    )
    trainer: CausalLmTrainerConfig = field(
        default_factory=lambda: CausalLmTrainerConfig(
            output_dir=Path.cwd() / "output" / "sft",
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
    config: SftConfig,
):
    """Log stars header, device introspection, then run summary.

    Returns ``(AcceleratorInfo, tokens_per_step)``.
    """
    header(logger=log)

    accel = collect_accelerator_info(
        prefer_bf16=config.trainer.prefer_bf16,
        prefer_fp16=config.trainer.prefer_fp16,
    )
    for line in introspect(info=accel).splitlines():
        log.info(line)
    blank(logger=log)

    banner(title, logger=log)
    step(f"run: {run_label}", logger=log)

    tps = tokens_per_optimizer_step(
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        block_size=config.data.block_size,
        world_size=accel.world_size,
    )
    detail(f"base model: {config.model.model_name}", logger=log)
    if config.trainer.uses_epochs():
        step(
            f"Fine-tuning epochs: {config.trainer.num_train_epochs:g} "
            f"(~{tps:,} tokens/step)",
            logger=log,
        )
    else:
        step(
            f"Fine-tuning steps: {config.trainer.max_steps:,} "
            f"(~{tps:,} tokens/step, "
            f"~{config.trainer.max_steps * tps:,} tokens total)",
            logger=log,
        )
    if config.trainer.gradient_checkpointing:
        step("Gradient checkpointing enabled", logger=log)
    if config.trainer.torch_compile:
        step("torch.compile enabled", logger=log)
    if config.trainer.tf32 is True:
        step("TF32 enabled", logger=log)
    if config.trainer.optim != "adamw_torch":
        detail(f"optimizer: {config.trainer.optim}", logger=log)
    return accel, tps


def finetune(
    config: SftConfig,
    *,
    run_label: str = "regular",
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
    env_files = env_files if env_files is not None else (Path.cwd() / ".env",)

    if resume_from_checkpoint is not None:
        config = replace(
            config,
            trainer=replace(
                config.trainer, resume_from_checkpoint=resume_from_checkpoint
            ),
        )

    want_wandb = resolve_use_wandb(config, use_wandb)
    if not want_wandb and not reporting_disabled(config.trainer.report_to):
        config = replace(config, trainer=replace(config.trainer, report_to="none"))

    if want_wandb:
        require_wandb_identity(entity=wandb_entity, project=wandb_project)

    config.validate()
    accel, tokens_per_step = log_sft_banner(
        title=title, run_label=run_label, config=config
    )

    blank(logger=log)
    step("Loading credentials...", logger=log)
    env = load_env(
        *env_files,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )

    if wandb_name and wandb_name != config.trainer.run_name:
        config = replace(config, trainer=replace(config.trainer, run_name=wandb_name))

    set_seed(config.trainer.seed)
    if want_wandb:
        set_wandb_dir(config.trainer.output_dir)

    model, tokenizer = load_hub_model_and_tokenizer(config.model)
    max_positions = model_max_positions(model)
    if max_positions and config.data.block_size > max_positions:
        raise ValueError(
            f"data.block_size ({config.data.block_size}) cannot exceed the "
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

    train_dataset, eval_dataset = prepare_lm_datasets(config.data, tokenizer)

    log_configs: dict[str, Any] = {
        "data": config.data,
        "model": config.model,
        "trainer": config.trainer,
    }
    if extra_configs:
        log_configs.update(dict(extra_configs))

    config_payload = build_run_config_payload(
        run_label=run_label,
        run_name=config.trainer.run_name,
        title=title,
        configs=log_configs,
        accelerator=accel,
        model_size=model_size,
        tokens_per_step=tokens_per_step,
    )
    blank(logger=log)
    step("Writing run config...", logger=log)
    save_run_config(config.trainer.output_dir, config_payload)

    trainer = build_causal_lm_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config.trainer,
    )
    software = collect_software_versions()
    run_config = build_run_config(
        run_label=run_label,
        env=env,
        configs=log_configs,
        prefer_bf16=config.trainer.prefer_bf16,
        prefer_fp16=config.trainer.prefer_fp16,
        accelerator=accel,
        tokens_per_step=tokens_per_step,
        model=model_size.as_dict(),
        software=software,
    )

    blank(logger=log)
    with wandb_run(
        entity=env.wandb_entity or "",
        project=env.wandb_project or "",
        name=config.trainer.run_name,
        config=run_config,
        dir=config.trainer.output_dir,
        enabled=want_wandb,
    ):
        return train_and_save(
            trainer=trainer,
            tokenizer=tokenizer,
            output_dir=config.trainer.output_dir,
            resume_from_checkpoint=config.trainer.resume_from_checkpoint,
            run_name=config.trainer.run_name,
            run_label=run_label,
            max_steps=config.trainer.max_steps,
            tokens_per_step=tokens_per_step,
            model_size=model_size,
            accelerator=accel,
        )
