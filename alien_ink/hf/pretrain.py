"""High-level GPT-2 causal-LM pretraining recipes on Hugging Face datasets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from transformers import set_seed

from alien_ink.device import device_info
from alien_ink.env import load_env
from alien_ink.hf.ds import PretrainDataConfig, load_streaming_train_eval
from alien_ink.hf.model import Gpt2ArchConfig, build_model_and_tokenizer
from alien_ink.hf.tok import tokenize_and_chunk
from alien_ink.hf.trainer import (
    CausalLmTrainerConfig,
    build_causal_lm_trainer,
    precision_label,
    tokens_per_optimizer_step,
    train_and_save,
)
from alien_ink.wb import build_run_config, set_wandb_dir, wandb_run


@dataclass
class Gpt2PretrainConfig:
    """Composable config for from-scratch GPT-2 pretraining on a Hub text corpus."""

    data: PretrainDataConfig
    arch: Gpt2ArchConfig = field(default_factory=Gpt2ArchConfig)
    trainer: CausalLmTrainerConfig = field(
        default_factory=lambda: CausalLmTrainerConfig(
            output_dir=Path.cwd() / "output" / "gpt2-pretrain",
            run_name="gpt2-pretrain",
        )
    )


def with_trainer(
    config: Gpt2PretrainConfig,
    **trainer_overrides,
) -> Gpt2PretrainConfig:
    """Return ``config`` with selected ``CausalLmTrainerConfig`` fields replaced."""
    return replace(config, trainer=replace(config.trainer, **trainer_overrides))


def with_data(
    config: Gpt2PretrainConfig,
    **data_overrides,
) -> Gpt2PretrainConfig:
    """Return ``config`` with selected ``PretrainDataConfig`` fields replaced."""
    return replace(config, data=replace(config.data, **data_overrides))


def with_arch(
    config: Gpt2PretrainConfig,
    **arch_overrides,
) -> Gpt2PretrainConfig:
    """Return ``config`` with selected ``Gpt2ArchConfig`` fields replaced."""
    return replace(config, arch=replace(config.arch, **arch_overrides))


def prepare_lm_datasets(
    data: PretrainDataConfig,
    tokenizer,
    *,
    verbose: bool = True,
):
    """Stream/tokenize a Hub corpus into train (iterable) and eval (map-style) LM blocks."""
    train_raw, eval_raw = load_streaming_train_eval(data, verbose=verbose)
    if verbose:
        print()
        print(">> Preprocessing datasets (train streams lazily)...")
    train_text_column = data.source.text_column
    eval_text_column = (data.eval_source or data.source).text_column
    train_dataset = tokenize_and_chunk(
        train_raw,
        tokenizer,
        block_size=data.block_size,
        text_column=train_text_column,
        num_proc=data.tokenizer_num_proc,
    )
    eval_dataset = tokenize_and_chunk(
        eval_raw,
        tokenizer,
        block_size=data.block_size,
        text_column=eval_text_column,
        num_proc=data.tokenizer_num_proc,
    )
    if verbose:
        print(f"   eval blocks:  {len(eval_dataset):,}")
    if len(eval_dataset) == 0:
        raise ValueError(
            "No eval blocks produced; increase max_eval_samples or lower block_size."
        )
    return train_dataset, eval_dataset


def log_pretrain_banner(
    *,
    title: str,
    run_label: str,
    config: Gpt2PretrainConfig,
) -> tuple[str, bool, bool]:
    """Print run header / device info; return ``(device, use_fp16, use_bf16)``."""
    print("----------------------------------------------------------------------")
    print(f":: {title}")
    print("----------------------------------------------------------------------")
    print(f">> run: {run_label}")

    device, use_fp16, use_bf16 = device_info(
        prefer_bf16=config.trainer.prefer_bf16,
        prefer_fp16=config.trainer.prefer_fp16,
    )
    tps = tokens_per_optimizer_step(
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        block_size=config.data.block_size,
    )
    print(
        f">> Device: {device} "
        f"(precision: {precision_label(use_fp16=use_fp16, use_bf16=use_bf16)})"
    )
    print(
        f">> Training steps: {config.trainer.max_steps:,} "
        f"(~{tps:,} tokens/step, "
        f"~{config.trainer.max_steps * tps:,} tokens total)"
    )
    if config.trainer.gradient_checkpointing:
        print(">> Gradient checkpointing enabled")
    return device, use_fp16, use_bf16


def pretrain_gpt2(
    config: Gpt2PretrainConfig,
    *,
    run_label: str = "regular",
    title: str = "GPT-2 from scratch",
    env_files: tuple[Path, ...] | None = None,
    wandb_project_fallback: str = "alien-ink",
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> None:
    """End-to-end: env → data → model → trainer → W&B train/save.

    Pass ``wandb_project`` / ``wandb_name`` explicitly (CLI flags or kwargs) to
    override code defaults. These are not read from the environment.
    """
    env_files = env_files if env_files is not None else (Path.cwd() / ".env",)

    log_pretrain_banner(title=title, run_label=run_label, config=config)

    print()
    print(">> Loading .env...")
    env = load_env(
        *env_files,
        wandb_project=wandb_project,
        wandb_project_fallback=wandb_project_fallback,
    )

    if wandb_name and wandb_name != config.trainer.run_name:
        config = with_trainer(config, run_name=wandb_name)

    set_seed(config.trainer.seed)
    set_wandb_dir(config.trainer.output_dir)
    model, tokenizer = build_model_and_tokenizer(config.arch)
    train_dataset, eval_dataset = prepare_lm_datasets(config.data, tokenizer)

    trainer = build_causal_lm_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config.trainer,
    )
    run_config = build_run_config(
        run_label=run_label,
        env=env,
        configs={"data": config.data, "arch": config.arch, "trainer": config.trainer},
        prefer_bf16=config.trainer.prefer_bf16,
        prefer_fp16=config.trainer.prefer_fp16,
    )

    print()
    with wandb_run(
        project=env.wandb_project,
        name=config.trainer.run_name,
        config=run_config,
        dir=config.trainer.output_dir,
    ):
        train_and_save(
            trainer=trainer,
            tokenizer=tokenizer,
            output_dir=config.trainer.output_dir,
        )
