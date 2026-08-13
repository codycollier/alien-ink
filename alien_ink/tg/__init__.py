"""Tinygrad backend for Alien Ink.

GPT-2 only. Architecture knobs come from
:class:`~alien_ink.hf.model.CausalLmArchConfig`; data packing and the
tokenizer stay on the Hugging Face path. Training does not use
``transformers.Trainer``.
"""
