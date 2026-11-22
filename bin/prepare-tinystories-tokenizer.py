#!/usr/bin/env python
"""Train the shared byte-level BPE tokenizer on the pinned TinyStories prefix."""

from collections.abc import Iterable
from contextlib import closing
from itertools import islice
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from alien_ink.hf.ds import stream_hub_text, tinystories
from alien_ink.zdeck.pre_tinystories_1m_5k_mist import MANIFEST

VOCAB_SIZE = 6_144
STORY_COUNT = 100_000


def train_tokenizer(texts: Iterable[str], *, vocab_size: int = VOCAB_SIZE):
    """Train a byte-complete BPE tokenizer with three distinct special tokens."""
    bos, eos, pad = "<|bos|>", "<|eos|>", "<|pad|>"
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
            special_tokens=[bos, eos, pad],
            show_progress=False,
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=bos,
        eos_token=eos,
        pad_token=pad,
        model_max_length=MANIFEST.model.n_positions,
        model_input_names=["input_ids", "attention_mask"],
    )


def main() -> None:
    source = tinystories().source
    with closing(iter(stream_hub_text(source))) as rows:
        texts = [row[source.text_column] for row in islice(rows, STORY_COUNT)]
    if len(texts) != STORY_COUNT:
        raise RuntimeError(f"Expected {STORY_COUNT:,} stories, got {len(texts):,}")
    tokenizer = train_tokenizer(texts)
    if len(tokenizer) != VOCAB_SIZE:
        raise RuntimeError(f"Expected {VOCAB_SIZE:,} tokens, got {len(tokenizer):,}")
    output = Path(MANIFEST.model.tokenizer_name)
    tokenizer.save_pretrained(output)
    print(f"Saved {len(tokenizer):,} tokens to {output}")


if __name__ == "__main__":
    main()
