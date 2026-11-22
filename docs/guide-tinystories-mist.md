# TinyStories on Mist

The three GPT-Neo models train from scratch on the official TinyStories train
and validation splits from the same pinned revision. All three share a
6,144-token byte-level BPE, 512-token blocks, and a 40,960,000-token budget on
one GPU. The packer appends EOS to each story and packs across story boundaries.
Validation uses 1,000 stories selected with seed 101.
[Pinned TinyStories splits](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/data)

| Run size | Trainable parameters |
|---|---:|
| 1M | 982,800 |
| 3M | 2,960,928 |
| 33M | 33,455,616 |

## Prepare the tokenizer and check GPU fit

From the repository root with the project environment active:

```sh
python bin/prepare-tinystories-tokenizer.py
python bin/smoke-tinystories-mist.py
```

Preparation streams the first 100,000 training stories in dataset order and
saves the tokenizer in `output/tokenizers/tinystories-bpe-6144`. The smoke run
trains the 33M model for ten steps, evaluates on 64 validation stories, reloads
checkpoint 10, and checks for finite losses and peak CUDA reserved memory below
7 GiB. The batch-4 smoke run on Mist reached 1.61 GiB, with train loss
6.904402 and validation loss 6.232171.

If batch 4 exceeds the memory limit or runs out of memory, repeat the smoke
check with `--batch-size 2`. If it passes, set
`per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, and
`per_device_eval_batch_size=2` in the 33M manifest. Both configurations process
16 blocks per optimizer step and retain the same training token budget.

## Train

```sh
python -m alien_ink.zdeck.pre_tinystories_1m_5k_mist
python -m alien_ink.zdeck.pre_tinystories_3m_5k_mist
python -m alien_ink.zdeck.pre_tinystories_33m_5k_mist
```

Each run evaluates and saves every 250 optimizer steps under
`output/train/pre-tinystories-<size>-5k-mist/`. Checkpoints include the tokenizer.

## Train the 1M model for one full epoch

```sh
python -m alien_ink.zdeck.pre_tinystories_1m_1ep_mist
# Or launch it in the background with a timestamped log.
./bin/mist-train-pretrain-general.sh --tinystories-1m-1ep
```

This deck downloads and packs the full pinned training split before training.
It uses the packed block count to determine the epoch length and checkpoint
cadence. The validation sample comes from the official validation split with
the same seed and row limit as the 5,000-step run. The complete and streaming
loaders use different shuffle algorithms, so the sampled rows can differ.
Outputs go to `output/train/pre-tinystories-1m-1ep-mist/`. Use
`pre_tinystories_1m_1ep_mist` as the deck name with the chat or evaluation helper.

## Compare the runs

In each run's latest numbered checkpoint, read `trainer_state.json` and compare
`eval_loss` in `log_history` at matching steps. Step 5,000 gives the comparison
at the full token budget. The final model saved at the run root uses the weights
from the checkpoint with the lowest validation loss.

```sh
python bin/mist-chat-with-model.py pre_tinystories_1m_5k_mist --max-new-tokens 120
python bin/mist-chat-with-model.py pre_tinystories_3m_5k_mist --max-new-tokens 120
python bin/mist-chat-with-model.py pre_tinystories_33m_5k_mist --max-new-tokens 120
```

Enter the same story starters in each session. Compare the greedy candidate for
each starter, then look at the sampled continuations for variety.

- Once upon a time, a little fox found a key under a leaf.
- Mia wanted to bake a cake for her friend, but the flour was gone.
- On the way home, Ben heard a tiny voice coming from a box.
