# Model families

Alien Ink trains **from-scratch** causal language models. Architecture and
tokenizer identity live in `CausalLmArchConfig` (`alien_ink.hf.model`). Field
names follow the GPT-2 convention (`n_embd`, `n_layer`, …) and are mapped onto
each Hugging Face config class.

Supported families: **`gpt-2`**, **`gpt-neox`**, **`gemma`**. Defaults are sized
for Mist (local RTX 3070, ~8 GB VRAM).

---

## At a glance

| | GPT-2 | GPT-NeoX | Gemma (Mist) |
|---|---|---|---|
| **`family` string** | `gpt-2` | `gpt-neox` | `gemma` |
| **HF class** | `GPT2LMHeadModel` | `GPTNeoXForCausalLM` | `GemmaForCausalLM` |
| **Factory** | `gpt2_arch()` | `gpt_neox_arch()` | `gemma_arch()` |
| **Default tokenizer** | `gpt2` | `EleutherAI/gpt-neox-20b` | `google/gemma-2b` |
| **Tokenizer type** | Byte-level BPE | BPE (NeoX) | SentencePiece |
| **Vocab size** | ~50,257 | ~50,432 | ~256,000 |
| **Mist layers / width** | 12 / 768 | 12 / 768 | 8 / 512 |
| **Mist heads** | 12 | 12 | 8 (`head_dim=64`) |
| **Context (`n_positions`)** | 1024 | 1024 | 1024 |
| **MLP width** | ~4× embd (HF default) | `4 * n_embd` | 2048 |
| **Params (Mist, approx.)** | **~124M** | **~125M** | **~280–300M** |
| **Zdeck programs** | WikiText, Wikipedia | none yet | C4 5k / 50k |

Parameter counts depend on exact vocab after `load_tokenizer` (pad token may
extend the table). GPT-2 Mist matches the classic “GPT-2 small” scale; Mist
Gemma is **not** full Gemma-2B — it is a small Gemma *architecture* that reuses
the Gemma-2B tokenizer.

---

## Shared training semantics

All families:

- Initialize with **random weights** (`build_model_from_scratch`) — Hub model
  ids are for tokenizer / config shape, not pretrained checkpoints.
- Train with packed causal-LM blocks (`labels = input_ids`).
- Use `use_cache=False` during training (compatible with gradient checkpointing).
- Must satisfy `data.block_size ≤ model.n_positions` (zdecks use 1024 / 1024).

Generation defaults differ by family (`alien_ink.hf.gen.gen_config_for_family`).

---

## GPT-2 (`family="gpt-2"`)

### Role in Alien Ink

Default, well-understood baseline. All GPT-2 zdecks target Mist with a
124M-class stack identical in width/depth to Hugging Face `gpt2` (small).

### Mist architecture

| Field | Value |
|---|---:|
| `n_positions` | 1024 |
| `n_embd` | 768 |
| `n_layer` | 12 |
| `n_head` | 12 |
| `head_dim` | `None` (implied `n_embd // n_head` = 64) |
| `intermediate_size` | `None` (HF GPT-2 MLP default) |

Approx. **124M** parameters with the `gpt2` vocab.

### Tokenizer

- Name: `gpt2`
- Byte-level BPE, vocab **50,257**
- No dedicated BOS; EOS / pad handled via `load_tokenizer` (pad ← eos if missing)
- Training and completion use ordinary plain text

### Hardware knobs (zdeck)

| Setting | Value |
|---|---:|
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 16 |
| Effective batch (examples) | 32 |
| Tokens / step @ 1024 | 32,768 |

Fits comfortably on 8 GB with bf16/fp16 + gradient checkpointing.

### Generation

| Setting | Default |
|---|---|
| `add_special_tokens` | `True` |
| `do_sample` | `False` (greedy) |
| `stop_strings` | `.` `!` `?` |

### Typical pairings

| Corpus | Zdeck |
|---|---|
| WikiText-103 | `gpt-2_wikitext_5k` |
| English Wikipedia | `gpt-2_wikipedia_5k` |

C4 works with GPT-2 via the same data loaders; there is simply no checked-in
GPT-2×C4 zdeck yet.

### Characteristics

- **Pros:** Small vocab → cheap logits; strong tooling; easy to reason about;
  good fit for clean wiki text.
- **Cons:** Older architecture (absolute learned positions, no RoPE/GQA);
  weaker inductive bias than Gemma/NeoX-style stacks at the same width.
- **Memory:** Dominated by activations and attention at 1024 context, not vocab.

---

## GPT-NeoX (`family="gpt-neox"`)

### Role in Alien Ink

Supported end-to-end (build, train, load, generate) but **no zdeck program**
yet. Same Mist-oriented dims as GPT-2 for a fair architecture comparison.

### Mist architecture

| Field | Value |
|---|---:|
| `n_positions` | 1024 |
| `n_embd` | 768 |
| `n_layer` | 12 |
| `n_head` | 12 |
| `intermediate_size` | `4 * n_embd` (3072) unless overridden |

Approx. **~125M** parameters with the NeoX tokenizer vocab.

Mapped HF fields: `max_position_embeddings`, `hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `intermediate_size`.

### Tokenizer

- Default: `EleutherAI/gpt-neox-20b`
- BPE, vocab **~50,432**
- Behavior for completion matches GPT-2 defaults (`add_special_tokens=True`)

### Characteristics

- **Pros:** Rotary / NeoX-style stack closer to modern pretraining recipes;
  easy drop-in swap from GPT-2 via `family` + tokenizer.
- **Cons:** Untested in this repo’s zdeck archive; validate VRAM and loss curves
  before long runs.
- **How to add a zdeck:** copy `gpt-2_wikitext_5k`, set `family="gpt-neox"`,
  `tokenizer_name="EleutherAI/gpt-neox-20b"`, keep Mist dims unless retuning.

---

## Gemma (`family="gemma"`)

### Role in Alien Ink

Second family with checked-in Mist runs. Uses a **downsized** Gemma config so
training fits on 8 GB while keeping the real Gemma-2B **tokenizer** (and thus
the large vocabulary).

### Mist architecture (not full Gemma-2B)

| Field | Mist value | Full Gemma-2B (reference) |
|---|---:|---:|
| `n_positions` | 1024 | 8192 |
| `n_embd` | 512 | 2048 |
| `n_layer` | 8 | 18 |
| `n_head` | 8 | 8 |
| `head_dim` | 64 | 256 |
| `intermediate_size` | 2048 | 16384 |
| Params | **~280–300M** | ~2B |

`num_key_value_heads` is set equal to `n_head` in code so SDPA does not break
(GemmaConfig’s default KV-head count would otherwise mismatch).

### Tokenizer

- Name: `google/gemma-2b` (requires Hugging Face access / `HF_TOKEN`)
- SentencePiece, vocab **~256k**
- Defines `<bos>`, `<eos>`, and pad
- During training, each Hub row is tokenized with the tokenizer’s defaults
  (BOS may appear at row starts; packed blocks are mostly mid-stream)

### Hardware knobs (zdeck)

| Setting | Value | Why |
|---|---:|---|
| `per_device_train_batch_size` | 1 | Vocab ~256k → large logit tensor |
| `gradient_accumulation_steps` | 32 | Keep effective batch 32 |
| Tokens / step @ 1024 | 32,768 | Same as GPT-2 Mist |

Batch size 2 tends to OOM on 8 GB: logits alone are on the order of
`batch × seq × vocab × 2 bytes` ≈ **~2 GiB** at batch 2 / 1024 / 256k in fp16.

### Generation

| Setting | Default |
|---|---|
| `add_special_tokens` | **`False`** |
| `do_sample` | `False` |
| `stop_strings` | `.` `!` `?` |

`add_special_tokens=False` avoids prepending BOS on mid-document continuation,
matching how packed training mostly looks.

### Typical pairings

| Corpus | Zdeck |
|---|---|
| C4 English | `gemma_c4_5k` (5k steps) |
| C4 English | `gemma_c4_50k` (50k steps) |

WikiText / Wikipedia are supported by the stack; C4 is the archived pairing.

### Characteristics

- **Pros:** Modern Gemma block (RoPE, gated MLP); large SentencePiece vocab
  handles multilingual / rare spellings better than GPT-2 BPE; good match for
  diverse web text.
- **Cons:** Embedding + LM head dominate parameter count at Mist width; VRAM
  sensitive to vocab; needs Hub auth for tokenizer; Mist config is **not**
  comparable to published Gemma-2B quality.
- **Memory:** Embedding tables (~256k × 512 × 2 for untied embed/lm_head) are
  large relative to the tiny trunk — expect ~300M params even with only 8
  layers.

---

## Parameter and VRAM intuition

Rough Mist footprints (order of magnitude, training with checkpointing):

| Family | Params | Embedding pressure | Typical train batch |
|---|---:|---|---:|
| GPT-2 | ~124M | Low (~50k vocab) | 2 × accum 16 |
| GPT-NeoX | ~125M | Low (~50k vocab) | start like GPT-2 |
| Gemma Mist | ~290M | High (~256k vocab) | 1 × accum 32 |

For any family, increasing `n_positions` or batch size costs activation memory
quadratically / linearly; increasing vocab costs logit and embedding memory
linearly in vocab size.

---

## Family × dataset matrix

| | WikiText-103 | Wikipedia EN | C4 EN |
|---|---|---|---|
| **GPT-2** | ✓ zdeck | ✓ zdeck | supported |
| **Gemma** | supported | supported | ✓ zdeck |
| **GPT-NeoX** | supported | supported | supported |

“Supported” means loaders + `build_model_from_scratch` + generation paths work.
Only **zdeck** cells have a checked-in manifest.

---

## Configuring a family in a manifest

```python
from alien_ink.hf.model import CausalLmArchConfig, gemma_arch, gpt2_arch

# Explicit (zdeck style)
model = CausalLmArchConfig(
    family="gpt-2",
    tokenizer_name="gpt2",
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    head_dim=None,
    intermediate_size=None,
    use_cache=False,
)

# Or factory + overrides
model = gemma_arch(n_layer=10)
model = gpt2_arch(n_embd=512, n_layer=8, n_head=8)
```

Load a trained checkpoint with the same `family` string:

```python
load_pretrained_model(path, device, family="gemma")
```

Completions:

```python
gen = manifest.gen_config(max_new_tokens=120)
# or: gen_config_for_family("gemma", max_new_tokens=120)
```

---

## Extending with a new family

1. Add the literal to `ModelFamily` and validate in `CausalLmArchConfig`.
2. Branch in `build_model_from_scratch` and `load_pretrained_model`.
3. Add Mist defaults + an `*_arch()` factory.
4. Register generation defaults in `_GEN_CONFIG_BY_FAMILY`.
5. Add a zdeck module with every field spelled out.

Keep functions thin and composable — prefer a factory + manifest over a new
class hierarchy.
