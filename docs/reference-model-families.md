# Model families

Alien Ink trains **from-scratch** causal language models. Architecture and
tokenizer identity live in `CausalLmArchConfig` (`alien_ink.hf.model`). Field
names follow the GPT-2 convention (`n_embd`, `n_layer`, …) and are mapped onto
each Hugging Face config class.

Supported families: **`gpt-2`**, **`gpt-neox`**, **`pythia`**, **`gemma`**,
**`llama`**. Defaults are sized for Mist (local RTX 3070, ~8 GB VRAM).

Off-the-shelf **pretrained** checkpoints (for the `sft` stage) are not a
family: they load generically through `PretrainedLmConfig` +
`AutoModelForCausalLM`, so any Hub model with a serialized config works. See
[Fine-tuning pretrained checkpoints](#fine-tuning-pretrained-checkpoints).

---

## At a glance

| | GPT-2 | GPT-NeoX | Pythia-160M | SmolLM2-135M | Gemma (Mist) |
|---|---|---|---|---|---|
| **`family` string** | `gpt-2` | `gpt-neox` | `pythia` | `llama` | `gemma` |
| **HF class** | `GPT2LMHeadModel` | `GPTNeoXForCausalLM` | `GPTNeoXForCausalLM` | `LlamaForCausalLM` | `GemmaForCausalLM` |
| **Factory** | `gpt2_arch()` | `gpt_neox_arch()` | `pythia_160m_arch()` | `smollm2_135m_arch()` | `gemma_arch()` |
| **Default tokenizer** | `gpt2` | `EleutherAI/gpt-neox-20b` | `EleutherAI/pythia-160m` | `HuggingFaceTB/SmolLM2-135M` | `google/gemma-2b` |
| **Tokenizer type** | Byte-level BPE | BPE (NeoX) | BPE (NeoX) | BPE | SentencePiece |
| **Vocab size** | ~50,257 | ~50,432 | ~50k | ~49k | ~256,000 |
| **Layers / width** | 12 / 768 | 12 / 768 | 12 / 768 | 30 / 576 | 8 / 512 |
| **Heads** | 12 | 12 | 12 | 9 (GQA, 3 KV) | 8 (`head_dim=64`) |
| **Context (`n_positions`)** | 1024 | 1024 | 2048 | 2048 (Mist cap) | 1024 |
| **MLP width** | ~4× embd (HF default) | `4 * n_embd` | 3072 | 1536 | 2048 |
| **Params (approx.)** | **~124M** | **~163M** | **~162M** | **~135M** | **~165M** |
| **Batch × accum (zdeck)** | 4 × 8 | 4 × 8 | 4 × 8 | 2 × 16 | 1 × 32 |
| **Zdeck programs** | WikiText, Wikipedia | WikiText baseline / 3 ep / 4 ep; C4 5k; curriculum | WikiText 4 ep (70M and 160M) | WikiText 4 ep | WikiText baseline / 4 ep; C4 5k / 50k |

Parameter counts depend on the exact tokenizer length after `load_tokenizer`.
They include the LM head: GPT-2 and Gemma tie it to the input embedding, while
the current GPT-NeoX config leaves it untied. GPT-2 Mist matches the classic
“GPT-2 small” scale; Mist Gemma is **not** full Gemma-2B — it is a small Gemma
*architecture* that reuses the Gemma-2B tokenizer.

---

## Shared training semantics

All families:

- Initialize with **random weights** (`build_model_from_scratch`) — Hub model
  ids are for tokenizer / config shape, not pretrained checkpoints.
- Train with EOS-delimited packed causal-LM blocks (`labels = input_ids`);
  tokenizer-added special tokens are disabled during preprocessing.
- Use `use_cache=False` during training (compatible with gradient checkpointing).
- Must satisfy `data.block_size ≤ model.n_positions` (zdecks use 1024 / 1024).
- Set activation, dropout, normalization, initialization, RoPE, embedding tying,
  KV heads, attention implementation, and special-token IDs explicitly.

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
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 8 |
| Effective batch (examples) | 32 |
| Tokens / step @ 1024 | 32,768 |

Fits comfortably on 8 GB with bf16/fp16 and gradient checkpointing **off**
(the small vocab leaves headroom); turn checkpointing back on if a variant
OOMs.

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

Supported end-to-end (build, train, load, generate), with WikiText baseline,
3-epoch, and 4-epoch programs, a C4 5k run, and the C4 → geo-us-states
curriculum zdeck. Same Mist-oriented dims as GPT-2 for a fair architecture
comparison.

### Mist architecture

| Field | Value |
|---|---:|
| `n_positions` | 1024 |
| `n_embd` | 768 |
| `n_layer` | 12 |
| `n_head` | 12 |
| `intermediate_size` | `4 * n_embd` (3072) unless overridden |

Approx. **~163M** parameters with the NeoX tokenizer vocab. GPT-NeoX does not
tie its LM head by default, so its ~39M-token embedding matrix is present once
for input and once for output.

Mapped HF fields: `max_position_embeddings`, `hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `intermediate_size`.

### Tokenizer

- Default: `EleutherAI/gpt-neox-20b`
- BPE, vocab **~50,432**
- Behavior for completion matches GPT-2 defaults (`add_special_tokens=True`)

### Characteristics

- **Pros:** Rotary / NeoX-style stack closer to modern pretraining recipes;
  easy drop-in swap from GPT-2 via `family` + tokenizer.
- **Cons:** Its untied output head raises static parameter/optimizer memory;
  validate VRAM and loss curves before long runs.

---

## Pythia (`family="pythia"`)

### Role in Alien Ink

Reference suite for education and research: the EleutherAI Pythia models come
in multiple sizes with consistent data ordering and many published
intermediate checkpoints. Pythia **is** the GPT-NeoX architecture (parallel
residual, partial rotary, untied embeddings) at published shapes, so the
family shares the NeoX config mapping and differs only in dimensions and
tokenizer identity.

### Shapes (published EleutherAI configs)

| Field | Pythia-70M | Pythia-160M |
|---|---:|---:|
| `n_positions` | 2048 | 2048 |
| `n_embd` | 512 | 768 |
| `n_layer` | 6 | 12 |
| `n_head` | 8 | 12 |
| `intermediate_size` | 2048 | 3072 |
| Factory | `pythia_70m_arch()` | `pythia_160m_arch()` |

Both use `rotary_pct=0.25`, `rope_theta=10_000`, `hidden_act="gelu"`, and
untied embeddings. Pythia-160M matches the existing NeoX Mist dims (12 × 768)
for a direct comparison; Pythia-70M is the fast-iteration option.

### Tokenizer

- Names: `EleutherAI/pythia-70m` / `EleutherAI/pythia-160m` (the NeoX BPE
  tokenizer, vocab ~50k)
- Completion behavior matches GPT-2 / NeoX defaults (`add_special_tokens=True`)

### Typical pairings

| Corpus | Zdeck |
|---|---|
| WikiText-103 | `pre_pythia-70m_wikitext_4ep_mist` |
| WikiText-103 | `pre_pythia-160m_wikitext_4ep_mist` |

The published `EleutherAI/pythia-160m` checkpoint is also the first SFT base
(`sft_pythia-160m_geo_mist`).

---

## Llama / SmolLM2 (`family="llama"`)

### Role in Alien Ink

Modern compact Llama-style block (RoPE, SwiGLU, RMSNorm, GQA) using the
published `HuggingFaceTB/SmolLM2-135M` shape and tokenizer with **random
weights** — an architecture experiment, not the pretrained checkpoint.

### Shape (SmolLM2-135M)

| Field | Value |
|---|---:|
| `n_positions` | 2048 (Mist cap; SmolLM2 ships 8192) |
| `n_embd` | 576 |
| `n_layer` | 30 |
| `n_head` | 9 (`head_dim=64`) |
| `num_key_value_heads` | 3 (GQA) |
| `intermediate_size` | 1536 |
| Factory | `smollm2_135m_arch()` |

Uses `hidden_act="silu"`, `rope_theta=100_000`, tied embeddings, and
`initializer_range=1/sqrt(576)`. The 30-layer stack carries more activation
memory than the 12-layer baselines: zdecks use microbatch 2 × accum 16.

### Tokenizer

- Name: `HuggingFaceTB/SmolLM2-135M` (BPE, vocab ~49k)
- Completion behavior matches GPT-2 defaults (`add_special_tokens=True`)

### Typical pairings

| Corpus | Zdeck |
|---|---|
| WikiText-103 | `pre_smollm2-135m_wikitext_4ep_mist` |

The published `HuggingFaceTB/SmolLM2-135M` checkpoint is also an SFT base
(`sft_smollm2-135m_geo_mist`).

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
| Params | **~165M** | ~2B |

`num_key_value_heads` is set equal to `n_head` in code so SDPA does not break
(GemmaConfig’s default KV-head count would otherwise mismatch).

### Tokenizer

- Name: `google/gemma-2b` (requires Hugging Face access / `HF_TOKEN`)
- SentencePiece, vocab **~256k**
- Defines `<bos>`, `<eos>`, and pad
- During training, tokenizer-added special tokens are disabled and one EOS is
  appended to each Hub row.

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
| WikiText-103 | `gemma_wikitext_4ep` (complete, 4 epochs) |
| WikiText-103 | `baseline_perf_gemma_mist` (0.25 epochs, perf baseline) |

Wikipedia is supported by the stack; C4 and WikiText are the archived
pairings.

### Characteristics

- **Pros:** Modern Gemma block (RoPE, gated MLP); large SentencePiece vocab
  handles multilingual / rare spellings better than GPT-2 BPE; good match for
  diverse web text.
- **Cons:** The tied embedding/LM-head table still dominates parameter count and
  the 256k-way logits dominate micro-batch memory and output compute; needs Hub
  auth for tokenizer; Mist config is **not** comparable to published Gemma-2B
  quality.
- **Memory:** The tied embedding/LM-head table is ~131M parameters, versus
  ~34M in the transformer trunk. Tying saves static memory, but does not shrink
  the `[batch, sequence, vocabulary]` logit tensor.

---

## Parameter and VRAM intuition

Rough Mist footprints (order of magnitude; GPT-2/NeoX without checkpointing,
Gemma with checkpointing):

| Family | Params | Embedding pressure | Typical train batch |
|---|---:|---|---:|
| GPT-2 | ~124M | Low (~50k vocab) | 4 × accum 8 |
| GPT-NeoX | ~163M | Low (~50k vocab) | 4 × accum 8 |
| Gemma Mist | ~165M | High (~256k vocab) | 1 × accum 32 |

For any family, increasing `n_positions` or batch size costs activation memory
quadratically / linearly; increasing vocab costs logit and embedding memory
linearly in vocab size.

---

## Family × dataset matrix

| | WikiText-103 | Wikipedia EN | C4 EN |
|---|---|---|---|
| **GPT-2** | ✓ zdeck | ✓ zdeck | supported |
| **Gemma** | ✓ zdeck | supported | ✓ zdeck |
| **GPT-NeoX** | ✓ zdeck | supported | ✓ zdeck |
| **Pythia** | ✓ zdeck | supported | supported |
| **Llama (SmolLM2)** | ✓ zdeck | supported | supported |

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
    hidden_act="gelu_new",
    hidden_dropout=0.1,
    attention_dropout=0.1,
    norm_epsilon=1e-5,
    initializer_range=0.02,
    rope_theta=None,
    rotary_pct=None,
    tie_word_embeddings=True,
    num_key_value_heads=None,
    attention_implementation="sdpa",
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

## Fine-tuning pretrained checkpoints

The `sft` manifest stage does **full-parameter** fine-tuning of an
off-the-shelf pretrained model (learning-plan Track B0). The model is not a
`CausalLmArchConfig` family — it is a `PretrainedLmConfig` loaded through
`AutoModelForCausalLM`, so any Hub model or local Alien Ink checkpoint with a
serialized config works without a per-family branch:

```python
from alien_ink.hf.model import PretrainedLmConfig

model = PretrainedLmConfig(
    model_name="EleutherAI/pythia-160m",   # Hub id or local output/train/<run> path
    tokenizer_name=None,                    # None => ships with the model
    attention_implementation="sdpa",
    use_cache=False,
    trust_remote_code=False,
)
manifest = Manifest(..., stage="sft", model=model, ...)
manifest.train()
```

Data preparation is shared with pretraining (packed EOS-delimited causal-LM
blocks); `block_size` is checked against the loaded model's context window at
train time. Checked-in SFT zdecks: `sft_pythia-160m_geo_mist`,
`sft_smollm2-135m_geo_mist`.

---

## Extending with a new family

1. Add the literal to `ModelFamily` and validate in `CausalLmArchConfig`.
2. Branch in `build_model_from_scratch` and `load_pretrained_model`.
3. Add Mist defaults + an `*_arch()` factory.
4. Register generation defaults in `_GEN_CONFIG_BY_FAMILY`.
5. Add a zdeck module with every field spelled out.

Keep functions thin and composable — prefer a factory + manifest over a new
class hierarchy.
