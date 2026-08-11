# Completions and evals

How to talk to a trained checkpoint and how to score it. Alien Ink
checkpoints are **base LMs**, not instruction-tuned chat models: prompt them
with a sentence starter or fragment and they continue in plain text. Two
tools sit on top of that — an interactive completion REPL for spot-reading a
model, and a completion eval harness that scores greedy continuations
against an external file of expected answers.

## At a glance

| Tool | Entry point | Input | Output |
|---|---|---|---|
| Completion REPL | `./bin/model-chat-mist.py <zdeck>` | Interactive prompts | 4 candidates per turn (greedy + 3 temperatures) |
| Eval harness | `./bin/model-eval-mist.py <zdeck> --evals file.json` | JSON of prompt/completion items | Scored report under `output/train/<run_name>/evals/` |
| Spot checks | `run_spot_check(...)` (`alien_ink.hf.gen`) | Fixed prompt seeds | Logged sample completions |

Both scripts take a zdeck name (e.g. `pre_gpt-2_wikitext_5k_mist`), resolve
the trained model under `output/train/<run_name>/`, and drive generation from
the manifest's model family. Do **not** use chat roles (`System:`, `User:`,
`Assistant:`) — the models were never trained on that format.

---

## Interactive REPL

```bash
./bin/model-chat-mist.py pre_gpt-2_wikitext_5k_mist
./bin/model-chat-mist.py pre_gemma_c4_5k_mist
./bin/model-chat-mist.py alien_ink.zdeck.pre_gemma_c4_5k_mist --max-new-tokens 120
```

Each turn is a fresh prompt — no history. The script prints **four**
candidates on separate lines: greedy (deterministic) first, then sampled at
temperatures `0.5`, `0.8`, and `1.2`. A counter and decoding stats sit beside
each completion.

```
input› The capital of Texas is
[1] Austin .  (greedy T=0, 2 tok)
[2] Austin, Texas.  (T=0.5 top_p=0.95, 4 tok)
[3] a city known for live music.  (T=0.8 top_p=0.95, 7 tok)
[4] located in the southern United States.  (T=1.2 top_p=0.95, 8 tok)
```

Type **Ctrl-C** to exit.

### Family-aware generation config

Completion settings live in `GenConfig` (`alien_ink.hf.gen`), selected by
model family:

```python
gen = manifest.gen_config(max_new_tokens=120)
completion = generate_completion(model, tokenizer, prompt, device, gen)
```

| Setting | GPT-2 / GPT-NeoX / Pythia / Llama | Gemma |
|---|---|---|
| `add_special_tokens` | `True` | `False` (avoids prepending BOS on continuation) |
| `do_sample` | `False` (greedy, deterministic) | same |
| `stop_strings` | `.`, `!`, `?` via `model.generate` | same |

Stop strings are passed to Hugging Face `generate()` as native stopping
criteria — generation halts when the model emits sentence-ending punctuation,
not by trimming the output afterward.

### Programmatic spot checks

```python
from alien_ink.hf.gen import run_spot_check, SpotCheckConfig

run_spot_check(
    output_dirs=[Path("output/train/pre-gpt-2-wikitext-5k-mist")],
    family="gpt-2",
    spot=SpotCheckConfig(num_samples=5, do_sample=True),
)
```

Default prompt seeds (`"The capital of France is"`, etc.) are short sentence
starters suited to base LM continuation.

---

## Completion eval harness

`alien_ink.hf.eval` scores a checkpoint against an **external eval JSON**:
greedy continuations for each prompt, compared to an expected completion.
Works for both `pre` and `sft` checkpoints.

```bash
./bin/model-eval-mist.py sft_smollm2-135m_geo_mist \
  --evals /path/to/geo-us-states.json

./bin/model-eval-mist.py pre_gpt-neox_wikitext_3ep_mist \
  --evals /path/to/eval.json --max-new-tokens 64
```

Or programmatically:

```python
from alien_ink.hf.eval import run_completion_eval

report = run_completion_eval(manifest=MANIFEST, evals_path=path, device=device)
```

### Eval file format

A JSON list of items, each with a unique `slug`, a `prompt`, and the expected
`completion`:

```json
[
  {"slug": "texas-capital", "prompt": "The capital of Texas is", "completion": "Austin."},
  {"slug": "everest", "prompt": "The tallest mountain on Earth is", "completion": "Mount Everest."}
]
```

Eval file contents are **never copied** into the run output — only predicted
text and scores are written.

### Decoding

Always greedy (`do_sample=False`), with the family's `add_special_tokens`
policy and stop strings disabled. `max_new_tokens` defaults to the token
length of the longest expected completion plus a small cushion; override with
`--max-new-tokens`.

### Metrics

| Metric | Kind | Meaning |
|---|---|---|
| `exact_rate` | discrete | Normalized prediction equals expected exactly |
| `prefix_rate` | discrete | Normalized prediction starts with expected |
| `char_sim` | continuous | `1 − levenshtein / max(len)` on normalized text |
| `token_f1` | continuous | Bag-of-words F1 over whitespace tokens |
| `rouge_l` | continuous | ROUGE-L F1 (longest common subsequence) |
| `mean_loss` / `mean_ppl` | continuous | Teacher-forced CE loss / perplexity of the expected completion given the prompt — the same signal as training `eval_loss` |

Text is normalized (whitespace stripped and collapsed) before string
comparison. The teacher-forced loss masks prompt tokens with `-100`, so it
averages over expected-completion tokens only.

### Results

A timestamped report — aggregate rates plus per-item predictions and scores —
is written to:

```
output/train/<run_name>/evals/<eval-file-stem>-<YYYYMMDD-HHMMSS>.json
```

A summary (hit rates, mean loss/ppl, mean similarities) is also logged at the
end of the run.
