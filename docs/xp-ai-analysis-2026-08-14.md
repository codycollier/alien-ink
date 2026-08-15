# Memorization Experiment Analysis

**Date:** 2026-08-14

## Summary

The memorization experiments establish a sharp result: the models can memorize explicitly supervised answers, but ordinary corpus training does not reliably turn prose into accessible factual knowledge.

The direct-completion experiments succeeded:

- Pythia-70M's direct completion loss fell from 3.701 to 0.00454 after 9.6 epochs.
- Pythia-410M reached 0.000026 after 8.6 epochs.

Corpus training behaved differently:

- At 70M, training continued for 100 epochs, yet exact-sentence completion loss only moved from 4.778 to 2.779.
- The interrupted 410M corpus run performed better, moving from 2.413 to roughly 1.4–1.55 by epoch 70, but remained far from memorization despite near-zero document-training loss.

Scale helps corpus absorption, but the shape of the supervision matters much more. The most useful next step is to separate question-answering behavior, extraction from supplied context, storage of facts in model parameters, and retrieval of those facts under novel question formulations.

## What the Existing Experiments Establish

The direct-completion manifests train and evaluate on the same prompt/completion pairs, with prompt tokens masked from the loss. Their falling loss demonstrates that both Pythia-70M and Pythia-410M can optimize the desired answer tokens and memorize the complete supervised dataset.

The sentence-memorization manifests instead train an ordinary causal-language-model objective over complete geo documents. They evaluate selected exact sentence continuations using masked completion loss. Their results show that minimizing document loss—even nearly to zero—does not imply that facts in those documents become readily accessible from a sentence prefix.

These experiments therefore distinguish two capabilities:

1. **Explicit answer learning:** Given direct prompt/completion supervision, the models memorize the answers successfully.
2. **Knowledge absorption and retrieval:** Given only prose, the models do not reliably expose the same information in response to a completion probe.

The existing roadmap identifies the appropriate conceptual progression: absorb a corpus with continued language-model training, then teach answering behavior with QA-pair supervised fine-tuning.

## Evaluation Caveats

Several entries in the exact-sentence evaluation use prompts that do not identify their source document. For example, `With a population of` could refer to Alaska, Florida, Michigan, Virginia, or another state. Even a knowledgeable closed-book model cannot uniquely infer the expected state from such a prompt.

The sentence-training path also drops trailing partial document blocks. Consequently, an evaluation fact may appear in the source document without appearing in any retained training block. Every evaluated completion should be checked against the actual tokenized training examples before interpreting its loss as a knowledge-absorption measurement.

The next evaluation should therefore:

- identify the relevant state or territory in every prompt;
- verify that each tested fact occurs in retained training tokens;
- separate ambiguous prompts from genuinely entity-conditioned questions;
- report answer-level metrics in addition to teacher-forced loss.

## Recommended Next Experiments

### 1. QA-Format Transfer with Known Facts

This is the highest-priority next experiment.

Construct supervised QA pairs from the same 53 exact facts. For example:

```text
Question: What was Arizona's population in 1910?
Answer: 294,353.
```

Create three partitions:

- **Training questions:** One phrasing per supervised fact.
- **Paraphrase evaluation:** Unseen question phrasings for the same facts.
- **Fact-held-out evaluation:** Questions about states whose facts receive no QA supervision.

Run the same QA fine-tuning treatment from two initial checkpoints:

1. Pythia-410M base → QA SFT
2. Corpus-trained Pythia-410M → QA SFT

This comparison isolates three capabilities:

- memorization of directly trained answer pairs;
- generalization across question paraphrases;
- transfer of facts learned from prose but omitted from QA training.

The difference between the two initial checkpoints on fact-held-out questions is the most direct measurement of the value of corpus absorption.

### 2. Context-Grounded QA Before Closed-Book QA

Train and evaluate examples in the following form:

```text
Context: [relevant corpus passage]
Question: What was Arizona's population in 1910?
Answer:
```

Compare four conditions:

- question only;
- question plus the exact supporting paragraph;
- question plus the complete source document;
- question plus one relevant passage and several irrelevant passages.

This tests whether the model can learn answer behavior independently of parameter memorization. If context-grounded QA succeeds while closed-book QA fails, the remaining bottleneck is knowledge storage or retrieval rather than question interpretation and answer generation.

### 3. Entity-Conditioned Sentence Absorption

Repair the existing exact-sentence probe so that every prompt identifies its source document:

```text
State: Alaska
Text: With a population of
```

Compare:

- the current raw sentence prefix;
- entity name plus sentence prefix;
- a natural question;
- a natural question plus its supporting passage.

Before training, verify that every scored completion occurs inside a retained training block. This experiment will estimate how much of the weak sentence-memorization result comes from ambiguous prompts or dropped blocks, rather than a genuine inability to absorb the corpus.

### 4. Supervision-Density Ladder

Generate 1, 2, 4, and 8 training question templates for each fact. Reserve a separate template family exclusively for evaluation, and keep the optimizer-step budget constant across treatments.

Measure:

- exact answer rate;
- normalized token F1;
- teacher-forced completion loss;
- per-fact success rate.

This determines how many linguistic views of a fact are required before the model learns a relation that transfers beyond one memorized prompt string.

### 5. Whole-Document Holdout

Split the dataset by state or territory before generating corpus and QA examples:

- **Corpus + QA trained:** States represented in both training stages.
- **Corpus only:** States present during corpus training but excluded from QA supervision.
- **Fully unseen:** States excluded from both stages.

Evaluate fresh questions on all three groups.

| Group | Interpretation of success |
|---|---|
| Corpus + QA trained | End-to-end learnability |
| Corpus only | Transfer from prose into QA behavior |
| Fully unseen | Pretrained knowledge, task generalization, or leakage baseline |

Do not randomly split question rows. Different templates describing the same fact would leak knowledge across training and evaluation.

### 6. Mixed-Objective Versus Sequential Training

Compare equal-token-budget treatments:

- QA SFT only;
- corpus training → QA SFT;
- interleaved corpus and QA examples;
- corpus training → QA SFT → short corpus replay.

Evaluate both QA accuracy and held-out general-language loss. This will show whether sequential QA training best exposes absorbed facts and whether corpus replay mitigates forgetting.

## Recommended Immediate Sequence

Run experiments 1 and 2 together with Pythia-410M. They form a compact diagnostic ladder:

```text
direct completion → paraphrased question → question over supplied context
                                   ↘ closed-book corpus-only facts
```

Use a single state-level split and fixed evaluation set across all conditions. Report exact match, normalized token F1, and teacher-forced loss. Preserve per-example results so failures can be classified as formatting errors, wrong-entity retrieval, partially correct answers, or complete misses.

This sequence advances the investigation from “can the model memorize?” to four more useful questions:

1. Can it interpret questions?
2. Can it extract answers from supplied context?
3. Can it retain corpus facts in its parameters?
4. Can it retrieve those facts under a novel formulation?
