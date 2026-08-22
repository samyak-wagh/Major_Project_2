# OS Tutor Fine-Tuning Dataset — Usage Guide

Quick orientation for fine-tuning on `dataset_v2*`. Read this before you start; it's a 5-minute read, not a project README.

## 1. Which files to use

| Purpose | File | Format |
|---|---|---|
| Training | `dataset_v2_train.json` / `.jsonl` / `.csv` | pick one |
| Validation | `dataset_v2_val.json` / `.jsonl` / `.csv` | pick one |

Don't use `dataset_v2.json`/`.jsonl`/`.csv` (no `_train`/`_val` suffix) for training — that's the full, unsplit set; `_train`/`_val` are already a fixed 92/8 split with no overlap.

All three formats hold **identical content**, just reshaped:

- **`.json`** — one JSON array of example objects. Simple to eyeball, but you must load the whole file into memory at once.
- **`.jsonl`** — the same examples, one JSON object per line. This is what most HF-style loaders and CLI fine-tuning tools expect (`datasets.load_dataset("json", data_files=...)` reads this natively, as does OpenAI's own fine-tuning format). **Default to this one** unless you have a reason not to.
- **`.csv`** — flattened to 5 columns (`category`, `difficulty`, `system`, `prompt`, `completion`). Use this only if your tool can't consume JSON/JSONL at all (e.g. a spreadsheet-based or no-code UI tool). Properly quoted per RFC 4180 — open with a real CSV reader, not a naive comma split, since prompt/completion text contains commas and newlines.

## 2. Schema

Each example (`.json`/`.jsonl`):

```json
{
  "category": "recall",
  "difficulty": "medium",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Textbook passages:\n<context>\n\nQuestion: <question>"},
    {"role": "assistant", "content": "<answer>"}
  ]
}
```

`category` and `difficulty` sit **outside** `messages` — they're metadata for filtering/sampling, not part of the training text itself.

### Categories

| Category | What it is |
|---|---|
| `recall` | Direct single-fact retrieval — "what is X" answered straight from one passage. |
| `applied_numerical` | A scheduling/paging/disk/Banker's/memory-addressing problem with specific numbers; answer is a computed, verified worked solution. |
| `applied_conceptual` | A scenario question requiring you to recognize which single concept applies and why (no numbers). |
| `comparative` | Two related concepts' passages given together; answer synthesizes both — e.g. "how does a race condition differ from a deadlock." |
| `analytical` | A "why" question probing the mechanism behind a fact, not just the fact itself. |
| `negative_edge_case` | An in-syllabus question where the naive/expected answer is wrong or doesn't apply — teaches boundary conditions. |
| `out_of_context` | A real OS passage paired with an unrelated, off-syllabus question; correct answer is the literal string `OUT_OF_CONTEXT`. |

### Difficulty

`easy` / `medium` / `hard` — roughly, single-fact recall vs. requires-some-reasoning vs. requires-real-synthesis. `applied_*`, `comparative`, and `analytical` are never tagged `easy` (enforced at generation time) — applying, comparing, or explaining is never as trivial as bare recall, so if you see one of those three at `easy` in some future export, something regenerated incorrectly.

## 3. Current category distribution

From `dataset_v2_train.json` (702 examples):

| Category | Count | % |
|---|---:|---:|
| `applied_numerical` | 354 | 50.4% |
| `recall` | 260 | 37.0% |
| `out_of_context` | 40 | 5.7% |
| `applied_conceptual` | 17 | 2.4% |
| `negative_edge_case` | 11 | 1.6% |
| `analytical` | 10 | 1.4% |
| `comparative` | 10 | 1.4% |

**`comparative`, `analytical`, `negative_edge_case`, and `applied_conceptual` together are only ~7% of the data.** This is intentional but not final — it reflects how the dataset was built (numerical problems are cheap to generate in bulk via simulation; recall facts come one-per-KB-entry; comparative/analytical/negative entries require hand-authored multi-concept reasoning, so there are fewer of them), not a judgment that they matter less.

**Recommendation: apply weighted sampling during training, oversampling those four categories ~3-4x**, rather than training on the raw distribution as-is. Reasoning: `comparative`/`analytical`/`negative_edge_case`/`applied_conceptual` are the highest-value examples for teaching actual reasoning — they require synthesizing multiple facts, explaining mechanisms, or handling boundary conditions, which is exactly the skill that plain recall and formulaic numerical-problem-solving don't teach. At their current ~7% share, a model trained on the raw distribution will see them so rarely relative to `recall`/`applied_numerical` that they'll have negligible influence on the final weights — the reasoning-heavy signal gets drowned out by sheer volume of the easier categories, even though it's the part of the dataset that actually teaches the model to handle synthesized exam-style questions rather than definitions.

## 4. Gotchas

- **The system prompt enforces strict grounding.** Every example's `system` message is: *"You are an Operating Systems tutor. Answer ONLY using the exact textbook passages provided below. Do NOT add your own examples, do NOT invent information. If the answer is not in the provided text, reply EXACTLY with the word 'OUT_OF_CONTEXT' and nothing else."* This is deliberate — it mirrors the exact prompt the production RAG pipeline uses at inference time, so the fine-tuning distribution matches what the model actually sees in production. **If you change the system prompt, regenerate the dataset with the new prompt rather than mixing prompt styles** — training on one prompt and serving another will degrade the model's ability to follow either reliably.
- **Category labels changed recently.** Earlier exports of this dataset used a single `"applied"` category; it has since been split into `"applied_numerical"` and `"applied_conceptual"` (to distinguish numerical-simulation-sourced examples from hand-authored scenario examples). If you have a cached copy of this dataset from before that split, **it's stale — re-pull the current files** rather than relying on an old copy.
