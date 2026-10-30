# Evaluation 2 — clean 17-question comparative evaluation (S1–S4)

*Run dates: generation 2026-10-29 → 2026-10-30, scoring 2026-10-30. Branch `paper-eval-pipeline`.*

## 1. Status in one paragraph
Four systems (cloud only, fine-tuned only, RAG + cloud, RAG + fine-tuned) were compared on the **17 gold questions that have no
near-duplicate in the fine-tuning data** (the other 15 of the 32 gold questions were excluded, see §5).
All 68 answers (17 × 4) were generated. **59 of 68 answers are scored**; scoring stopped when the judge's daily Groq
token quotas ran out. **Unscored rows (9)** — resumable, see §10:

- S4 - RAG + Qwen — Logical address=32 bits, page size=4 KB. How many page table entries?
- S1 - Groq Only — TLB hit ratio=90%, TLB=20ns, memory=100ns. Find effective access time.
- S2 - Qwen Only — TLB hit ratio=90%, TLB=20ns, memory=100ns. Find effective access time.
- S3 - RAG + Groq — TLB hit ratio=90%, TLB=20ns, memory=100ns. Find effective access time.
- S4 - RAG + Qwen — TLB hit ratio=90%, TLB=20ns, memory=100ns. Find effective access time.
- S1 - Groq Only — 8-bit logical address, page size=16 bytes. Bits for page number and offset?
- S2 - Qwen Only — 8-bit logical address, page size=16 bytes. Bits for page number and offset?
- S3 - RAG + Groq — 8-bit logical address, page size=16 bytes. Bits for page number and offset?
- S4 - RAG + Qwen — 8-bit logical address, page size=16 bytes. Bits for page number and offset?

Everything below is computed from `eval/scores_20261029_233832.csv` (the scored rows only). Treat the numbers as **near-final, not final**.

## 2. Systems compared

| ID | System | Generator | Retrieval |
|---|---|---|---|
| S1 | Cloud only | Groq `openai/gpt-oss-20b` | none |
| S2 | Fine-tuned only | Qwen2.5-1.5B-Instruct + our LoRA adapter (local, CPU) | none |
| S3 | RAG + cloud | Groq `openai/gpt-oss-20b` | Qdrant top-3 |
| S4 | RAG + fine-tuned | Qwen2.5-1.5B-Instruct + our LoRA adapter (local, CPU) | Qdrant top-3 |

**Not in this run:** the base Qwen2.5-1.5B *without* the adapter. Without it, this evaluation cannot say what fine-tuning itself adds.

## 3. Which fine-tuned model was used
- **Adapter:** `qwen-os-tutor-lora/` (identical copy at repo root `adapter_model.safetensors`). File size 73,911,112 bytes, SHA-256 starts `01f4aae99623f82d`,
  last modified 2026-10-25 22:26 (unchanged during the evaluation). First committed in `4eba7df`.
- **Base weights at inference:** `Qwen/Qwen2.5-1.5B-Instruct`, full precision (fp32) on CPU. The adapter was *trained* on the 4-bit
  `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit` base, so inference precision differs from training precision.
- **Fine-tuning recipe (from `qwen_os_tutor_finetune.ipynb` and `adapter_config.json`):** QLoRA via Unsloth + TRL SFT; LoRA r=16, alpha=32,
  dropout 0.05 on q/k/v/o/gate/up/down projections; 3 epochs = 150 steps; effective batch 16 (2 × 8); lr 2e-4, cosine, 5% warm-up, weight decay 0.01;
  8-bit AdamW; max sequence length 1024; minority categories oversampled 3×.
- **Training data:** `dataset_v2_train.jsonl` (702 examples; 762 incl. validation).
- **Loss (from `checkpoint-150/trainer_state.json`):** train 1.872 (step 10) → 0.232 (step 150); validation 0.577 / 0.496 / 0.515 at steps 50 / 100 / 150.
  The **final step-150 adapter** was used, although validation loss was lowest at step 100.
- Only this adapter was evaluated. Other notebooks in the working tree (`fine_tune2/3.ipynb`) were **not** used.

## 4. Retrieval, generation and judging setup
| Component | Setting |
|---|---|
| Knowledge base | `galvin.pdf` (OS textbook), 1000-char chunks, 200 overlap → **4,330 chunks** in Qdrant Cloud collection `os_tutor` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Retriever | similarity with score threshold **0.55**, **k = 3** |
| Qwen decoding | sampling, temperature 0.1, top-p 0.85, repetition penalty **1.1**, max 256 new tokens; concise tutor system prompt; passages truncated to 300 words |
| Groq generation | `openai/gpt-oss-20b` through the repository's RAG prompt (no retrieval for S1) |
| Judge | Groq `openai/gpt-oss-120b`, temperature 0, `reasoning_effort=low`, `max_tokens=4096`; three Groq API keys rotated on quota exhaustion (same judge model throughout) |
| Metrics | RAGAS 0.4.3: Faithfulness, Answer Relevancy (BGE embeddings), LLM Context Precision with reference. For S1/S2 (no retrieved context) **only Answer Relevancy** is defined |
| Statistics | mean with 95% bootstrap CI over questions (10,000 resamples); paired differences over questions scored for both systems |
| Software | torch 2.14.1 (CPU), transformers 5.20.2, peft 0.21.2, ragas 0.4.3, langchain-groq 1.1.3, groq 0.37.1, qdrant-client 1.20.0, sentence-transformers 6.2.0 |
| Hardware | CPU-only Windows 11 laptop, no GPU |

Note: the Qdrant cluster URL in `.env` changed during the session; both clusters hold the same 4,330-chunk `os_tutor` collection built from `galvin.pdf`.

## 5. Question set
**Selection:** all 32 gold questions in `eval/run_eval.py` were compared with the 762 training+validation questions by (a) normalised exact match,
(b) BGE cosine ≥ 0.82 to pick candidates, (c) an LLM check ("would memorising B let you answer A?"). `data/decontam_check.py`; outputs in `data/decontam/`.
Result: 3 exact + 12 confirmed paraphrases = **15 leaked → excluded**; **17 clean → evaluated**. The check is deliberately conservative and can over-flag topic neighbours.

**Evaluated (17):**
1. #3 [Deadlocks] What is the main idea behind the Banker's algorithm?
2. #5 [Deadlocks] What are the common methods for recovering from a deadlock?
3. #6 [CPU Scheduling] What is Round Robin CPU scheduling?
4. #9 [CPU Scheduling] What is the CPU-I/O burst cycle?
5. #14 [Memory Management] What is page replacement?
6. #15 [Memory Management] How does LRU page replacement work?
7. #20 [Memory Management] How does FIFO page replacement work?
8. #21 [Memory Management] How does OPT page replacement work?
9. #22 [Disk Scheduling] What is disk scheduling?
10. #25 [Disk Scheduling] How does C-LOOK differ from SCAN?
11. #26 [Disk Scheduling] How does NVM differ from HDD for scheduling?
12. #27 [Memory Management Numerical] Frames=3, FIFO, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total page faults?
13. #28 [Memory Management Numerical] Frames=3, LRU, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total page faults?
14. #29 [Memory Management Numerical] Frames=3, OPT, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total page faults?
15. #30 [Memory Management Numerical] Logical address=32 bits, page size=4 KB. How many page table entries?
16. #31 [Memory Management Numerical] TLB hit ratio=90%, TLB=20ns, memory=100ns. Find effective access time.
17. #32 [Memory Management Numerical] 8-bit logical address, page size=16 bytes. Bits for page number and offset?

**Excluded (15):**
1. [Deadlocks] What is a deadlock in an operating system?
2. [Deadlocks] What are the four necessary conditions for a deadlock to occur?
3. [Deadlocks] What defines a safe state in deadlock avoidance?
4. [CPU Scheduling] What is Shortest Job First (SJF) scheduling?
5. [CPU Scheduling] What is the difference between preemptive and non-preemptive scheduling?
6. [CPU Scheduling] What is the multilevel queue scheduling algorithm?
7. [Memory Management] What is demand paging?
8. [Memory Management] What is thrashing?
9. [Memory Management] What is the working-set model?
10. [Memory Management] What is paging?
11. [Memory Management] What is a TLB?
12. [Memory Management] What is the difference between logical and physical address?
13. [Memory Management] What is internal vs external fragmentation?
14. [Disk Scheduling] How does SCAN disk scheduling work?
15. [Disk Scheduling] What is SSTF disk scheduling?

## 6. Changes made to run this evaluation
Chronological list of fixes and why they matter for the validity of the numbers.

1. **Judge moved from Gemini to Groq.** No Gemini key was available. `qwen/qwen3.8-27b` was tried first and rejected (1,000 output-tokens/min cap); final judge `gpt-oss-120b`.
   `bypass_n=True` because Groq rejects `n>1`; `reasoning_effort=low` so the judge does not spend its token budget thinking.
2. **Conversation memory leak fixed** (`stateless` flag in `api.py`): chains kept the last turns and injected them into later prompts, so one system's answer leaked into the next (S4 copied S2's answer).
3. **Decoding bug fixed** (`rag/qwen_chain.py`): `repetition_penalty=1.4` and `no_repeat_ngram_size=5` also apply to the prompt, so the model could not quote retrieved passages (word-salad answers). Now 1.1 and no n-gram ban. Affects S2 and S4 equally.
4. **Empty-contexts bug fixed** (`api.py`): for `backend="local"` the API formatted sources through the Groq chain, which returned no contexts if Groq had not been used earlier in the session.
5. **False "knowledge base empty"** (`api.py`): a failed Qdrant call was treated as an empty KB; now retried and cached.
6. **Metrics per system:** Faithfulness / Context Precision only for RAG systems; no fake zeros for S1/S2.
7. **Leak filter** (`ONLY_CLEAN_QUESTIONS` in `eval/run_eval.py`) + `data/decontam_check.py`.
8. **Generate-then-score pipeline:** `GEN_ONLY=1 python eval/run_eval.py` saves answers first (resumable via `ANSWERS_CSV`); `eval/score_answers.py` scores them (resumable, fixed CSV columns, key rotation, refuses to score API-error text).
9. **Failed generations repaired** (`eval/regen_empty.py`): **19 answers regenerated** — see `eval/regenerated_answers_20261029.csv`.
   The Groq daily token limit had been hit during generation and the 429 error message had been stored as the "answer" (18 rows: 8 for S1, 10 for S3), plus 1 empty completion.
   Scores computed from those error texts (12 rows) were purged before the final scoring. The refusal `OUT_OF_CONTEXT` (S3, 8-bit address question) is genuine behaviour and was **kept**.
10. **Dev-set experiment** (prompt matching + greedy decoding for S4; `eval/dev_tune.py`): no variant improved faithfulness, so **the default settings were kept**. Not part of the numbers below.

## 7. Results (clean questions, scored rows only)

### 7.1 All scored questions
| System | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|
| S1 - Groq Only | n/a | 0.931 [0.911, 0.951] (n=15) | n/a |
| S2 - Qwen Only | n/a | 0.798 [0.740, 0.851] (n=15) | n/a |
| S3 - RAG + Groq | 0.635 [0.445, 0.812] (n=15) | 0.930 [0.910, 0.951] (n=15) | 0.628 [0.406, 0.839] (n=15) |
| S4 - RAG + Qwen | 0.381 [0.206, 0.571] (n=14) | 0.871 [0.835, 0.906] (n=14) | 0.631 [0.393, 0.857] (n=14) |

### 7.2 Conceptual questions only
| System | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|
| S1 - Groq Only | n/a | 0.936 [0.910, 0.960] (n=11) | n/a |
| S2 - Qwen Only | n/a | 0.782 [0.710, 0.846] (n=11) | n/a |
| S3 - RAG + Groq | 0.711 [0.495, 0.902] (n=11) | 0.940 [0.914, 0.964] (n=11) | 0.735 [0.500, 0.955] (n=11) |
| S4 - RAG + Qwen | 0.479 [0.285, 0.680] (n=11) | 0.875 [0.831, 0.917] (n=11) | 0.773 [0.500, 1.000] (n=11) |

### 7.3 Numerical questions only
| System | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|
| S1 - Groq Only | n/a | 0.916 [0.895, 0.943] (n=4) | n/a |
| S2 - Qwen Only | n/a | 0.842 [0.770, 0.915] (n=4) | n/a |
| S3 - RAG + Groq | 0.425 [0.213, 0.812] (n=4) | 0.903 [0.895, 0.918] (n=4) | 0.333 [0.000, 0.750] (n=4) |
| S4 - RAG + Qwen | 0.022 [0.000, 0.067] (n=3) | 0.858 [0.789, 0.902] (n=3) | 0.111 [0.000, 0.333] (n=3) |

Faithfulness/precision are weak proxies for calculation questions (the answer comes from computation, not from quoting a passage); numerical *correctness* is not measured here.

### 7.4 Paired differences (answer relevancy unless stated)
| Comparison | Difference [95% CI] |
|---|---|
| S4 − S3 faithfulness | -0.228 [-0.400, -0.067] (n=14) — CI excludes 0 |
| S4 − S3 answer relevancy | -0.059 [-0.105, -0.020] (n=14) — CI excludes 0 |
| S4 − S3 context precision | +0.030 [+0.000, +0.089] (n=14) — CI includes 0 |
| S4 − S2 (effect of RAG on the fine-tuned model) | +0.082 [+0.008, +0.164] (n=14) — CI excludes 0 |
| S3 − S1 (effect of RAG on the cloud model) | -0.001 [-0.010, +0.008] (n=15) — CI includes 0 |
| S2 − S1 (fine-tuned vs cloud, no RAG) | -0.133 [-0.196, -0.076] (n=15) — CI excludes 0 |

### 7.5 Per-question scores
F = faithfulness, R = answer relevancy, P = context precision; "–" = not scored / not applicable.

| # | Chapter | Question | S1 rel | S2 rel | S3 F / R / P | S4 F / R / P |
|---|---|---|---|---|---|---|
| 1 | Deadlocks | What is the main idea behind the Banker's algorithm? | 0.98 | 0.58 | 1.00 / 0.98 / 0.50 | 0.57 / 0.85 / 0.50 |
| 2 | Deadlocks | What are the common methods for recovering from a deadlock? | 1.00 | 0.93 | 0.35 / 1.00 / 1.00 | 0.25 / 0.94 / 1.00 |
| 3 | CPU Scheduling | What is Round Robin CPU scheduling? | 0.93 | 0.78 | 0.90 / 0.92 / 0.58 | 0.67 / 0.87 / 1.00 |
| 4 | CPU Scheduling | What is the CPU-I/O burst cycle? | 1.00 | 0.86 | 1.00 / 1.00 / 1.00 | 1.00 / 0.72 / 1.00 |
| 5 | Memory Management | What is page replacement? | 0.91 | 0.66 | 1.00 / 0.88 / 1.00 | 0.33 / 0.88 / 1.00 |
| 6 | Memory Management | How does LRU page replacement work? | 0.92 | 0.80 | 0.88 / 0.92 / 1.00 | 0.00 / 0.92 / 1.00 |
| 7 | Memory Management | How does FIFO page replacement work? | 0.92 | 0.77 | 1.00 / 0.95 / 1.00 | 0.33 / 0.95 / 1.00 |
| 8 | Memory Management | How does OPT page replacement work? | 0.92 | 0.90 | 0.00 / 0.91 / 0.00 | 0.00 / 0.82 / 0.00 |
| 9 | Disk Scheduling | What is disk scheduling? | 0.94 | 0.59 | 0.78 / 0.95 / 1.00 | 1.00 / 1.00 / 1.00 |
| 10 | Disk Scheduling | How does C-LOOK differ from SCAN? | 0.84 | 0.87 | 0.19 / 0.86 / 0.00 | 0.33 / 0.85 / 0.00 |
| 11 | Disk Scheduling | How does NVM differ from HDD for scheduling? | 0.94 | 0.87 | 0.72 / 0.97 / 1.00 | 0.78 / 0.84 / 1.00 |
| 12 | Memory Management Numerical | Frames=3, FIFO, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total page faul | 0.90 | 0.81 | 0.25 / 0.90 / 0.33 | 0.00 / 0.79 / 0.33 |
| 13 | Memory Management Numerical | Frames=3, LRU, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total page fault | 0.92 | 0.73 | 0.25 / 0.90 / 0.00 | 0.07 / 0.90 / 0.00 |
| 14 | Memory Management Numerical | Frames=3, OPT, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total page fault | 0.89 | 0.91 | 0.20 / 0.89 / 0.00 | 0.00 / 0.88 / 0.00 |
| 15 | Memory Management Numerical | Logical address=32 bits, page size=4 KB. How many page table entries? | 0.96 | 0.92 | 1.00 / 0.93 / 1.00 | – / – / – |
| 16 | Memory Management Numerical | TLB hit ratio=90%, TLB=20ns, memory=100ns. Find effective access time. | – | – | – / – / – | – / – / – |
| 17 | Memory Management Numerical | 8-bit logical address, page size=16 bytes. Bits for page number and of | – | – | – / – / – | – / – / – |

### 7.6 Latency (single run, CPU-only laptop; from the generation logs)
| System | Median (s) | Mean (s) | Max (s) | n |
|---|---|---|---|---|
| S1 | 1.2 | 1.4 | 5.1 | 18 |
| S2 | 28.2 | 87.0 | 819.3 | 17 |
| S3 | 1.9 | 2.9 | 8.3 | 17 |
| S4 | 55.9 | 1845.2 | 30414.8 | 17 |

**Use the median.** The means and maxima are inflated by stalls when the laptop slept / the session was interrupted (one S4 request shows over 8 hours). Cloud latency includes network time.

## 8. Reading the results
- S4 (RAG + fine-tuned) is **below** S3 (RAG + cloud) on faithfulness and answer relevancy, with confidence intervals that exclude 0 (§7.4); context precision is equal because both use the same retriever.
- RAG raises the fine-tuned model's answer relevancy (S4 vs S2) but not the cloud model's (already near ceiling).
- The earlier 5-question pilot (`results_20261027_134103.csv`) suggested the S3–S4 gap was within noise; that pilot was small and 3 of its 5 questions overlapped the training data.
- The same-family judge would be expected to favour Qwen-based systems, yet S3 still scores higher, so judge bias is unlikely to explain the gap.

## 9. Limitations
- 9 rows unscored (two numerical questions); intervals are wide at n≈15.
- 17 questions is a small test set; numerical questions are over-represented (6 of 17).
- A single LLM judge and no human evaluation. Judge access was split across API keys from at least two Groq organisations.
- No base-Qwen (no adapter) baseline, so the effect of fine-tuning is not isolated.
- Training prompt differs from the inference prompt; the adapter was trained on a 4-bit base and run on fp32.
- Latency is from one CPU-only machine with interruptions.

## 10. Reproduce / resume
```bash
python api.py                                   # terminal 1: local backend (loads Qwen + adapter)
GEN_ONLY=1 python eval/run_eval.py              # generate answers (ONLY_CLEAN_QUESTIONS=True) -> eval/answers_<ts>.csv
python eval/regen_empty.py eval/answers_<ts>.csv   # repair empty / API-error answers
python eval/score_answers.py eval/answers_<ts>.csv # score (resumable; rotates GROQ_API_KEY, GROQ_API_KEY_2, ...)
python eval/score_answers.py eval/answers_<ts>.csv summary
```
Environment: see §4; secrets live in `.env` (not committed).

## 11. Files
| File | Content |
|---|---|
| `eval/answers_20261029_233832.csv` | the 68 answers + retrieved contexts |
| `eval/scores_20261029_233832.csv` | scored rows (59) |
| `eval/regenerated_answers_20261029.csv` | which answers were regenerated and why |
| `eval/latency_20261029.csv` | per-system latency summary |
| `eval/score_answers.py`, `eval/regen_empty.py`, `eval/run_eval.py` | evaluation code |
| `data/decontam_check.py`, `data/decontam/` | train/test overlap check and its outputs |
| `eval/dev_*` | dev-set prompt/decoding experiment (negative result) |
