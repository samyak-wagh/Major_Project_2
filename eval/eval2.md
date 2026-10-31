# Evaluation 2 — clean 17-question comparative evaluation (six systems) + calculation benchmark

*Generation 2026-10-29 → 2026-10-31, scoring 2026-10-30/31. Branch `paper-eval-pipeline`. All numbers below are computed from the result files listed in §11.*

## 1. Summary
- Six systems were compared on the **17 gold questions that have no near-duplicate in the fine-tuning data** (15 of the 32 gold questions were excluded, §5):
  cloud only (S1), fine-tuned only (S2), base Qwen only (S5), RAG + cloud (S3), RAG + fine-tuned (S4), RAG + base Qwen (S6).
  All 102 answers were generated and **all 102 scored** (RAGAS faithfulness, answer relevancy, context precision with an LLM judge).
- A separate **judge-free calculation benchmark** (27 new problems × 3 no-RAG systems = 81 answers) checks numerical correctness automatically.
- **Findings, stated plainly** (details and intervals in §7):
  1. RAG + cloud (S3) is the most faithful system; RAG + fine-tuned (S4) is lower by about 0.27 on faithfulness (interval excludes 0).
  2. **Fine-tuning did not improve the text metrics.** Against the *same* base model, the fine-tuned model has *lower* answer relevancy (without RAG: S2−S5; with RAG: S4−S6; both intervals exclude 0), and no detectable gain in faithfulness (S4−S6 interval includes 0).
  3. On calculations the cloud model is far ahead (22/27 correct); both Qwen models mostly fail. Fine-tuned is directionally better than base (8 vs 4 correct on the lenient measure) but this is **not statistically established** (sign test p = 0.22).
  4. RAG did not change the base model's or the cloud model's answer relevancy (intervals include 0).
- Limitations (small clean set, one same-family judge, no human evaluation, 256-token cap, CPU-only latency) are in §9.

## 2. Systems compared

| ID | System | Generator | Retrieval |
|---|---|---|---|
| S1 | Cloud only | Groq `openai/gpt-oss-20b` | none |
| S2 | Fine-tuned only | Qwen2.5-1.5B-Instruct + our LoRA adapter (local, CPU) | none |
| S5 | Base Qwen only | Qwen2.5-1.5B-Instruct, **adapter bypassed** (local, CPU) | none |
| S3 | RAG + cloud | Groq `openai/gpt-oss-20b` | Qdrant top-3 |
| S4 | RAG + fine-tuned | Qwen2.5-1.5B-Instruct + our LoRA adapter | Qdrant top-3 |
| S6 | RAG + base Qwen | Qwen2.5-1.5B-Instruct, adapter bypassed | Qdrant top-3 |

S2/S5 and S4/S6 use identical prompts, decoding settings, retriever and machine; only the adapter differs (`qwen_opts={"adapter": "off"}` runs the base weights via PEFT `disable_adapter()`).

## 3. Which fine-tuned model was used
- **Adapter:** `qwen-os-tutor-lora/` (identical copy at repo root `adapter_model.safetensors`). 73,911,112 bytes, SHA-256 starts `01f4aae99623f82d`, last modified 2026-10-25 22:26 (unchanged during the evaluation). First committed in `4eba7df`.
- **Base weights at inference:** `Qwen/Qwen2.5-1.5B-Instruct`, fp32 on CPU. The adapter was *trained* on the 4-bit `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit` base, so inference precision differs from training precision.
- **Recipe** (`qwen_os_tutor_finetune.ipynb`, `adapter_config.json`): QLoRA (Unsloth + TRL SFT); LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down; 3 epochs = 150 steps; effective batch 16 (2×8); lr 2e-4, cosine, 5% warm-up, weight decay 0.01; 8-bit AdamW; max length 1024; minority categories oversampled 3×.
- **Training data:** `dataset_v2_train.jsonl` (702 examples; 762 incl. validation).
- **Loss** (`checkpoint-150/trainer_state.json`): train 1.872 (step 10) → 0.232 (step 150); validation 0.577 / 0.496 / 0.515 at steps 50 / 100 / 150. The **final step-150 adapter** was used (validation loss was lowest at step 100).
- Only this adapter was evaluated; the newer fine-tune notebooks in the working tree were **not** used.

## 4. Setup
| Component | Setting |
|---|---|
| Knowledge base | `galvin.pdf`, 1000-char chunks, 200 overlap → **4,330 chunks**, Qdrant Cloud collection `os_tutor` |
| Embeddings / retriever | `BAAI/bge-small-en-v1.5`; similarity with score threshold **0.55**, **k = 3** |
| Qwen decoding | sampling, temperature 0.1, top-p 0.85, repetition penalty **1.1**, **max 256 new tokens**; concise tutor system prompt; passages truncated to 300 words |
| Groq generation | `openai/gpt-oss-20b` via the repository's RAG prompt (no retrieval for S1) |
| Judge | Groq `openai/gpt-oss-120b`, temperature 0, `reasoning_effort=low`, `max_tokens=4096`; three Groq API keys (at least two organisations) rotated when a daily quota ran out — same judge model throughout |
| Metrics | RAGAS 0.4.3: Faithfulness, Answer Relevancy (BGE embeddings), LLM Context Precision with reference. For systems without retrieval (S1, S2, S5) only Answer Relevancy is defined |
| Statistics | mean with 95% bootstrap CI over questions (10,000 resamples); paired differences over questions scored for both systems; Wilson intervals for proportions |
| Software | torch 2.14.1 (CPU), transformers 5.20.2, peft 0.21.2, ragas 0.4.3, langchain-groq 1.1.3, groq 0.37.1, qdrant-client 1.20.0, sentence-transformers 6.2.0 |
| Hardware | CPU-only Windows 11 laptop, no GPU. Part of the run was on battery power, which slowed local generation several-fold (§7.8) |

The Qdrant cluster URL in `.env` changed during the session; both clusters hold the same 4,330-chunk `os_tutor` collection built from `galvin.pdf`.

## 5. Question set
**Selection:** the 32 gold questions in `eval/run_eval.py` were compared with the 762 training+validation questions by (a) normalised exact match, (b) BGE cosine ≥ 0.82 to pick candidates, (c) an LLM check ("would memorising B let you answer A?"); `data/decontam_check.py`, outputs in `data/decontam/`.
3 exact + 12 confirmed paraphrases = **15 leaked → excluded**; **17 clean → evaluated**. The check is conservative and can over-flag topic neighbours.

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
1. **Judge moved from Gemini to Groq** (no Gemini key). `qwen/qwen3.8-27b` was rejected (1,000 output-tokens/min cap); final judge `gpt-oss-120b`. `bypass_n=True` (Groq rejects `n>1`), `reasoning_effort=low`.
2. **Conversation-memory leak fixed** (`stateless` flag): chains injected earlier turns into later prompts, so one system's answer leaked into the next.
3. **Decoding bug fixed** (`rag/qwen_chain.py`): `repetition_penalty=1.4` + `no_repeat_ngram_size=5` also apply to the prompt, preventing the model from quoting retrieved text. Now 1.1, no n-gram ban (affects all Qwen systems equally).
4. **Empty-contexts bug fixed** (`api.py`): for `backend="local"` sources were formatted through the Groq chain, returning no contexts unless Groq had been used earlier in the session.
5. **False "knowledge base empty"** (`api.py`): a failed Qdrant call was treated as an empty KB; now retried and cached.
6. **Metrics per system:** faithfulness/context precision only for RAG systems.
7. **Leak filter** (`ONLY_CLEAN_QUESTIONS`) + `data/decontam_check.py`.
8. **Generate-then-score pipeline:** `GEN_ONLY=1 python eval/run_eval.py` saves answers first (resumable via `ANSWERS_CSV`, `ONLY_SYSTEMS`); `eval/score_answers.py` scores them (resumable, key rotation, refuses to score API-error text).
9. **Failed generations repaired** (`eval/regen_empty.py`): 19 answers regenerated (18 Groq answers that were stored as `429` error text because the daily token limit was hit during generation, plus 1 empty completion) — `eval/regenerated_answers_20261029.csv`. Scores computed from error text (12 rows) were purged before final scoring. One further empty cloud answer in the calculation benchmark was regenerated.
10. **Gold reference answers corrected.** Three gold references were wrong: for the string `7 0 1 2 0 3 0 4 2 3 0 3 2` with 3 frames the correct page-fault counts are **FIFO 10, LRU 9, OPT 7** (simulation, validated on the textbook's 15/12/9 example), but the file said 9, 8, 6. Corrected in `eval/run_eval.py` and in the saved answers; only context precision uses the reference, and **re-scoring those six rows (`eval/rescore_cp.py`) changed none of the values**.
11. **Refusal-scoring rule.** S3 replied `OUT_OF_CONTEXT` to the 8-bit-address question. The judge cannot score a refusal, so the row is recorded as: answer relevancy 0.0 (RAGAS treats non-committal answers as 0), faithfulness *not applicable* (no claims; excluded from the mean), context precision 0.0 (same single retrieved passage as S4's scored row). §7.5 shows the effect of this rule.
12. **Base-Qwen systems S5/S6** added (adapter bypass per request) to isolate the effect of fine-tuning.
13. **Judge-free calculation benchmark** (`eval/num_bench.py`): 45 problems were generated (fixed seed 2026; wording differs from the training templates; ground truth from the dataset's simulators), then **reduced to the first 3 per algorithm = 27 problems** because local CPU generation was slow. Selection is deterministic, not cherry-picked.
14. **Dev-set experiment** (training prompt + greedy decoding; `eval/dev_tune.py`): no variant improved faithfulness; defaults kept. Not part of the numbers.
15. **Hardware note:** on battery power local generation ran several times slower; once on AC with the high-performance mode the benchmark ran at ≈1 answer/minute.

## 7. Results (17 clean questions, all rows scored)

### 7.1 All questions
| System | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|
| S1 - Groq Only | n/a | 0.929 [0.910, 0.948] (n=17) | n/a |
| S2 - Qwen Only | n/a | 0.814 [0.759, 0.864] (n=17) | n/a |
| S5 - Base Qwen Only | n/a | 0.944 [0.929, 0.959] (n=17) | n/a |
| S3 - RAG + Groq | 0.609 [0.424, 0.785] (n=16) | 0.872 [0.754, 0.942] (n=17) | 0.574 [0.363, 0.775] (n=17) |
| S4 - RAG + Qwen | 0.321 [0.164, 0.490] (n=17) | 0.876 [0.843, 0.907] (n=17) | 0.598 [0.382, 0.804] (n=17) |
| S6 - RAG + Base Qwen | 0.429 [0.256, 0.607] (n=17) | 0.946 [0.923, 0.966] (n=17) | 0.603 [0.392, 0.804] (n=17) |

### 7.2 Conceptual questions only (11)
| System | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|
| S1 - Groq Only | n/a | 0.936 [0.910, 0.960] (n=11) | n/a |
| S2 - Qwen Only | n/a | 0.782 [0.710, 0.846] (n=11) | n/a |
| S5 - Base Qwen Only | n/a | 0.955 [0.938, 0.972] (n=11) | n/a |
| S3 - RAG + Groq | 0.711 [0.495, 0.902] (n=11) | 0.940 [0.914, 0.964] (n=11) | 0.735 [0.500, 0.955] (n=11) |
| S4 - RAG + Qwen | 0.479 [0.285, 0.680] (n=11) | 0.875 [0.831, 0.917] (n=11) | 0.773 [0.500, 1.000] (n=11) |
| S6 - RAG + Base Qwen | 0.555 [0.339, 0.769] (n=11) | 0.962 [0.940, 0.983] (n=11) | 0.735 [0.500, 0.955] (n=11) |

### 7.3 Numerical questions only (6; RAGAS metrics are weak proxies for calculation correctness)
| System | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|
| S1 - Groq Only | n/a | 0.916 [0.896, 0.939] (n=6) | n/a |
| S2 - Qwen Only | n/a | 0.872 [0.806, 0.927] (n=6) | n/a |
| S5 - Base Qwen Only | n/a | 0.924 [0.904, 0.944] (n=6) | n/a |
| S3 - RAG + Groq | 0.384 [0.219, 0.694] (n=5) | 0.749 [0.446, 0.908] (n=6) | 0.278 [0.056, 0.611] (n=6) |
| S4 - RAG + Qwen | 0.032 [0.000, 0.074] (n=6) | 0.877 [0.829, 0.922] (n=6) | 0.278 [0.056, 0.611] (n=6) |
| S6 - RAG + Base Qwen | 0.199 [0.025, 0.412] (n=6) | 0.916 [0.879, 0.949] (n=6) | 0.361 [0.056, 0.694] (n=6) |

### 7.4 Paired differences (positive = first system higher)
| Comparison | Metric | Difference [95% CI] | Meaning |
|---|---|---|---|
| S4 − S3 | faithfulness | -0.268 [-0.445, -0.106] (n=16) — CI excludes 0 | fine-tuned vs cloud, with RAG |
| S4 − S3 | answer_relevancy | +0.003 [-0.084, +0.138] (n=17) — CI includes 0 | fine-tuned vs cloud, with RAG |
| S4 − S3 | context_precision | +0.025 [+0.000, +0.074] (n=17) — CI includes 0 | fine-tuned vs cloud, with RAG |
| S4 − S6 | faithfulness | -0.108 [-0.285, +0.050] (n=17) — CI includes 0 | effect of fine-tuning, with RAG |
| S4 − S6 | answer_relevancy | -0.070 [-0.108, -0.033] (n=17) — CI excludes 0 | effect of fine-tuning, with RAG |
| S4 − S6 | context_precision | -0.005 [-0.088, +0.074] (n=17) — CI includes 0 | effect of fine-tuning, with RAG |
| S2 − S5 | answer_relevancy | -0.130 [-0.186, -0.080] (n=17) — CI excludes 0 | effect of fine-tuning, no RAG |
| S6 − S5 | answer_relevancy | +0.002 [-0.011, +0.015] (n=17) — CI includes 0 | effect of RAG on the base model |
| S4 − S2 | answer_relevancy | +0.062 [-0.001, +0.132] (n=17) — CI includes 0 | effect of RAG on the fine-tuned model |
| S3 − S1 | answer_relevancy | -0.057 [-0.172, +0.005] (n=17) — CI includes 0 | effect of RAG on the cloud model |
| S2 − S1 | answer_relevancy | -0.115 [-0.176, -0.061] (n=17) — CI excludes 0 | fine-tuned vs cloud, no RAG |

### 7.5 Sensitivity to the refusal-scoring rule (§6.11)
S3 answer relevancy: **0.872 [0.754, 0.942]** (n=17, refusal counted as 0)
vs **0.927 [0.907, 0.947]** (n=16, that question dropped).
S4 − S3 answer relevancy on the other 16 questions: -0.056 [-0.097, -0.020] (n=16) — CI excludes 0.
The S3-vs-S4 relevancy comparison therefore depends on how that single refusal is scored.

### 7.6 Per-question scores
F = faithfulness, R = answer relevancy, P = context precision; "–" = not applicable.

| # | Chapter | Question | S1 R | S2 R | S5 R | S3 F/R/P | S4 F/R/P | S6 F/R/P |
|---|---|---|---|---|---|---|---|---|
| 1 | Deadlocks | What is the main idea behind the Banker's algorithm? | 0.98 | 0.58 | 0.98 | 1.00/0.98/0.50 | 0.57/0.85/0.50 | 1.00/1.00/0.50 |
| 2 | Deadlocks | What are the common methods for recovering from a deadlock? | 1.00 | 0.93 | 0.98 | 0.35/1.00/1.00 | 0.25/0.94/1.00 | 0.00/1.00/1.00 |
| 3 | CPU Scheduling | What is Round Robin CPU scheduling? | 0.93 | 0.78 | 0.94 | 0.90/0.92/0.58 | 0.67/0.87/1.00 | 0.25/0.91/0.58 |
| 4 | CPU Scheduling | What is the CPU-I/O burst cycle? | 1.00 | 0.86 | 1.00 | 1.00/1.00/1.00 | 1.00/0.72/1.00 | 1.00/0.97/1.00 |
| 5 | Memory Management | What is page replacement? | 0.91 | 0.66 | 0.94 | 1.00/0.88/1.00 | 0.33/0.88/1.00 | 0.75/0.97/1.00 |
| 6 | Memory Management | How does LRU page replacement work? | 0.92 | 0.80 | 0.93 | 0.88/0.92/1.00 | 0.00/0.92/1.00 | 1.00/0.96/1.00 |
| 7 | Memory Management | How does FIFO page replacement work? | 0.92 | 0.77 | 0.91 | 1.00/0.95/1.00 | 0.33/0.95/1.00 | 0.43/0.91/1.00 |
| 8 | Memory Management | How does OPT page replacement work? | 0.92 | 0.90 | 0.93 | 0.00/0.91/0.00 | 0.00/0.82/0.00 | 0.00/0.90/0.00 |
| 9 | Disk Scheduling | What is disk scheduling? | 0.94 | 0.59 | 0.94 | 0.78/0.95/1.00 | 1.00/1.00/1.00 | 0.86/0.97/1.00 |
| 10 | Disk Scheduling | How does C-LOOK differ from SCAN? | 0.84 | 0.87 | 1.00 | 0.19/0.86/0.00 | 0.33/0.85/0.00 | 0.38/1.00/0.00 |
| 11 | Disk Scheduling | How does NVM differ from HDD for scheduling? | 0.94 | 0.87 | 0.94 | 0.72/0.97/1.00 | 0.78/0.84/1.00 | 0.44/0.99/1.00 |
| 12 | Memory Management Numerical | Frames=3, FIFO, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total | 0.90 | 0.81 | 0.90 | 0.25/0.90/0.33 | 0.00/0.79/0.33 | 0.05/0.91/0.83 |
| 13 | Memory Management Numerical | Frames=3, LRU, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total  | 0.92 | 0.73 | 0.89 | 0.25/0.90/0.00 | 0.07/0.90/0.00 | 0.00/0.85/0.00 |
| 14 | Memory Management Numerical | Frames=3, OPT, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total  | 0.89 | 0.91 | 0.92 | 0.20/0.89/0.00 | 0.00/0.88/0.00 | 0.00/0.88/0.00 |
| 15 | Memory Management Numerical | Logical address=32 bits, page size=4 KB. How many page table | 0.96 | 0.92 | 0.96 | 1.00/0.93/1.00 | 0.00/0.94/1.00 | 0.38/0.97/1.00 |
| 16 | Memory Management Numerical | TLB hit ratio=90%, TLB=20ns, memory=100ns. Find effective ac | 0.89 | 0.91 | 0.93 | 0.22/0.88/0.33 | 0.12/0.81/0.33 | 0.10/0.94/0.33 |
| 17 | Memory Management Numerical | 8-bit logical address, page size=16 bytes. Bits for page num | 0.95 | 0.95 | 0.94 | –/0.00/0.00 | 0.00/0.94/0.00 | 0.67/0.95/0.00 |

### 7.7 Calculation benchmark (no retrieval; 27 problems; judge-free)
Ground truth comes from deterministic simulators (page-fault counts for FIFO/LRU/OPT, total disk-head movement for FCFS/SSTF/LOOK, average waiting time for FCFS/SJF/round-robin). "Strict" = the last number in the answer equals the truth; "lenient" = the truth appears among the numbers in the answer's last 250 characters (waiting times: ±0.011).

| System | Strict (last number) | Lenient (answer in final 250 chars) |
|---|---|---|
| S1 - Groq Only | 22/27 = 0.81 [0.63, 0.92] | 23/27 = 0.85 [0.68, 0.94] |
| S2 - Qwen Only | 1/27 = 0.04 [0.01, 0.18] | 8/27 = 0.30 [0.16, 0.48] |
| S5 - Base Qwen Only | 0/27 = 0.00 [0.00, 0.12] | 4/27 = 0.15 [0.06, 0.32] |

Per category:

**cpu_scheduling** (n=9 problems)

| System | Strict (last number) | Lenient (answer in final 250 chars) |
|---|---|---|
| S1 - Groq Only | 7/9 = 0.78 [0.45, 0.94] | 7/9 = 0.78 [0.45, 0.94] |
| S2 - Qwen Only | 0/9 = 0.00 [0.00, 0.30] | 2/9 = 0.22 [0.06, 0.55] |
| S5 - Base Qwen Only | 0/9 = 0.00 [0.00, 0.30] | 1/9 = 0.11 [0.02, 0.44] |

**disk_scheduling** (n=9 problems)

| System | Strict (last number) | Lenient (answer in final 250 chars) |
|---|---|---|
| S1 - Groq Only | 9/9 = 1.00 [0.70, 1.00] | 9/9 = 1.00 [0.70, 1.00] |
| S2 - Qwen Only | 0/9 = 0.00 [0.00, 0.30] | 1/9 = 0.11 [0.02, 0.44] |
| S5 - Base Qwen Only | 0/9 = 0.00 [0.00, 0.30] | 0/9 = 0.00 [0.00, 0.30] |

**page_replacement** (n=9 problems)

| System | Strict (last number) | Lenient (answer in final 250 chars) |
|---|---|---|
| S1 - Groq Only | 6/9 = 0.67 [0.35, 0.88] | 7/9 = 0.78 [0.45, 0.94] |
| S2 - Qwen Only | 1/9 = 0.11 [0.02, 0.44] | 5/9 = 0.56 [0.27, 0.81] |
| S5 - Base Qwen Only | 0/9 = 0.00 [0.00, 0.30] | 3/9 = 0.33 [0.12, 0.65] |

Fine-tuned (S2) vs base (S5), same 27 problems:
- strict: fine-tuned correct & base wrong = 1; base correct & fine-tuned wrong = 0; exact two-sided sign test p = 1.000
- lenient: fine-tuned correct & base wrong = 5; base correct & fine-tuned wrong = 1; exact two-sided sign test p = 0.219

Manual spot checks confirm the Qwen failures are genuine arithmetic/procedure errors (for example computing every seek from the head's starting position), not a parsing artefact. Qwen output is capped at 256 tokens, which truncates some long worked solutions; this may understate its accuracy.

### 7.8 Latency (single run, CPU-only laptop; from the generation logs)
| System | Median (s) | Mean (s) | Max (s) | n |
|---|---|---|---|---|
| S1 | 1.2 | 1.4 | 5.1 | 18 |
| S2 | 28.2 | 87.0 | 819.3 | 17 |
| S3 | 1.9 | 2.9 | 8.3 | 17 |
| S4 | 55.9 | 1845.2 | 30414.8 | 17 |
| S5 | 37.4 | 129.7 | 1471.0 | 17 |
| S6 | 48.3 | 239.8 | 1437.6 | 17 |

**Use the median.** Means/maxima are inflated by stalls when the laptop slept or the session was interrupted, and by the battery-power slowdown. Cloud latency includes network time.

## 8. Reading the results
- **Cloud model with retrieval is the most faithful system** (S3). The fine-tuned model with retrieval (S4) is clearly lower on faithfulness (§7.4).
- **No benefit from fine-tuning is detectable on the text metrics.** Relative to the base model, fine-tuning lowered answer relevancy and showed no faithfulness gain. A plausible but *untested* explanation is that the fine-tuned model answers tersely in a textbook style while RAGAS relevancy favours longer, more generic answers; the training prompt also differs from the inference prompt.
- **On calculations**, the cloud model is far ahead. The fine-tuned model is directionally better than the base model but not significantly (p = 0.22, n = 27).
- **Retrieval** is the same for S3/S4/S6 (context precision ≈ 0.57–0.60, no detectable differences) and is weak in absolute terms.
- The earlier 5-question pilot (`results_20261027_134103.csv`) was too small and 3 of its 5 questions overlapped the training data; it should not be cited as evidence.
- The same-family judge would be expected to favour Qwen-based systems, which makes the cloud model's lead on faithfulness conservative, not inflated.

## 9. Limitations
- 17 clean questions (6 numerical); intervals are wide. The calculation benchmark has 27 problems and 3 systems.
- A single LLM judge (same family as the fine-tuned model) and no human evaluation; judge access was split across keys from at least two Groq organisations.
- Calculation correctness of the RAG systems and of the full question set is not measured; the benchmark covers three problem families only and tests the no-RAG systems.
- Qwen outputs capped at 256 tokens; the training prompt differs from the inference prompt; the adapter was trained on a 4-bit base and run on fp32.
- Latency is from one CPU-only machine with interruptions and a period on battery power.
- Faithfulness is undefined for refusals; the refusal rule in §6.11 affects the S3 relevancy mean.

## 10. Reproduce / resume
```bash
python api.py                                            # terminal 1: local backend (Qwen + adapter)
GEN_ONLY=1 python eval/run_eval.py                       # S1-S6 answers for the clean questions
ANSWERS_CSV=eval/answers_<ts>.csv GEN_ONLY=1 ONLY_SYSTEMS=S5,S6 python eval/run_eval.py   # add systems to an existing file
python eval/regen_empty.py eval/answers_<ts>.csv         # repair empty / API-error answers
python eval/score_answers.py eval/answers_<ts>.csv       # score (resumable; rotates GROQ_API_KEY, GROQ_API_KEY_2, ...)
python eval/rescore_cp.py eval/answers_<ts>.csv          # only if gold references change
python eval/num_bench.py build && python eval/num_bench.py gen && python eval/num_bench.py regen && python eval/num_bench.py score
```
Secrets live in `.env` (not committed).

## 11. Files
| File | Content |
|---|---|
| `eval/answers_20261029_233832.csv` | the 102 answers (S1–S6) + retrieved contexts + (corrected) references |
| `eval/scores_20261029_233832.csv` | the 102 scored rows |
| `eval/regenerated_answers_20261029.csv` | which answers were regenerated and why |
| `eval/latency_20261029.csv` | per-system latency summary |
| `eval/num_bench_problems.json`, `eval/num_bench_answers.csv`, `eval/num_bench_scored.csv` | calculation benchmark (45 generated problems; first 3 per algorithm used) |
| `eval/score_answers.py`, `eval/rescore_cp.py`, `eval/regen_empty.py`, `eval/num_bench.py`, `eval/run_eval.py` | evaluation code |
| `data/decontam_check.py`, `data/decontam/` | train/test overlap check and outputs |
| `eval/dev_*` | dev-set prompt/decoding experiment (no improvement) |
