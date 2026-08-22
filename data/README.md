# OS Tutor — Fine-Tuning Dataset v2

## Why this replaces `dataset.json`

The original `dataset.json` (10,000 examples) turned out to have only **19
unique topics**, all generated from one Socratic template. The assistant
never stated a fact — every reply was a randomly-filled
`"If you think about {analogy}, what kind of information..."` deflection.
There were zero worked numerical problems anywhere in it. That's why the
fine-tuned model could only handle the most basic definitional questions and
had no ability to solve scheduling/paging/disk/Banker's-algorithm numericals.

`dataset_v2.json` fixes both problems.

## What's in it

**2,327 examples total** (`dataset_v2_train.json` = 2,141, `dataset_v2_val.json` = 186, 8% held out), built from three sources:

### 1. Curated conceptual bank — `kb_conceptual.py` (direct textbook recall)
Hand-written, direct, textbook-accurate answers (not Socratic deflection),
covering the full standard syllabus: OS introduction & structure, process
management & threads, CPU scheduling, synchronization, deadlocks, memory
management, virtual memory, file systems, disk/mass storage, I/O systems,
protection/security, and virtualization. Includes a 36-term glossary (each
with 3 question phrasings) and 40 `OUT_OF_CONTEXT` examples (an OS passage
paired with an off-syllabus question), since that refusal behavior is part
of the production system prompt and was completely absent before.

### 2. Applied / exam-synthesis bank — `kb_advanced.py` (reasoning-chain answers)
Real exams build questions *over* concepts, not just recall of one — so this
tier targets multi-concept scenarios with full reasoning-chain answers
(phenomenon -> mechanism -> resolution), the way a strong general LLM
answers when asked cold:
- **Hardware memory models**: store buffers, StoreLoad reordering, memory
  barriers/fences, how kernels make spinlocks/mutexes correct under relaxed
  hardware ordering, MESI cache coherence.
- **Applied synchronization/deadlock scenarios**: a traced priority-inversion
  walkthrough (why it's *unbounded* and how priority inheritance bounds it),
  a concrete machine-instruction race-condition trace on `counter++`, and a
  resource-allocation-graph deadlock analysis (including the single- vs
  multi-instance distinction).
- **Real-time scheduling**: RMS vs. EDF, the Liu & Layland utilization bound
  with a worked numerical schedulability check.
- **Applied virtual memory & security**: Copy-on-Write `fork()` mechanics,
  stack buffer overflow control-flow hijacking plus ASLR/NX-bit defenses.
- **TLB shootdowns**: the race condition if a shootdown is skipped, the
  IPI-based cross-core mechanism (Local APIC / GIC), and why it scales
  poorly (IPI fan-out cost, synchronous blocking) plus batched-shootdown
  mitigation (`mmu_gather`).
- **Scalable multicore synchronization**: RCU (read-side-lock-free updates
  and grace periods), spinlock implementations (why naive test-and-set
  thrashes cache lines, TTAS, ticket locks, and MCS/queued locks reaching
  O(1) unlock cost), false sharing, and futexes (userspace fast path vs.
  kernel slow path).
- **NUMA and memory locality**: why remote memory access is slower, how a
  NUMA-aware scheduler and allocator respond, and the first-touch policy.

`build_conceptual.py` merges `kb_conceptual.py` + `kb_advanced.py` into one
pipeline automatically — no special-casing needed.

### 3. Numerical generator — 2,000 examples (`numerical_gen.py`, 400/category)
Every numerical answer is **computed by actually simulating the algorithm**,
never hand-typed or guessed, so correctness is guaranteed by construction:

| Category | Algorithms | Verified against |
|---|---|---|
| CPU scheduling | FCFS, SJF, SRTF, Round Robin, Priority | Classic GATE SRTF/RR traces (exact match) |
| Page replacement | FIFO, LRU, Optimal | Belady's Anomaly (9→10 faults) and Silberschatz's worked example (LRU=12, Optimal=9 faults) — exact match |
| Disk scheduling | FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK | Classic Silberschatz disk example (SCAN=236, LOOK=208) — exact match |
| Banker's algorithm | Safety check + resource-request grant/deny | Standard safety algorithm, hand-traced |
| Memory addressing | Paging translation, first/best/worst-fit allocation | Pure arithmetic, deterministic |

## Format

Every example matches **exactly** the RAG prompt shape `rag/qa_chain.py` and
`rag/groq_chain.py` build at inference time, so the fine-tuning distribution
matches the production distribution (important for a 1.1B model):

```json
{
  "messages": [
    {"role": "system", "content": "You are an Operating Systems tutor. Answer ONLY using the exact textbook passages provided below. ..."},
    {"role": "user", "content": "Textbook passages:\n<context>\n\nQuestion: <question>"},
    {"role": "assistant", "content": "<direct, correct answer>"}
  ]
}
```

This is a drop-in replacement for the old `dataset.json`'s schema.

## File formats

`build_dataset.py` writes the canonical chat-messages JSON (`dataset_v2*.json`)
— this is the format most chat/instruction fine-tuning frameworks (axolotl,
LLaMA-Factory, Unsloth, HF `SFTTrainer`, OpenAI's own fine-tuning API) expect
natively, and it's the same schema the original `dataset.json` already used.

Run `python export_formats.py` afterward to also produce, for tools that
want a different shape:
- **`.jsonl`** — the same examples, one JSON object per line (required by
  some tools instead of a single JSON array).
- **`.csv`** — flattened to three columns: `system`, `prompt`, `completion`
  (for tools/UIs that only accept a flat table). Values with embedded
  commas/newlines are properly quoted per RFC 4180 — open with a real CSV
  reader (`csv.DictReader`, pandas, Excel), not a naive split on `,`.

All three formats are byte-for-byte equivalent in content — pick whichever
your training tool expects.

## Regenerating / expanding

```bash
cd data
python build_conceptual.py          # sanity-check the conceptual bank alone
python build_dataset.py             # rebuild with defaults (400/category = 2,000 numericals)
python build_dataset.py --per-category 1000 --seed 7   # bigger dataset, different seed
```

- To add more **recall facts**: edit `kb_conceptual.py` (`CHAPTERS` or `GLOSSARY`).
- To add more **applied/exam-synthesis scenarios** (multi-concept, full reasoning-chain answers): edit `kb_advanced.py`, following the same `CHAPTERS` structure. This is the right place for "explain why X happens, given Y and Z interact" style questions, not simple recall.
- Either way, every entry you add is automatically wrapped into the correct chat format by `build_conceptual.py`.
- To add more **numerical problem types**: add a `gen_*` function to `numerical_gen.py` following the existing pattern (return `(context, question, answer)`, compute the answer via simulation, not by hand) and register it in `GENERATORS`.
- Both generators use a fixed random seed by default, so output is reproducible; pass `--seed` to get a different sample.

## Known caveat

The `en-dash`/`em-dash` characters (—) in this file's terminal output may
display as `�` in a non-UTF-8 Windows console (e.g. Git Bash) — this is a
display-only artifact. The actual JSON files are valid UTF-8
(`json.load(..., encoding='utf-8')` parses cleanly; verified against raw
bytes).
