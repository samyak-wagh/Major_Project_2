# 🎓 OS Tutor AI — a course-specific RAG tutor

Students increasingly study from general-purpose chatbots, whose fluent answers are not tied to their course material. **OS Tutor** is a specialised chatbot for a specific course, grounded in the course's standard textbook. **Operating Systems (Silberschatz, Galvin & Gagne) is the first course**; the aim is to extend the same pipeline to further courses, each built on its own standard books.

The project contains the tutor itself (RAG pipeline, API, UI), a LoRA-fine-tuned local model, a leakage-controlled evaluation, and the IEEE-format paper describing it.

---

## ✨ Features
- **RAG over the textbook** — 1000-char chunks (200 overlap), `BAAI/bge-small-en-v1.5` embeddings, Qdrant vector store, top-3 retrieval above a 0.55 similarity threshold.
- **Two interchangeable generators** sharing one retriever (selected with `LLM_BACKEND`):
  - **Local:** `Qwen2.5-1.5B-Instruct` + our QLoRA adapter (`qwen-os-tutor-lora/`), runs offline.
  - **Cloud:** Groq-hosted `openai/gpt-oss-20b`.
- **Multimodal input & diagrams** — voice input, image/OCR input and textbook-figure search (`rag/voice_handler.py`, `rag/image_handler.py`, `rag/image_search.py`, CLIP index via `ingest_clip.py`).
- **FastAPI backend + Streamlit UI** with streaming answers.
- **Leakage-controlled evaluation** — overlap screening of gold questions against the fine-tuning data, six-system RAGAS comparison with bootstrap intervals, and a judge-free calculation benchmark.

---

## 🚀 Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure `.env`
Copy `.env.example` to `.env` and fill it in (`.env` is git-ignored — never commit keys). The main settings:

| Variable | Meaning |
|---|---|
| `LLM_BACKEND` | `qwen` (fine-tuned local), `local` (legacy TinyLlama) or `groq` (cloud) |
| `QWEN_BASE_MODEL`, `LOCAL_ADAPTER_PATH` | base model id and path of the LoRA adapter |
| `GROQ_API_KEY`, `GROQ_MODEL` | cloud generator (free key at console.groq.com) |
| `QDRANT_HOST/PORT/COLLECTION` | vector store (`start_local.bat` starts a local Docker Qdrant) |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `SCORE_THRESHOLD` | retrieval settings |
| `HF_HOME` | where model weights are cached (≥ 5 GB free) |

### 3. Build the knowledge base (one-time)
Place the textbook PDF in the project root (it is not committed — copyrighted and large) and run:
```bash
python setup_db.py
```

### 4. Run
```bash
python api.py                    # terminal 1: FastAPI backend (http://localhost:8000)
python -m streamlit run app.py   # terminal 2: UI (http://localhost:8501)
```
API endpoints: `GET /status`, `POST /ask`, `POST /upload`, `POST /images/search`, `GET /images/check`. A terminal REPL is available with `python main.py`.

---

## 🧠 Fine-tuning

**Dataset** (`dataset_v2*.jsonl/.json/.csv`, see `DATASET_USAGE.md`): 762 chat-format examples in 7 categories (recall, applied_numerical, applied_conceptual, comparative, analytical, negative_edge_case, out_of_context). Each example uses the prompt shape of the RAG pipeline (`Textbook passages: … Question: …`).

**Notebooks** (Colab, free T4):

| Notebook | Purpose | Status |
|---|---|---|
| `qwen_os_tutor_finetune.ipynb` | Original QLoRA fine-tune; produced the deployed adapter (`qwen-os-tutor-lora/`, step-150 checkpoint) | used in the system-level evaluation |
| `fine_tune2.ipynb` / `fine_tune2_results.ipynb` | Leakage-aware retrain: split grouped by passage, answer-only loss, base-model baseline, 3 seeds, bootstrap tests; the second file is the executed run with outputs | executed |
| `fine_tune3.ipynb` | Follow-up pipeline: grounding-consistent data policy, refusal-aware augmentation, checkpoint selection on generation quality, prompt mixing, `PRESET` ablations (start with `PRESET = 'smoke'`) | **not yet run/evaluated** |

**Data audit findings** (these shape every result below): the 762 examples use only **126 distinct passages**; in the original split **58 of 60 validation examples shared a passage with training**; refusal is taught by only **14 distinct questions**; and about **46 % of `recall` answers are not contained in their passage**.

---

## 📊 Evaluation

### Leakage screening
`data/decontam_check.py` compares the 32 gold questions with all 762 training/validation questions (normalised exact match → BGE cosine ≥ 0.82 → LLM check). 3 exact + 12 paraphrases were excluded, leaving **17 clean questions** (outputs in `data/decontam/`).

### Six-system comparison (17 clean questions, RAGAS, judge `openai/gpt-oss-120b`)
| System | Faithfulness | Answer relevancy | Context precision | Median latency (s) |
|---|---|---|---|---|
| S1 Cloud only | n/a | 0.929 | n/a | 1.2 |
| S2 Fine-tuned only | n/a | 0.814 | n/a | 28.2 |
| S5 Base Qwen only | n/a | 0.944 | n/a | 37.4 |
| S3 RAG + cloud | **0.609** | 0.872 | 0.574 | 1.9 |
| S4 RAG + fine-tuned | 0.321 | 0.876 | 0.598 | 55.9 |
| S6 RAG + base Qwen | 0.429 | 0.946 | 0.603 | 48.3 |

Intervals, paired differences, per-question scores and caveats are in [`eval/eval2.md`](eval/eval2.md). Latency is from a CPU-only laptop (cloud includes network time).

### Judge-free calculation benchmark (27 problems, no retrieval)
Cloud 22/27 strict-correct; fine-tuned 1/27 (8/27 lenient); base Qwen 0/27 (4/27 lenient). The fine-tuned-vs-base difference is not statistically established (sign test p = 0.22).

### Generator-level study (97 held-out items, gold passages, 3 seeds — `fine_tune2_results.ipynb`)
Under the training prompt fine-tuning raises token-F1 from **0.321 to 0.393** (paired-bootstrap 95 % CI of the difference 0.039–0.104) and exact match from 0.02 to 0.11; under the deployed system prompt the advantage disappears (0.346 vs 0.359). The `OUT_OF_CONTEXT` refusal behaviour was learned in one of three seeds.

### Main takeaways
1. RAG with the cloud model is the most faithful system and by far the best at calculations.
2. At this data scale, LoRA fine-tuning of the 1.5 B model mainly changes **answer style**: it improves overlap with reference answers on held-out passages, but not LLM-judged relevancy/faithfulness with retrieved passages.
3. Train/serve **prompt consistency** and **grounded training data** matter more than model size.
4. Retrieval (context precision ≈ 0.6) is the shared ceiling for all RAG systems.

> The judge (`gpt-oss-120b`) belongs to the same family as the cloud generator, which may favour S1/S3. The judge-free benchmark and the S4-vs-S6 comparison are not affected.

### Reproduce
```bash
python api.py                                        # terminal 1 (local backend)
GEN_ONLY=1 python eval/run_eval.py                   # generate S1–S6 answers for the clean questions
python eval/regen_empty.py eval/answers_<ts>.csv     # repair empty / API-error answers
python eval/score_answers.py eval/answers_<ts>.csv   # score with RAGAS (resumable; judge via JUDGE_MODEL)
python eval/num_bench.py build && python eval/num_bench.py gen && python eval/num_bench.py score
python data/decontam_check.py                        # overlap screening
```

---

## 📄 Paper
`paper/main.tex` is the IEEE conference-format manuscript (IEEEtran; TikZ/pgfplots figures, no external images needed — compile with pdfLaTeX, e.g. on Overleaf). `paper/figures/` holds training/dataset figures generated by `paper/figures/make_figures.py`. Placeholders to fill before submission: author block and the public repository link.

---

## 📂 Project structure
```text
├── app.py, api.py, main.py      ← Streamlit UI, FastAPI backend, terminal REPL
├── setup_db.py                  ← textbook ingestion into Qdrant
├── rag/
│   ├── qwen_chain.py            ← local fine-tuned Qwen generator (+ adapter bypass for the base model)
│   ├── groq_chain.py            ← cloud generator
│   ├── qa_chain.py, vector_store.py, embeddings.py, document_processor.py
│   └── image_*.py, voice_handler.py, clip_embeddings.py
├── qwen-os-tutor-lora/          ← deployed LoRA adapter (+ training checkpoints)
├── dataset_v2*.{jsonl,json,csv} ← fine-tuning data (full / train / val)
├── data/                        ← dataset builders, decontam_check.py, data/decontam/ outputs
├── eval/                        ← run_eval.py, score_answers.py, num_bench.py, results & eval2.md
├── fine_tune2.ipynb, fine_tune2_results.ipynb, fine_tune3.ipynb, qwen_os_tutor_finetune.ipynb
└── paper/                       ← main.tex, figures/
```

## ⚠️ Status & limitations
Evaluation uses 17 clean questions (wide intervals), one LLM judge and no human evaluation; local latency is from one CPU-only machine; only one textbook, one model family and English are covered. See `eval/eval2.md` §9 for the full list.
