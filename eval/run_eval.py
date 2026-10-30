"""
run_eval.py — 4-System Comparative Evaluation for OS Tutor
===========================================================
Compares four configurations side-by-side using RAGAS metrics:

  S1 — Groq Only       : Groq LLM, NO retrieval (pure LLM baseline)
  S2 — Qwen Only       : Qwen2.5-1.5B + LoRA, NO retrieval
  S3 — RAG + Groq      : Groq LLM WITH Qdrant textbook context
  S4 — RAG + Qwen      : Qwen2.5-1.5B + LoRA WITH Qdrant context (primary)

Judge: Groq qwen/qwen3.8-27b (via GROQ_API_KEY in .env)
Metrics: Faithfulness | Answer Relevancy | Context Precision

USAGE:
  Terminal 1:  python api.py        (keep running — port 8001)
  Terminal 2:  python eval/run_eval.py
"""

import os
import sys
import time
import warnings
import datetime
import traceback

import pandas as pd
import requests
from dotenv import load_dotenv

# UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
warnings.filterwarnings("ignore", category=UserWarning)

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference as ContextPrecision
from ragas.run_config import RunConfig

from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# =============================================================================
#  CONFIGURATION — Edit these values to control the evaluation run
# =============================================================================

# ── Question selection ────────────────────────────────────────────────────────
# How many questions to run (set to None to run ALL 20 questions in DATASET)
EVAL_LIMIT = 5         # first 5 questions through all 4 systems (None = all)

# ── Batch size ────────────────────────────────────────────────────────────────
# Questions are processed in batches. Each batch is scored by the judge after
# collection, and results are checkpointed to CSV immediately.
# Smaller batches = more checkpoints = safer for long runs.
BATCH_SIZE = 1         # 1 question per batch (checkpointed to CSV)

# ── Leak filter ───────────────────────────────────────────────────────────────
# True  = evaluate only gold questions with NO near-duplicate in the fine-tuning data
#         (list built by data/decontam_check.py -> data/decontam/leaked_gold.json)
# False = all 32 questions (some also appear in the training data - results are inflated)
ONLY_CLEAN_QUESTIONS = True

# ── Systems to run ────────────────────────────────────────────────────────────
# Set to False to skip a system (useful for quick tests or if Qwen is slow)
RUN_S1_GROQ_ONLY = True    # S1: Groq LLM, no RAG
RUN_S2_QWEN_ONLY = True    # S2: Qwen + LoRA, no RAG
RUN_S3_RAG_GROQ  = True    # S3: Groq LLM + RAG
RUN_S4_RAG_QWEN  = True    # S4: Qwen + LoRA + RAG  (primary)

# ── Delay between API calls (seconds) ─────────────────────────────────────────
# Increase if hitting Groq 429 rate limits (free tier: 200K tokens/day)
INTER_CALL_DELAY = 0.5

# ── API endpoint ──────────────────────────────────────────────────────────────
API_URL  = "http://127.0.0.1:8001"

# ── Judge (RAGAS scorer) ──────────────────────────────────────────────────────
# Different model family than the Groq generator (GROQ_MODEL) to limit self-grading bias
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-27b")

# =============================================================================
# INTERNAL SETUP  (no need to edit below this line)
# =============================================================================

EVAL_DIR    = "eval"
TIMESTAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH    = os.path.join(EVAL_DIR, f"results_{TIMESTAMP}.csv")
REPORT_PATH = os.path.join(EVAL_DIR, "evaluation_report.md")
LATEST_CSV  = os.path.join(EVAL_DIR, "results_latest.csv")
ANSWERS_CSV = os.path.join(EVAL_DIR, f"answers_{TIMESTAMP}.csv")   # raw answers, saved BEFORE judging

# Build active systems list from toggle flags
_ALL_SYSTEMS = [
    {"name": "S1 - Groq Only",   "backend": "groq",  "use_rag": False, "enabled": RUN_S1_GROQ_ONLY},
    {"name": "S2 - Qwen Only",   "backend": "local", "use_rag": False, "enabled": RUN_S2_QWEN_ONLY},
    {"name": "S3 - RAG + Groq",  "backend": "groq",  "use_rag": True,  "enabled": RUN_S3_RAG_GROQ},
    {"name": "S4 - RAG + Qwen",  "backend": "local", "use_rag": True,  "enabled": RUN_S4_RAG_QWEN},
]
SYSTEMS = [s for s in _ALL_SYSTEMS if s["enabled"]]

# =============================================================================
# GOLD STANDARD QUESTION BANK — 20 Questions, 4 OS Chapters (Galvin textbook)
# =============================================================================
# Add, remove, or edit questions here freely.
# Each entry: {"question": "...", "ground_truth": "...", "chapter": "..."}
# =============================================================================
DATASET = [
    # ── Chapter 8: Deadlocks ─────────────────────────────────────────────────
    {
        "question": "What is a deadlock in an operating system?",
        "ground_truth": "A deadlock is a situation where a set of processes are blocked because each process is holding a resource and waiting for another resource acquired by some other process.",
        "chapter": "Deadlocks",
    },
    {
        "question": "What are the four necessary conditions for a deadlock to occur?",
        "ground_truth": "The four necessary conditions are mutual exclusion, hold and wait, no preemption, and circular wait. All four must hold simultaneously for a deadlock to occur.",
        "chapter": "Deadlocks",
    },
    {
        "question": "What is the main idea behind the Banker's algorithm?",
        "ground_truth": "The Banker's algorithm is a deadlock avoidance algorithm that simulates resource allocation for predetermined maximum possible amounts, then performs a safety check to ensure the system remains in a safe state before granting any resource request.",
        "chapter": "Deadlocks",
    },
    {
        "question": "What defines a safe state in deadlock avoidance?",
        "ground_truth": "A state is safe if the system can allocate resources to each process up to its maximum in some order and still avoid a deadlock. This ordering is called a safe sequence.",
        "chapter": "Deadlocks",
    },
    {
        "question": "What are the common methods for recovering from a deadlock?",
        "ground_truth": "Common methods include aborting one or more deadlocked processes to break the circular wait, or preempting resources from one or more deadlocked processes and giving them to other processes.",
        "chapter": "Deadlocks",
    },

    # ── Chapter 5: CPU Scheduling ─────────────────────────────────────────────
    {
        "question": "What is Round Robin CPU scheduling?",
        "ground_truth": "Round Robin is a preemptive CPU scheduling algorithm where each process is assigned a fixed time quantum. The CPU scheduler cycles through the ready queue, giving each process a turn. If a process does not finish in its time quantum, it is preempted and placed at the end of the ready queue.",
        "chapter": "CPU Scheduling",
    },
    {
        "question": "What is Shortest Job First (SJF) scheduling?",
        "ground_truth": "SJF scheduling assigns the CPU to the process with the smallest next CPU burst. It can be preemptive (also called Shortest Remaining Time First) or non-preemptive, and it provides the minimum average waiting time for a given set of processes.",
        "chapter": "CPU Scheduling",
    },
    {
        "question": "What is the difference between preemptive and non-preemptive scheduling?",
        "ground_truth": "In non-preemptive scheduling, once the CPU is allocated to a process it keeps the CPU until it releases it voluntarily. In preemptive scheduling, the OS can forcibly take the CPU away from a running process when a higher-priority process becomes available.",
        "chapter": "CPU Scheduling",
    },
    {
        "question": "What is the CPU-I/O burst cycle?",
        "ground_truth": "Process execution alternates between CPU execution bursts and I/O wait bursts. Processes begin with a CPU burst, followed by an I/O burst, then another CPU burst, and so on until the final CPU burst ends with a system request to terminate.",
        "chapter": "CPU Scheduling",
    },
    {
        "question": "What is the multilevel queue scheduling algorithm?",
        "ground_truth": "Multilevel queue scheduling partitions the ready queue into several separate queues, each with its own scheduling algorithm. Processes are permanently assigned to one queue based on properties such as process type. The queues themselves are scheduled with fixed priority preemptive scheduling.",
        "chapter": "CPU Scheduling",
    },

    # ── Chapter 9: Memory Management — Theory ────────────────────────────────
    {
        "question": "What is demand paging?",
        "ground_truth": "Demand paging loads pages into memory only when referenced. A page fault triggers the OS to load the page from disk into a free frame.",
        "chapter": "Memory Management",
    },
    {
        "question": "What is thrashing?",
        "ground_truth": "Thrashing occurs when a process spends more time paging than executing due to insufficient frames, causing continuous page faults and degraded performance.",
        "chapter": "Memory Management",
    },
    {
        "question": "What is the working-set model?",
        "ground_truth": "The working-set model tracks pages referenced in a recent window. The OS allocates enough frames to hold the full working set to prevent thrashing.",
        "chapter": "Memory Management",
    },
    {
        "question": "What is page replacement?",
        "ground_truth": "Page replacement evicts a victim page when a page fault occurs with no free frames. The victim is written to disk if dirty, and the new page is loaded.",
        "chapter": "Memory Management",
    },
    {
        "question": "How does LRU page replacement work?",
        "ground_truth": "LRU replaces the page not used for the longest time, approximating the optimal algorithm using past reference history.",
        "chapter": "Memory Management",
    },
    {
        "question": "What is paging?",
        "ground_truth": "Paging divides physical memory into fixed-size frames and logical memory into equal pages. A page table maps logical pages to physical frames, eliminating external fragmentation.",
        "chapter": "Memory Management",
    },
    {
        "question": "What is a TLB?",
        "ground_truth": "A TLB is a fast hardware cache for page table entries. A TLB hit gives the frame directly; a miss consults memory and updates the TLB.",
        "chapter": "Memory Management",
    },
    {
        "question": "What is the difference between logical and physical address?",
        "ground_truth": "A logical address is generated by the CPU (virtual space). A physical address is the actual RAM location. The MMU maps logical to physical at runtime.",
        "chapter": "Memory Management",
    },
    {
        "question": "What is internal vs external fragmentation?",
        "ground_truth": "Internal fragmentation is wasted space inside an allocated block. External fragmentation is free memory in non-contiguous chunks too small to satisfy a large request.",
        "chapter": "Memory Management",
    },
    {
        "question": "How does FIFO page replacement work?",
        "ground_truth": "FIFO evicts the page in memory the longest using a queue. Simple but can suffer Belady's anomaly where more frames cause more faults.",
        "chapter": "Memory Management",
    },
    {
        "question": "How does OPT page replacement work?",
        "ground_truth": "OPT replaces the page not used for the longest future time. It gives minimum page faults but is theoretical since future references are unknown.",
        "chapter": "Memory Management",
    },

    # ── Chapter 11: Disk I/O Scheduling ──────────────────────────────────────
    {
        "question": "What is disk scheduling?",
        "ground_truth": "Disk scheduling decides the order to service disk I/O requests to minimize total head movement and improve throughput.",
        "chapter": "Disk Scheduling",
    },
    {
        "question": "How does SCAN disk scheduling work?",
        "ground_truth": "SCAN moves the disk head end-to-end servicing requests, then reverses direction. Also called the elevator algorithm.",
        "chapter": "Disk Scheduling",
    },
    {
        "question": "What is SSTF disk scheduling?",
        "ground_truth": "SSTF services the pending request closest to the current head position. Reduces seek time but can cause starvation for far-end requests.",
        "chapter": "Disk Scheduling",
    },
    {
        "question": "How does C-LOOK differ from SCAN?",
        "ground_truth": "C-LOOK sweeps in one direction servicing requests, then jumps back to the lowest pending request and sweeps again instead of reversing like SCAN.",
        "chapter": "Disk Scheduling",
    },
    {
        "question": "How does NVM differ from HDD for scheduling?",
        "ground_truth": "NVM (SSDs) has no moving parts so seek time is irrelevant. FCFS works well for NVM, while HDDs need SCAN-type algorithms to reduce head movement.",
        "chapter": "Disk Scheduling",
    },

    # ── Numerical — Memory Management ─────────────────────────────────────────
    {
        "question": "Frames=3, FIFO, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total page faults?",
        "ground_truth": "FIFO with 3 frames: faults at 7,0,1,2,3,0,4,2,3,0 = 9 page faults.",
        "chapter": "Memory Management Numerical",
    },
    {
        "question": "Frames=3, LRU, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total page faults?",
        "ground_truth": "LRU with 3 frames: faults at 7,0,1,2,3,4,2,3,0 = 8 page faults. LRU outperforms FIFO here.",
        "chapter": "Memory Management Numerical",
    },
    {
        "question": "Frames=3, OPT, ref string: 7 0 1 2 0 3 0 4 2 3 0 3 2. Total page faults?",
        "ground_truth": "OPT with 3 frames: faults at 7,0,1,2,3,4 = 6 page faults. OPT gives the minimum possible.",
        "chapter": "Memory Management Numerical",
    },
    {
        "question": "Logical address=32 bits, page size=4 KB. How many page table entries?",
        "ground_truth": "Entries = 2^32 / 2^12 = 2^20 = 1,048,576 entries.",
        "chapter": "Memory Management Numerical",
    },
    {
        "question": "TLB hit ratio=90%, TLB=20ns, memory=100ns. Find effective access time.",
        "ground_truth": "EAT = 0.9*(20+100) + 0.1*(20+200) = 108 + 22 = 130 ns.",
        "chapter": "Memory Management Numerical",
    },
    {
        "question": "8-bit logical address, page size=16 bytes. Bits for page number and offset?",
        "ground_truth": "Offset = log2(16) = 4 bits. Page number = 8-4 = 4 bits. Total pages = 2^4 = 16.",
        "chapter": "Memory Management Numerical",
    },
]


# =============================================================================
# HELPERS
# =============================================================================

def get_rag_response(question: str, backend: str = "local", use_rag: bool = True):
    """
    Call the FastAPI /ask endpoint.
    Returns (answer_str, [context_strings]).
      backend : 'groq' | 'local'
      use_rag : True  = RAG mode (retrieves textbook chunks)
                False = LLM-only baseline (no context)
    """
    try:
        payload = {"question": question, "backend": backend, "use_rag": use_rag, "stateless": True}
        res = requests.post(f"{API_URL}/ask", json=payload, timeout=600)
        res.raise_for_status()
        data = res.json()
        contexts = data.get("contexts", [])
        if use_rag and not contexts:
            contexts = data.get("sources", []) or ["No specific context provided."]
        elif not use_rag:
            contexts = ["No context used — LLM-only baseline mode."]
        return data["answer"], contexts
    except requests.exceptions.ConnectionError:
        print(f"\n    [WARN] Cannot connect to API at {API_URL}. Is api.py running?")
        return "Error: API server not reachable.", ["Error"]
    except Exception as e:
        print(f"\n    [WARN] API Error: {e}")
        return "Error fetching answer.", ["Error"]


def make_eval_dataset(samples: list) -> EvaluationDataset:
    return EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["user_input"],
            response=s["response"],
            retrieved_contexts=s["retrieved_contexts"],
            reference=s["reference"],
        )
        for s in samples
    ])


def check_backend() -> bool:
    """Ping /status up to 5 times. Returns True when a chain is ready."""
    print(f"\n[CHECK] Pinging API backend at {API_URL} ...")
    for attempt in range(1, 6):
        try:
            r    = requests.get(f"{API_URL}/status", timeout=240)  # /status hits Qdrant Cloud; can take ~100 s on a slow link
            data = r.json()
            if not data.get("chain_ready"):
                print("  [WARN] Backend up but chain NOT ready. Upload the PDF first.")
                return False
            model = data.get("model", "unknown")
            print(f"  [OK] Backend ready  |  Active model: {model}")
            return True
        except requests.exceptions.ConnectionError:
            print(f"  [{attempt}/5] Not reachable — retrying in 5 s ...")
            time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"  [{attempt}/5] Timeout — retrying in 5 s ...")
            time.sleep(5)
    print("  [FAIL] Could not reach backend after 5 attempts. Run 'python api.py' first.")
    return False


def _bar(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total)
    return "[" + "=" * filled + "-" * (width - filled) + f"] {current}/{total}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  OS TUTOR — 4-SYSTEM RAGAS EVALUATION")
    print(f"  Run timestamp : {TIMESTAMP}")
    print("=" * 70)

    # ── Config summary ────────────────────────────────────────────────────────
    pool = DATASET
    if ONLY_CLEAN_QUESTIONS:
        import json
        leak_file = os.path.join("data", "decontam", "leaked_gold.json")
        if not os.path.exists(leak_file):
            print(f"\n  [ERROR] {leak_file} missing. Run: python data/decontam_check.py")
            sys.exit(1)
        leaked = set(json.load(open(leak_file, encoding="utf-8"))["leaked_questions"])
        pool = [d for d in DATASET if d["question"] not in leaked]
        print(f"\n  Leak filter ON: {len(pool)} clean of {len(DATASET)} gold questions "
              f"({len(DATASET) - len(pool)} excluded - near-duplicates of training data)")
    active_q = pool[:EVAL_LIMIT] if EVAL_LIMIT else pool
    total_q  = len(active_q)

    print(f"\n  Questions     : {total_q} of {len(DATASET)} (EVAL_LIMIT={EVAL_LIMIT})")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  # batches     : {(total_q + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"\n  Systems enabled:")
    for s in SYSTEMS:
        rag_tag = "WITH RAG" if s["use_rag"] else "NO RAG "
        print(f"    [ON]  {s['name']:<22}  backend={s['backend']:<6}  {rag_tag}")
    for s in _ALL_SYSTEMS:
        if not s["enabled"]:
            print(f"    [OFF] {s['name']:<22}  (skipped)")

    if not SYSTEMS:
        print("\n  [ERROR] No systems enabled. Set at least one RUN_Sx flag to True.")
        sys.exit(1)

    # ── Groq judge LLM ────────────────────────────────────────────────────────
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print("\n  [ERROR] GROQ_API_KEY not set in .env")
        sys.exit(1)

    print(f"\n  Loading Groq judge LLM ({JUDGE_MODEL}) ...")
    # max_tokens: gpt-oss spends tokens on reasoning, so a low cap truncates the judge.
    # bypass_n: Groq rejects n>1, so RAGAS must issue n separate calls instead.
    eval_llm = LangchainLLMWrapper(
        ChatGroq(
            model=JUDGE_MODEL,
            api_key=groq_key,
            temperature=0,
            max_retries=10,
            max_tokens=4096,   # Groq TPM cap is 8000 for this model
        ),
        bypass_n=True,
    )
    print("  Loading BAAI/bge-small-en-v1.5 embeddings ...")
    eval_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
    ]
    safe_config = RunConfig(max_retries=10, max_workers=1, timeout=300)

    # ── Backend health ────────────────────────────────────────────────────────
    if not check_backend():
        sys.exit(1)

    os.makedirs(EVAL_DIR, exist_ok=True)
    print(f"\n  Output CSV    : {CSV_PATH}\n")

    # ── Batch loop ────────────────────────────────────────────────────────────
    for i in range(0, total_q, BATCH_SIZE):
        batch     = active_q[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        batch_end = min(i + BATCH_SIZE, total_q)

        print(f"\n{'─' * 70}")
        print(f"  BATCH {batch_num}  (Q{i+1}--Q{batch_end} of {total_q})")
        print(f"  {_bar(i, total_q)}")
        print(f"{'─' * 70}")

        system_samples = {s["name"]: [] for s in SYSTEMS}

        for j, item in enumerate(batch, start=i + 1):
            q, gt, ch = item["question"], item["ground_truth"], item["chapter"]
            print(f"\n  [{j:>2}/{total_q}] [{ch}]")
            print(f"  Q: {q[:80]}{'...' if len(q) > 80 else ''}")

            for sys_cfg in SYSTEMS:
                tag = sys_cfg["name"]
                print(f"    -> {tag:<22}", end="", flush=True)
                t0 = time.time()
                ans, ctx = get_rag_response(
                    q,
                    backend=sys_cfg["backend"],
                    use_rag=sys_cfg["use_rag"],
                )
                elapsed = time.time() - t0
                print(f"  {len(ans):>5} chars  {elapsed:.1f}s")

                system_samples[tag].append({
                    "user_input": q,
                    "response": ans,
                    "retrieved_contexts": ctx,
                    "reference": gt,
                    "chapter": ch,
                })
                time.sleep(INTER_CALL_DELAY)

        raw_rows = [dict(system=t, **smp) for t, smps in system_samples.items() for smp in smps]
        pd.DataFrame(raw_rows).to_csv(ANSWERS_CSV, mode="a", header=not os.path.exists(ANSWERS_CSV), index=False)

        # ── Score this batch with the judge ──────────────────────────────────────
        print(f"\n  [SCORING] Batch {batch_num} — judge scoring {len(SYSTEMS)} systems ...")
        batch_dfs = []

        for sys_cfg in SYSTEMS:
            tag     = sys_cfg["name"]
            samples = system_samples[tag]
            if not samples:
                continue
            print(f"    Scoring {tag} ...", end="", flush=True)
            try:
                # No-RAG systems have no retrieved context, so Faithfulness / Context Precision
                # are undefined for them -> score Answer Relevancy only (leaves NaN, not a fake 0).
                sys_metrics = metrics if sys_cfg["use_rag"] else [m for m in metrics if isinstance(m, AnswerRelevancy)]
                result = evaluate(
                    dataset=make_eval_dataset(samples),
                    metrics=sys_metrics,
                    llm=eval_llm,
                    embeddings=eval_embeddings,
                    run_config=safe_config,
                )
                df_sys             = result.to_pandas()
                df_sys["system"]   = tag
                df_sys["chapter"]  = [s["chapter"] for s in samples]
                df_sys["backend"]  = sys_cfg["backend"]
                df_sys["use_rag"]  = sys_cfg["use_rag"]
                batch_dfs.append(df_sys)
                print(" [OK]")
            except Exception as e:
                print(f" [FAIL]: {e}")
                traceback.print_exc()

        # ── Checkpoint to CSV ─────────────────────────────────────────────────
        if batch_dfs:
            batch_df     = pd.concat(batch_dfs, ignore_index=True)
            write_header = not os.path.exists(CSV_PATH)
            batch_df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)
            print(f"  [SAVED] Batch {batch_num} -> {CSV_PATH}")

    # ── Final summary ─────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print("\n  [ERROR] No results were written — all batches failed.")
        sys.exit(1)

    df_all = pd.read_csv(CSV_PATH)
    df_all.to_csv(LATEST_CSV, index=False)

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
    ]
    available = [c for c in metric_cols if c in df_all.columns]

    def fmt(s, c):
        return f"{s[c]:.4f}" if c in s.index and pd.notna(s[c]) else "N/A"

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  FINAL RESULTS — 4-System Comparative Evaluation")
    print(f"{'=' * 75}")
    print(f"  {'System':<28} {'Faithfulness':>13} {'Ans Relevancy':>14} {'Ctx Precision':>14}")
    print(f"  {'-' * 70}")

    md_rows = []
    for sys_cfg in _ALL_SYSTEMS:
        sysname = sys_cfg["name"]
        if not sys_cfg["enabled"]:
            print(f"  {sysname:<28} {'(skipped)':>13}")
            continue
        mask = df_all["system"] == sysname
        if not mask.any():
            print(f"  {sysname:<28} {'(no data)':>13}")
            continue
        avg   = df_all[mask][available].mean()
        faith = fmt(avg, "faithfulness")
        relev = fmt(avg, "answer_relevancy")
        prec  = fmt(avg, "llm_context_precision_with_reference")
        print(f"  {sysname:<28} {faith:>13} {relev:>14} {prec:>14}")
        md_rows.append(f"| **{sysname}** | {faith} | {relev} | {prec} |")

    print(f"{'=' * 75}")

    # ── Markdown report ───────────────────────────────────────────────────────
    md_table = (
        "| System | Faithfulness | Answer Relevancy | Context Precision |\n"
        "| :--- | :---: | :---: | :---: |\n"
        + "\n".join(md_rows)
    )

    md = f"""# OS Tutor -- 4-System Comparative Evaluation Report

**Run timestamp**: {TIMESTAMP}
**Questions evaluated**: {total_q} of {len(DATASET)} (EVAL_LIMIT={EVAL_LIMIT}, BATCH_SIZE={BATCH_SIZE})
**Evaluation library**: ragas  |  **Judge LLM**: {JUDGE_MODEL}

## System Definitions

| System | Backend | RAG | Description |
| :--- | :---: | :---: | :--- |
| **S1 - Groq Only** | Groq cloud | No | Pure LLM baseline -- no textbook context |
| **S2 - Qwen Only** | Qwen2.5-1.5B+LoRA | No | Fine-tuned model only -- no retrieval |
| **S3 - RAG + Groq** | Groq cloud | Yes | Cloud LLM with Qdrant textbook retrieval |
| **S4 - RAG + Qwen** | Qwen2.5-1.5B+LoRA | Yes | **Primary system** -- fine-tuned + RAG |

## Comparative Results

{md_table}

## Metric Definitions

| Metric | Meaning |
| :--- | :--- |
| **Faithfulness** | Is the answer grounded in retrieved context? (0-1, higher = less hallucination) |
| **Answer Relevancy** | Does the answer address the question? (0-1, higher = better) |
| **Context Precision** | Were the most relevant chunks ranked highest? (0-1, higher = better) |

## Run Configuration

| Parameter | Value |
| :--- | :--- |
| EVAL_LIMIT | {EVAL_LIMIT} |
| BATCH_SIZE | {BATCH_SIZE} |
| Total questions | {total_q} |

## Output Files
- Full CSV: `{CSV_PATH}`
- Latest CSV: `{LATEST_CSV}`
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n  CSV    -> {CSV_PATH}")
    print(f"  Latest -> {LATEST_CSV}")
    print(f"  Report -> {REPORT_PATH}")
    print(f"\n  Dashboard: python -m streamlit run eval/ragas_dashboard.py")


if __name__ == "__main__":
    main()

