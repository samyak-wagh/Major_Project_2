"""
RAG Evaluation Script — verified for ragas 0.4.3
=================================================
Key ragas 0.4.3 facts (verified from source):
  - ragas.metrics._faithfulness.Faithfulness   → MetricWithLLM, works with LangChain
  - ragas.metrics._answer_relevance.AnswerRelevancy → same
  - ragas.metrics._context_precision.ContextPrecision → same
  - evaluate() auto-wraps raw LangChain LLMs (evaluation.py lines 158-161)
  - SingleTurnSample columns: user_input, response, retrieved_contexts, reference
  - ragas.metrics.collections.* metrics ONLY accept InstructorLLM — do NOT use these

Run:
    Terminal 1: python api.py           (keep running)
    Terminal 2: python eval/run_eval.py
"""

import pandas as pd
import requests
import time
import os
import warnings

# Suppress ragas deprecation noise — we know evaluate() is deprecated, we still use it
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

# ── ragas 0.4.3 — use the OLD MetricWithLLM-based metrics (LangChain-compatible) ──
# Do NOT import from ragas.metrics.collections — those only accept InstructorLLM
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference as ContextPrecision
from ragas.run_config import RunConfig

# ── LangChain + Google ────────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# GOLD STANDARD DATASET — Galvin OS, Chapter 8: Deadlocks
# =============================================================================
DATASET = [
    {"question": "What is a deadlock in an operating system?",
     "ground_truth": "A deadlock is a situation where a set of processes are blocked because each process is holding a resource and waiting for another resource acquired by some other process."},
    {"question": "What are the four necessary conditions for a deadlock to occur?",
     "ground_truth": "The four necessary conditions are mutual exclusion, hold and wait, no preemption, and circular wait."},
    {"question": "What is the main idea behind the Banker's algorithm?",
     "ground_truth": "The Banker's algorithm is a deadlock avoidance algorithm that simulates the allocation of predetermined maximum possible amounts of all resources, and then makes an 's-state' check to test for possible deadlock activities before deciding whether allocation should be allowed."},
    {"question": "What defines a safe state in deadlock avoidance?",
     "ground_truth": "A state is safe if the system can allocate resources to each process (up to its maximum) in some order and still avoid a deadlock. This order is called a safe sequence."},
    {"question": "What are the common methods for recovering from a deadlock?",
     "ground_truth": "Common methods include aborting one or more of the deadlocked processes to break the circular wait, or preempting resources from one or more deadlocked processes."},
]

API_URL   = "http://127.0.0.1:8000/ask"
BATCH_SIZE = 5
CSV_PATH   = "eval/results.csv"



# =============================================================================
# HELPERS
# =============================================================================

def get_rag_response(question: str, use_rag: bool = True):
    """Call the FastAPI backend, return (answer_str, [context_strings])."""
    try:
        payload = {"question": question, "use_rag": use_rag}
        res = requests.post(API_URL, json=payload, timeout=600)
        res.raise_for_status()
        data = res.json()
        contexts = data.get("contexts", [])
        if not contexts:
            contexts = data.get("sources", []) or ["No specific context provided."]
        return data["answer"], contexts
    except requests.exceptions.ConnectionError:
        print("    ⚠ RAG Error: server not reachable.")
        return "Error fetching answer.", ["Error"]
    except Exception as e:
        print(f"    ⚠ RAG Error: {e}")
        return "Error fetching answer.", ["Error"]


def make_eval_dataset(samples: list) -> EvaluationDataset:
    """
    Build an EvaluationDataset from a list of dicts.
    Required keys (ragas 0.4.3 SingleTurnSample schema):
        user_input, response, retrieved_contexts (List[str]), reference
    """
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
    """Ping /status with up to 5 retries (30s each). Returns True if ready."""
    print("\n🔍 Checking FastAPI backend at http://127.0.0.1:8000 ...")
    for attempt in range(1, 6):
        try:
            r = requests.get("http://127.0.0.1:8000/status", timeout=30)
            status_data = r.json()
            if not status_data.get("chain_ready"):
                print("❌ Backend up but RAG chain NOT ready — upload PDF first.")
                return False
            
            model_type = status_data.get("model", "unknown")
            if "local" in model_type.lower() or "tinyllama" in model_type.lower() or "os-tutor" in model_type.lower():
                print(f"✅ Backend is UP and running the LOCAL FINE-TUNED model: {model_type}")
            else:
                print(f"⚠ Backend is using model: {model_type}")
            
            return True
        except requests.exceptions.ConnectionError:
            print(f"   ⏳ Attempt {attempt}/5: not up yet, retrying in 5s...")
            time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"   ⏳ Attempt {attempt}/5: timeout (Qdrant slow?), retrying in 5s...")
            time.sleep(5)
    print("❌ Could not reach backend after 5 attempts. Run 'python api.py' first.")
    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("🚀 Initializing Evaluator...")

    # ── Paid API key for evaluation ───────────────────────────────────────────
    # paid_key = os.environ.get("GEMINI_PAID_API_KEY")
    # if not paid_key:
    #     print("❌ GEMINI_PAID_API_KEY not set in .env")
    #     return
    

    # ── LLM + Embeddings (raw LangChain — evaluate() wraps them automatically) ─
    # print("🤖 Loading Gemini 2.5 Flash (evaluator LLM)...")
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=paid_key,
        temperature=0,
        max_retries=10,
    )
    print("📐 Loading HuggingFace embeddings (BAAI/bge-small-en-v1.5)...")
    eval_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # ── Metrics — OLD MetricWithLLM-based classes (LangChain-compatible) ──────
    # evaluate() injects eval_llm automatically because metric.llm is None here
    metrics = [
        Faithfulness(),       # needs: user_input, response, retrieved_contexts
        AnswerRelevancy(),    # needs: user_input, response, retrieved_contexts
        ContextPrecision(),   # needs: user_input, retrieved_contexts, reference
    ]

    # ── Rate-limit protection ─────────────────────────────────────────────────
    safe_config = RunConfig(max_retries=10, max_workers=2, timeout=300)

    # ── Backend health check ──────────────────────────────────────────────────
    if not check_backend():
        return

    # ── Clear old results ─────────────────────────────────────────────────────
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    print(f"\n📥 Processing {len(DATASET)} questions in batches of {BATCH_SIZE}...")

    # ── Batch loop ────────────────────────────────────────────────────────────
    for i in range(0, len(DATASET), BATCH_SIZE):
        batch     = DATASET[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"\n--- Batch {batch_num} "
              f"({i+1} to {min(i+BATCH_SIZE, len(DATASET))}) ---")

        rag_samples  = []
        base_samples = []

        for item in batch:
            q  = item["question"]
            gt = item["ground_truth"]
            print(f"  Q: {q[:60]}...")

            # 1. RAG — answer + context from the backend
            rag_ans, rag_ctx = get_rag_response(q, use_rag=True)
            rag_samples.append({
                "user_input":         q,
                "response":           rag_ans,
                "retrieved_contexts": rag_ctx,
                "reference":          gt,
            })

            # 2. Baseline — same model BUT with NO context (use_rag=False)
            base_ans, _ = get_rag_response(q, use_rag=False)
            base_samples.append({
                "user_input":         q,
                "response":           base_ans,
                "retrieved_contexts": ["No retrieved context used (baseline mode)."],
                "reference":          gt,
            })

            time.sleep(1)  # avoid hammering APIs back-to-back

        # ── Score this batch ──────────────────────────────────────────────────
        print("  📊 Running ragas evaluation...")
        try:
            rag_result = evaluate(
                dataset=make_eval_dataset(rag_samples),
                metrics=metrics,
                llm=eval_llm,
                embeddings=eval_embeddings,
                run_config=safe_config,
            )
            base_result = evaluate(
                dataset=make_eval_dataset(base_samples),
                metrics=metrics,
                llm=eval_llm,
                embeddings=eval_embeddings,
                run_config=safe_config,
            )

            df_rag = rag_result.to_pandas();   df_rag["system"]  = "Fine-Tuned RAG"
            df_base = base_result.to_pandas(); df_base["system"] = "Fine-Tuned Baseline"
            df_combined = pd.concat([df_rag, df_base], ignore_index=True)

            write_header = not os.path.exists(CSV_PATH)
            df_combined.to_csv(CSV_PATH, mode="a", header=write_header, index=False)
            print(f"  ✅ Batch {batch_num} saved → {CSV_PATH}")

        except Exception as e:
            print(f"  ❌ Batch {batch_num} failed: {e}")
            import traceback; traceback.print_exc()

    # ── Final summary ─────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print("\n⚠ No results written — all batches failed. See errors above.")
        return

    df_all = pd.read_csv(CSV_PATH)
    metric_cols = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference"]
    available   = [c for c in metric_cols if c in df_all.columns]

    rag_avg  = df_all[df_all["system"] == "Fine-Tuned RAG"][available].mean()
    base_avg = df_all[df_all["system"] == "Fine-Tuned Baseline"][available].mean()

    def fmt(s, c):
        return f"{s[c]:.4f}" if c in s.index and pd.notna(s[c]) else "0.0000"

    # Pretty console summary
    print(f"\n{'='*65}")
    print("✅  EVALUATION COMPLETE")
    print(f"{'='*65}")
    print(f"{'System':<25} {'Faithfulness':>12} {'Ans Relevancy':>14} {'Ctx Precision':>14}")
    print(f"{'-'*65}")
    print(f"{'Fine-Tuned Baseline':<25} "
          f"{fmt(base_avg,'faithfulness'):>12} "
          f"{fmt(base_avg,'answer_relevancy'):>14} "
          f"{fmt(base_avg,'llm_context_precision_with_reference'):>14}")
    print(f"{'Fine-Tuned RAG':<25} "
          f"{fmt(rag_avg,'faithfulness'):>12} "
          f"{fmt(rag_avg,'answer_relevancy'):>14} "
          f"{fmt(rag_avg,'llm_context_precision_with_reference'):>14}")

    # Markdown report
    md = f"""# OS Tutor — Fine-Tuned Evaluation Report
**Model**: TinyLlama-1.1B + OS-tutor Adapter
**Questions**: {len(DATASET)}  |  **Library**: ragas 0.4.3  |  **Evaluator**: llama-3.3-70b-versatile

| System | Faithfulness | Answer Relevancy | Context Precision |
| :--- | :---: | :---: | :---: |
| **Fine-Tuned Baseline** | {fmt(base_avg,'faithfulness')} | {fmt(base_avg,'answer_relevancy')} | {fmt(base_avg,'llm_context_precision_with_reference')} |
| **Fine-Tuned RAG**      | {fmt(rag_avg,'faithfulness')}  | {fmt(rag_avg,'answer_relevancy')}  | {fmt(rag_avg,'llm_context_precision_with_reference')}  |

### Metric Definitions
- **Faithfulness** — Is the answer grounded in retrieved context? (0–1, ↑ = less hallucination)
- **Answer Relevancy** — Does the answer directly address the question? (0–1, ↑ = better)
- **Context Precision** — Were the most relevant chunks ranked highest? (0–1, ↑ = better)
"""
    with open("eval/evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n📄 CSV    → {CSV_PATH}")
    print(f"📄 Report → eval/evaluation_report.md")


if __name__ == "__main__":
    main()
