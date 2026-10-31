"""
dev_tune.py - compare S4 (RAG + fine-tuned Qwen) generation variants on a DEV set.

Why a dev set: tuning prompt/decoding on the gold bank would overfit the numbers we later
report. The dev questions come from the decontaminated validation split
(data/decontam/dataset_v2_val_decontam.jsonl) and are never used for the final test.

Variants (all RAG, same retriever, stateless):
  base        concise inference prompt, sampling (current behaviour)
  trainp      prompt the adapter was fine-tuned with
  greedy      concise prompt, deterministic decoding
  trainp+gr   training prompt + deterministic decoding

USAGE (api.py running on :8001):
  python eval/dev_tune.py gen      # generate answers -> eval/dev_answers.csv (resumable)
  python eval/dev_tune.py score    # judge them     -> eval/dev_scores.csv + summary
"""
import ast
import json
import os
import sys
import time
import warnings

import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

API_URL = "http://127.0.0.1:8001"
N_DEV = int(os.environ.get("N_DEV", 12))
DEV_JUDGE = os.environ.get("DEV_JUDGE", "openai/gpt-oss-120b")  # qwen3.8-27b has a 1000 output-tokens/min cap
SCORE_ROWS = int(os.environ.get("SCORE_ROWS", 24))              # rows are question-major: 24 = first 6 questions x 4 variants
ANS_CSV = os.path.join("eval", "dev_answers.csv")
SCORE_CSV = os.path.join("eval", "dev_scores.csv")

VARIANTS = {
    "base":      {},
    "trainp":    {"prompt": "train"},
    "greedy":    {"greedy": True},
    "trainp+gr": {"prompt": "train", "greedy": True},
}


def dev_questions():
    rows = [json.loads(l) for l in open(
        os.path.join("data", "decontam", "dataset_v2_val_decontam.jsonl"), encoding="utf-8")]
    out = []
    for r in rows:
        if r.get("category") == "applied_numerical":
            continue                                    # numericals need worked arithmetic, not retrieval
        q = r["messages"][1]["content"].split("Question:")[-1].strip()
        if len(q) <= 160:                               # skip long scenario prompts
            out.append({"question": q, "reference": r["messages"][2]["content"], "category": r["category"]})
    return out[:N_DEV]


def ask(q, opts):
    payload = {"question": q, "backend": "local", "use_rag": True, "stateless": True, "qwen_opts": opts}
    res = requests.post(f"{API_URL}/ask", json=payload, timeout=900)
    res.raise_for_status()
    d = res.json()
    return d["answer"], d.get("contexts", [])


def gen():
    qs = dev_questions()
    done = set()
    if os.path.exists(ANS_CSV):
        d = pd.read_csv(ANS_CSV)
        done = set(zip(d.variant, d.question))
    print(f"dev questions: {len(qs)}   variants: {list(VARIANTS)}   already done: {len(done)}")
    for item in qs:
        for name, opts in VARIANTS.items():
            if (name, item["question"]) in done:
                continue
            t0 = time.time()
            try:
                ans, ctx = ask(item["question"], opts)
            except Exception as e:
                print(f"  [WARN] {name} | {item['question'][:40]} -> {e}")
                continue
            row = {"variant": name, "question": item["question"], "reference": item["reference"],
                   "category": item["category"], "response": ans, "contexts": json.dumps(ctx),
                   "seconds": round(time.time() - t0, 1)}
            pd.DataFrame([row]).to_csv(ANS_CSV, mode="a", header=not os.path.exists(ANS_CSV), index=False)
            print(f"  {name:<10} {len(ans):>5} chars {row['seconds']:>6.1f}s | {item['question'][:50]}", flush=True)
    print("generation done ->", ANS_CSV)


def score():
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    d = pd.read_csv(ANS_CSV).head(SCORE_ROWS)
    extra = {"reasoning_effort": "low"} if "gpt-oss" in DEV_JUDGE else {}   # only gpt-oss takes "low"
    llm = LangchainLLMWrapper(ChatGroq(model=DEV_JUDGE, api_key=os.environ["GROQ_API_KEY"],
                                       temperature=0, max_retries=10, max_tokens=4096,
                                       **extra), bypass_n=True)
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    cfg = RunConfig(max_retries=10, max_workers=1, timeout=300)
    scored = pd.read_csv(SCORE_CSV) if os.path.exists(SCORE_CSV) else pd.DataFrame()
    have = set(zip(scored.variant, scored.question)) if len(scored) else set()
    for i, r in d.iterrows():
        if (r.variant, r.question) in have:
            continue
        ctx = json.loads(r.contexts) or ["No context retrieved."]
        ds = EvaluationDataset(samples=[SingleTurnSample(user_input=r.question, response=r.response,
                                                         retrieved_contexts=ctx, reference=r.reference)])
        try:
            res = evaluate(ds, [Faithfulness(), AnswerRelevancy()], llm=llm, embeddings=emb,
                           run_config=cfg, show_progress=False).to_pandas().iloc[0]
            row = {"variant": r.variant, "question": r.question,
                   "faithfulness": res.get("faithfulness"), "answer_relevancy": res.get("answer_relevancy"),
                   "chars": len(r.response)}
        except Exception as e:
            print("  [FAIL]", r.variant, r.question[:40], str(e)[:80])
            continue
        pd.DataFrame([row]).to_csv(SCORE_CSV, mode="a", header=not os.path.exists(SCORE_CSV), index=False)
        print(f"  {r.variant:<10} F={row['faithfulness']} AR={row['answer_relevancy']}", flush=True)
    s = pd.read_csv(SCORE_CSV)
    print("\nDEV SUMMARY (mean; n per variant below)")
    print(s.groupby("variant")[["faithfulness", "answer_relevancy", "chars"]].mean().round(3).to_string())
    print(s.groupby("variant").size().to_string())


if __name__ == "__main__":
    {"gen": gen, "score": score}[sys.argv[1]]()
