"""
score_answers.py - judge saved answers (from `GEN_ONLY=1 python eval/run_eval.py`) with RAGAS.

  * Resumable: a (system, question) row is written only when every applicable metric was
    scored; re-running scores whatever is still missing.
  * Stops cleanly when the judge's daily token quota runs out (rows would come back NaN).
  * Question-major order, so a partial run still compares all systems on the same questions.
  * `summary` prints the paper table: mean with 95% bootstrap CI over questions, n, and the
    paired S4-S3 difference.

USAGE (project root):
  python eval/score_answers.py eval/answers_<timestamp>.csv           # score (resumes)
  python eval/score_answers.py eval/answers_<timestamp>.csv summary   # table only
"""
import ast
import os
import sys
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

JUDGE = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
M_FULL = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference"]
M_NORAG = ["answer_relevancy"]
COLS = ["system", "user_input", "chapter", "faithfulness", "answer_relevancy",
        "llm_context_precision_with_reference", "empty_answer"]


import re  # noqa: E402
ERROR_ANSWER = re.compile(r"^\s*(Groq API error|Error code: \d|Error fetching answer|Error: API server|⚠)", re.I)


def is_rag(system: str) -> bool:
    return "RAG" in system


def needed(system: str):
    return M_FULL if is_rag(system) else M_NORAG


def score(ans_path: str, out_path: str):
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference as CP
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    a = pd.read_csv(ans_path)
    done = set()
    if os.path.exists(out_path):
        o = pd.read_csv(out_path)
        done = set(zip(o.system, o.user_input))
    extra = {"reasoning_effort": "low"} if "gpt-oss" in JUDGE else {}
    # Keys from separate Groq accounts have separate daily budgets; rotate when one runs dry.
    keys = [v for k, v in sorted(os.environ.items()) if k.startswith("GROQ_API_KEY") and v.strip()]
    state = {"i": 0}

    def make_llm():
        return LangchainLLMWrapper(ChatGroq(model=JUDGE, api_key=keys[state["i"] % len(keys)], temperature=0,
                                            max_retries=3, max_tokens=4096, **extra), bypass_n=True)
    llm = make_llm()
    print(f"Groq keys available: {len(keys)}")
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    cfg = RunConfig(max_retries=3, max_workers=1, timeout=300)
    pending = a[[(s, q) not in done for s, q in zip(a.system, a.user_input)]]
    print(f"answers: {len(a)}   already scored: {len(done)}   to score: {len(pending)}   judge: {JUDGE}")

    consecutive_fail = 0
    for _, r in pending.iterrows():          # answers file is question-major
        if ERROR_ANSWER.search(str(r.response)):
            # An API error message was stored as the "answer" (e.g. Groq 429 daily quota).
            # Never score it: regenerate with eval/regen_empty.py first.
            print(f"  [SKIP failed generation] {r.system:<18} {r.user_input[:42]}")
            continue
        if pd.isna(r.response) or not str(r.response).strip():
            # No answer was produced (generation failure) -> no credit, flagged in the output.
            vals = {m: 0.0 for m in needed(r.system)}
            row = {"system": r.system, "user_input": r.user_input, "chapter": r.chapter, **vals, "empty_answer": True}
            pd.DataFrame([row]).reindex(columns=COLS).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
            print(f"  ok {r.system:<18} {r.user_input[:42]:<42} EMPTY ANSWER -> 0.0")
            continue
        ctx = ast.literal_eval(r.retrieved_contexts) if str(r.retrieved_contexts).startswith("[") else [str(r.retrieved_contexts)]
        ds = EvaluationDataset(samples=[SingleTurnSample(user_input=r.user_input, response=r.response,
                                                         retrieved_contexts=ctx, reference=r.reference)])
        metrics = [Faithfulness(), AnswerRelevancy(), CP()] if is_rag(r.system) else [AnswerRelevancy()]
        vals = None
        for attempt in range(len(keys)):                 # try each key at most once per row
            try:
                res = evaluate(ds, metrics, llm=llm, embeddings=emb, run_config=cfg, show_progress=False).to_pandas().iloc[0]
                v = {m: res.get(m) for m in needed(r.system)}
                if not any(pd.isna(x) for x in v.values()):
                    vals = v
                    break
                print(f"  [incomplete] {r.system} | {r.user_input[:45]} (key #{state['i'] % len(keys) + 1})")
            except Exception as e:
                print("  [ERR]", r.system, str(e)[:100])
            if len(keys) > 1:                            # quota most likely exhausted on this key
                state["i"] += 1
                llm = make_llm()
                print(f"  -> switching to Groq key #{state['i'] % len(keys) + 1}")
        if vals is None:
            consecutive_fail += 1
            if consecutive_fail >= 2:
                print("\n  Rows failing on every key -> all daily quotas likely exhausted. Stopping; re-run later to resume.")
                break
            continue
        consecutive_fail = 0
        row = {"system": r.system, "user_input": r.user_input, "chapter": r.chapter, **vals}
        pd.DataFrame([row]).reindex(columns=COLS).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print(f"  ok {r.system:<18} {r.user_input[:42]:<42} " + " ".join(f"{k[:5]}={v:.2f}" for k, v in vals.items()), flush=True)


def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    return (x.mean(), *np.percentile(means, [2.5, 97.5]), len(x))


def summary(out_path: str):
    s = pd.read_csv(out_path)
    subsets = {"ALL": s,
               "conceptual": s[~s.chapter.str.contains("Numerical")],
               "numerical": s[s.chapter.str.contains("Numerical")]}
    for name, sub in subsets.items():
        if sub.empty:
            continue
        print(f"\n=== {name}  (questions: {sub.user_input.nunique()}) ===")
        print(f"{'System':<20}{'Faithfulness':>26}{'Ans.Relevancy':>26}{'Ctx.Precision':>26}")
        for sysname in sorted(sub.system.unique()):
            g = sub[sub.system == sysname]
            cells = []
            for m in M_FULL:
                if g[m].notna().any():
                    mu, lo, hi, n = boot_ci(g[m])
                    cells.append(f"{mu:.3f} [{lo:.3f},{hi:.3f}] n={n}")
                else:
                    cells.append("n/a")
            print(f"{sysname:<20}" + "".join(f"{c:>26}" for c in cells))
        s3 = [x for x in sub.system.unique() if x.startswith("S3")]
        s4 = [x for x in sub.system.unique() if x.startswith("S4")]
        if s3 and s4:
            for m in M_FULL:
                p = sub.pivot_table(index="user_input", columns="system", values=m).dropna(subset=[s3[0], s4[0]])
                if len(p):
                    mu, lo, hi, n = boot_ci(p[s4[0]] - p[s3[0]])
                    verdict = "includes 0 -> not distinguishable" if lo <= 0 <= hi else "excludes 0"
                    print(f"  paired S4-S3 {m[:22]:<22}: {mu:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}  ({verdict})")


if __name__ == "__main__":
    ans = sys.argv[1]
    out = os.path.join(os.path.dirname(ans), "scores_" + os.path.basename(ans).replace("answers_", ""))
    if len(sys.argv) < 3 or sys.argv[2] != "summary":
        score(ans, out)
    if os.path.exists(out):
        summary(out)
