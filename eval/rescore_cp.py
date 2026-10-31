"""
rescore_cp.py - re-score ONLY Context Precision for rows whose gold reference answer was corrected.

The three page-fault gold questions (Frames=3, FIFO/LRU/OPT, ref string 7 0 1 2 0 3 0 4 2 3 0 3 2) had wrong
reference totals (9/8/6 instead of 10/9/7). Context Precision compares the retrieved passages with the
reference, so it must be recomputed; Faithfulness and Answer Relevancy do not use the reference and are kept.

USAGE: python eval/rescore_cp.py eval/answers_<ts>.csv   (reads/updates eval/scores_<ts>.csv in place)
"""
import ast
import os
import sys
import warnings

import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

from langchain_groq import ChatGroq                                   # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings               # noqa: E402
from ragas import evaluate                                            # noqa: E402
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.llms import LangchainLLMWrapper                            # noqa: E402
from ragas.metrics._context_precision import LLMContextPrecisionWithReference as CP   # noqa: E402
from ragas.run_config import RunConfig                                # noqa: E402

CP_COL = "llm_context_precision_with_reference"
ans_path = sys.argv[1]
out_path = os.path.join(os.path.dirname(ans_path), "scores_" + os.path.basename(ans_path).replace("answers_", ""))
a = pd.read_csv(ans_path)
s = pd.read_csv(out_path)
targets = a[a.user_input.str.match(r"^Frames=3, (FIFO|LRU|OPT)") & a.system.str.contains("RAG")]
keys = [v for k, v in sorted(os.environ.items()) if k.startswith("GROQ_API_KEY") and v.strip()]
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
JUDGE = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
extra = {"reasoning_effort": "low"} if "gpt-oss" in JUDGE else {}
ki = 0
for _, r in targets.iterrows():
    ctx = ast.literal_eval(r.retrieved_contexts)
    ds = EvaluationDataset(samples=[SingleTurnSample(user_input=r.user_input, response=r.response,
                                                     retrieved_contexts=ctx, reference=r.reference)])
    val = None
    for _try in range(len(keys)):
        llm = LangchainLLMWrapper(ChatGroq(model=JUDGE, api_key=keys[ki % len(keys)], temperature=0,
                                           max_retries=3, max_tokens=4096, **extra), bypass_n=True)
        try:
            res = evaluate(ds, [CP()], llm=llm, embeddings=emb, run_config=RunConfig(max_retries=3, max_workers=1, timeout=300),
                           show_progress=False).to_pandas().iloc[0]
            if not pd.isna(res.get(CP_COL)):
                val = float(res[CP_COL])
                break
        except Exception as e:
            print("  [ERR]", str(e)[:80])
        ki += 1
    mask = (s.system == r.system) & (s.user_input == r.user_input)
    old = s.loc[mask, CP_COL].iloc[0]
    print(f"{r.system:<16} {r.user_input[:30]:<30} CP {old} -> {val}")
    if val is not None:
        s.loc[mask, CP_COL] = val
s.to_csv(out_path, index=False)
print("done")
