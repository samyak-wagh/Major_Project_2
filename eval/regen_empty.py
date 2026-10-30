"""
regen_empty.py - regenerate EMPTY answers of the Groq systems in an answers CSV.

An empty completion is a generation failure (e.g. the reasoning model used its whole token
budget), not an answer. Retries up to 3 times with a fresh, memory-less chain; a backup of the
file is written first. Rows that stay empty are left empty (the scorer gives them 0 and flags
them). Refusals such as OUT_OF_CONTEXT are real behaviour and are NOT touched.

USAGE (project root):  python eval/regen_empty.py eval/answers_<timestamp>.csv
"""
import os
import shutil
import sys
import warnings

import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()
sys.path.insert(0, os.getcwd())
from main import build_qa_chain, init_components, load_config   # noqa: E402

path = sys.argv[1]
d = pd.read_csv(path)
import re  # noqa: E402
ERROR_ANSWER = re.compile(r"^\s*(Groq API error|Error code: \d|Error fetching answer|Error: API server|⚠)", re.I)
# "failed" = empty completion OR an API error message that was stored as the answer
empty = d.index[d.response.isna() | (d.response.astype(str).str.strip() == "")
                | d.response.astype(str).str.contains(ERROR_ANSWER)].tolist()
print("empty rows:", [(i, d.loc[i, "system"], d.loc[i, "user_input"][:40]) for i in empty])
if not empty:
    sys.exit(0)

shutil.copy(path, path.replace(".csv", ".before_regen.csv"))
cfg = load_config()
store, emb, _ = init_components(cfg)
chain = build_qa_chain(cfg, store, emb, backend="groq")

for i in empty:
    r = d.loc[i]
    if "Groq" not in r.system:
        print("  skip (not a Groq system):", r.system)
        continue
    use_rag = "RAG" in r.system
    for attempt in range(1, 9):
        chain.clear_memory()
        out = chain.ask(r.user_input, use_rag=use_rag)
        ans = (out.get("answer") or "").strip()
        failed = (not ans) or bool(ERROR_ANSWER.search(ans))
        print(f"  row {i} attempt {attempt}: {len(ans)} chars {'(FAILED - API error/empty)' if failed else 'ok'}", flush=True)
        if failed:
            import time
            time.sleep(150)          # daily-quota errors clear gradually; wait and retry
            continue
        if ans:
            d.loc[i, "response"] = ans
            if use_rag:
                d.loc[i, "retrieved_contexts"] = str([x.page_content for x in out.get("source_documents", [])])
            break
d.to_csv(path, index=False)
print("remaining empty:", int((d.response.isna() | (d.response.astype(str).str.strip() == "")).sum()))
