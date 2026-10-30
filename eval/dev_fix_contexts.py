"""
dev_fix_contexts.py - fill the empty `contexts` column of eval/dev_answers.csv.

The dev answers were generated while api.py returned empty contexts for backend="local"
(see api.py fmt_chain fix). Retrieval is deterministic for a given question, so the passages
the model saw can be recovered by re-running the same retriever; no answers are regenerated.
"""
import json
import os
import warnings

import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

import sys                                                      # noqa: E402
sys.path.insert(0, os.getcwd())                                 # run from the project root
from main import init_components, load_config                   # noqa: E402

cfg = load_config()
store, emb, _ = init_components(cfg)
query_emb = emb.get_query_embeddings_instance()
retriever = store.get_vector_store(query_emb).as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": cfg["top_k"], "score_threshold": cfg["score_threshold"]},
)

p = os.path.join("eval", "dev_answers.csv")
d = pd.read_csv(p)
cache = {}
for q in d.question.unique():
    for attempt in range(3):
        docs = retriever.invoke(q)
        if docs:
            break
    cache[q] = [x.page_content for x in docs]
    print(f"{len(cache[q])} passages | {q[:60]}")
d["contexts"] = d.question.map(lambda q: json.dumps(cache[q]))
d.to_csv(p, index=False)
print("empty-context rows now:", sum(len(json.loads(c)) == 0 for c in d.contexts), "of", len(d))
