"""
decontam_check.py - find gold-bank questions that leak into the fine-tuning data.

For every gold question (eval/run_eval.py DATASET) it finds the most similar
question in dataset_v2_train + dataset_v2_val using
  (a) normalised exact match, and
  (b) cosine similarity of BAAI/bge-small-en-v1.5 embeddings to pick candidates, which an
      LLM then confirms as "same question" or not (embedding score alone over-flags).

Writes data/decontam/report.csv and prints a summary. Original dataset files are
never modified.

USAGE (from the project root):  python data/decontam_check.py
"""
import ast
import csv
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "decontam")
EXACT = 1.00        # normalised string equality
PARAPHRASE = 0.82   # cosine >= this -> candidate pair, sent to the LLM same-question check


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def load_gold():
    src = open(os.path.join(ROOT, "eval", "run_eval.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DATASET":
            return ast.literal_eval(node.value)
    raise RuntimeError("DATASET not found in eval/run_eval.py")


def load_train_questions():
    rows = []
    for split in ("train", "val"):
        path = os.path.join(ROOT, f"dataset_v2_{split}.jsonl")
        for i, line in enumerate(open(path, encoding="utf-8")):
            r = json.loads(line)
            q = r["messages"][1]["content"].split("Question:")[-1].strip()
            rows.append({"split": split, "idx": i, "category": r.get("category", ""), "question": q})
    return rows


def main():
    from sentence_transformers import SentenceTransformer

    gold = load_gold()
    train = load_train_questions()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    g = model.encode([x["question"] for x in gold], normalize_embeddings=True)
    t = model.encode([x["question"] for x in train], normalize_embeddings=True)
    sims = g @ t.T                                   # (gold, train) cosine

    train_norm = {}
    for j, x in enumerate(train):
        train_norm.setdefault(norm(x["question"]), []).append(j)

    # Embedding similarity alone cannot tell a duplicate from a topic neighbour
    # (bge-small scores same-domain questions >= 0.8), so candidate pairs are
    # confirmed by an LLM yes/no check ("same answer?"). Conservative: a YES counts as a leak.
    from dotenv import load_dotenv
    from groq import Groq
    load_dotenv(os.path.join(ROOT, ".env"))
    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=5)
    prompt = ("Two exam questions follow. Would a student who memorised the model answer to Question B "
              "be able to answer Question A with essentially the same answer? "
              "Reply with exactly one word: YES or NO.\n\nQuestion A: {a}\nQuestion B: {b}")
    cache = {}

    def same_question(a, b):
        if (a, b) not in cache:
            r = client.chat.completions.create(
                model="openai/gpt-oss-20b", temperature=0, max_tokens=400, reasoning_effort="low",
                messages=[{"role": "user", "content": prompt.format(a=a, b=b)}])
            cache[(a, b)] = (r.choices[0].message.content or "").strip().upper().startswith("YES")
        return cache[(a, b)]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = []
    for i, gq in enumerate(gold):
        j = int(np.argmax(sims[i]))
        s = float(sims[i, j])
        exact_rows = train_norm.get(norm(gq["question"]), [])
        cands = [int(k) for k in np.argsort(-sims[i])[:5] if sims[i, k] >= PARAPHRASE]
        confirmed = sorted(set(exact_rows) | {k for k in cands if k in exact_rows or same_question(gq["question"], train[k]["question"])})
        status = "EXACT" if exact_rows else ("LEAK" if confirmed else "clean")
        out.append({
            "gold_idx": i + 1, "chapter": gq["chapter"], "gold_question": gq["question"],
            "status": status, "max_cosine": round(s, 3),
            "closest_train_question": train[j]["question"], "closest_split": train[j]["split"],
            "n_train_rows_to_remove": len(confirmed), "_hits": confirmed,
        })

    with open(os.path.join(OUT_DIR, "report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in out[0] if not k.startswith("_")])
        w.writeheader()
        for r in out:
            w.writerow({k: v for k, v in r.items() if not k.startswith("_")})

    leaked = [r for r in out if r["status"] != "clean"]
    clean = [r for r in out if r["status"] == "clean"]
    json.dump({"leaked_gold_idx": [r["gold_idx"] for r in leaked],
               "clean_gold_idx": [r["gold_idx"] for r in clean],
               "leaked_questions": [r["gold_question"] for r in leaked],
               "candidate_cosine": PARAPHRASE},
              open(os.path.join(OUT_DIR, "leaked_gold.json"), "w", encoding="utf-8"), indent=2)

    print(f"gold={len(gold)}  train+val={len(train)}")
    for st in ("EXACT", "LEAK", "clean"):
        print(f"  {st:<11}{sum(r['status'] == st for r in out)}")
    print()
    for r in out:
        if r["status"] != "clean":
            print(f"  #{r['gold_idx']:<2} {r['status']:<10} cos={r['max_cosine']:.2f}  {r['gold_question'][:52]}")
            print(f"       ~ {r['closest_train_question'][:70]}  [{r['closest_split']}]")
    print("\nfirst 5 gold questions (the ones evaluated so far):")
    for r in out[:5]:
        print(f"  #{r['gold_idx']}  {r['status']:<10} cos={r['max_cosine']:.2f}  {r['gold_question'][:55]}")

    # cleaned training files (training-set rows that overlap ANY gold question removed)
    drop_train = sorted({h for r in out for h in r["_hits"] if train[h]["split"] == "train"})
    drop_val = sorted({h for r in out for h in r["_hits"] if train[h]["split"] == "val"})
    print(f"\ntrain rows overlapping a gold question: {len(drop_train)}   val rows: {len(drop_val)}")
    for split, drop in (("train", drop_train), ("val", drop_val)):
        base = [json.loads(l) for l in open(os.path.join(ROOT, f"dataset_v2_{split}.jsonl"), encoding="utf-8")]
        drop_idx = {train[h]["idx"] for h in drop}
        kept = [r for k, r in enumerate(base) if k not in drop_idx]
        with open(os.path.join(OUT_DIR, f"dataset_v2_{split}_decontam.jsonl"), "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(base)} -> {len(kept)}  (removed {len(base) - len(kept)})")
    print(f"\nwritten to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
