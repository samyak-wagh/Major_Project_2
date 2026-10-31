"""
num_bench.py - judge-free numerical-correctness benchmark.

Fresh numerical OS problems (fixed seed) whose ground truth is computed by the dataset's own
simulators (data/numerical_gen.py, which reproduce the textbook's classic examples: 15/12/9 page faults
for FIFO/LRU/OPT on the 20-page string). Models answer WITHOUT retrieval, so this measures the
calculation skill the dataset teaches, not RAG. Answers are checked automatically by number matching.

Categories (15 problems each = 45): page-replacement faults, disk head movement, CPU average waiting time.
Questions are re-worded (not the dataset's templates) and use fresh random inputs.

USAGE (project root; api.py running for `gen`):
  python eval/num_bench.py build
  python eval/num_bench.py gen        # S1 cloud, S2 fine-tuned, S5 base Qwen  -> eval/num_bench_answers.csv (resumable)
  python eval/num_bench.py score      # accuracy with 95% Wilson CIs, by category
"""
import json
import math
import os
import random
import re
import sys

import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from data.numerical_gen import (disk_fcfs, disk_scan_family, disk_sstf,   # noqa: E402
                                sim_fcfs, sim_fifo, sim_lru, sim_optimal, sim_rr, sim_sjf_np)

API_URL = "http://127.0.0.1:8001"
PROB = os.path.join("eval", "num_bench_problems.json")
ANS = os.path.join("eval", "num_bench_answers.csv")
SYSTEMS = {
    "S1 - Groq Only":      {"backend": "groq",  "qwen_opts": {}},
    "S2 - Qwen Only":      {"backend": "local", "qwen_opts": {}},
    "S5 - Base Qwen Only": {"backend": "local", "qwen_opts": {"adapter": "off"}},
}
N_PER = 5          # problems GENERATED per algorithm
# Problems actually USED: the first NB_PER_ALGO of each (category, algorithm) group, in the fixed-seed order
# (deterministic, no cherry-picking). Reduced from 5 to 3 (45 -> 27 problems) because local CPU generation is slow.
NB_PER_ALGO = int(os.environ.get("NB_PER_ALGO", 3))


def select(problems):
    seen, keep = {}, []
    for p in problems:
        k = (p["cat"], p["algo"])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= NB_PER_ALGO:
            keep.append(p)
    return keep


def build(seed=2026):
    rng = random.Random(seed)
    P = []
    for algo, fn in (("FIFO", sim_fifo), ("LRU", sim_lru), ("OPT", sim_optimal)):
        for _ in range(N_PER):
            frames = rng.randint(3, 4)
            ref = [rng.randint(0, 7) for _ in range(rng.randint(12, 15))]
            truth = fn(ref, frames)[1]
            name = {"FIFO": "FIFO", "LRU": "LRU", "OPT": "optimal (OPT)"}[algo]
            q = (f"Reference string: {' '.join(map(str, ref))}. The system has {frames} frames, all initially empty, "
                 f"and uses the {name} page replacement policy. How many page faults occur in total? Give the number.")
            P.append({"cat": "page_replacement", "algo": algo, "question": q, "truth": float(truth), "tol": 0})
    for algo in ("FCFS", "SSTF", "LOOK"):
        for _ in range(N_PER):
            head = rng.randint(20, 180)
            reqs = rng.sample(range(0, 200), rng.randint(6, 8))
            direction = rng.choice(["up", "down"])
            if algo == "FCFS":
                total = disk_fcfs(reqs, head)[1]
                tail = "Requests are served in arrival order."
            elif algo == "SSTF":
                total = disk_sstf(reqs, head)[1]
                tail = "Use shortest-seek-time-first."
            else:
                total = disk_scan_family(reqs, head, 200, direction, "LOOK")[1]
                tail = (f"Use LOOK, with the head initially moving toward "
                        f"{'higher' if direction == 'up' else 'lower'} cylinder numbers.")
            q = (f"Disk cylinders are numbered 0 to 199 and the head starts at cylinder {head}. Pending requests "
                 f"(in arrival order): {', '.join(map(str, reqs))}. {tail} What is the total head movement in "
                 f"cylinders? Give the number.")
            P.append({"cat": "disk_scheduling", "algo": algo, "question": q, "truth": float(total), "tol": 0})
    for algo in ("FCFS", "SJF", "RR"):
        for _ in range(N_PER):
            n = rng.randint(3, 5)
            bursts = rng.sample(range(2, 13), n)                      # distinct bursts: no SJF ties
            arrivals = [0] * n if algo == "RR" else [rng.randint(0, 6) for _ in range(n)]
            procs = [{"pid": f"P{i + 1}", "arrival": arrivals[i], "burst": bursts[i], "priority": 1} for i in range(n)]
            quantum = rng.choice([2, 3])
            if algo == "FCFS":
                comp = sim_fcfs(procs)[1]
                rule = "Scheduling is FCFS (first-come, first-served); arrival ties are broken by process number."
            elif algo == "SJF":
                comp = sim_sjf_np(procs)[1]
                rule = "Scheduling is non-preemptive shortest-job-first; arrival ties are broken by process number."
            else:
                comp = sim_rr(procs, quantum)[1]
                rule = f"Scheduling is round robin with time quantum {quantum}; all processes arrive at time 0 and queue in process-number order."
            wt = sum(comp[p["pid"]] - p["arrival"] - p["burst"] for p in procs) / n
            table = "; ".join(f"{p['pid']} (arrival {p['arrival']}, burst {p['burst']})" for p in procs)
            q = f"Processes: {table}. {rule} What is the average waiting time? Round to two decimal places."
            P.append({"cat": "cpu_scheduling", "algo": algo, "question": q, "truth": round(wt, 2), "tol": 0.011})
    for i, p in enumerate(P):
        p["id"] = i + 1
    json.dump(P, open(PROB, "w", encoding="utf-8"), indent=1)
    print("problems:", len(P), pd.Series([p["cat"] for p in P]).value_counts().to_dict())


def gen():
    P = select(json.load(open(PROB, encoding="utf-8")))
    print(f"problems used: {len(P)} ({NB_PER_ALGO} per algorithm)")
    done = set()
    if os.path.exists(ANS):
        a = pd.read_csv(ANS)
        done = set(zip(a.system, a.id))
    for p in P:
        for name, cfg in SYSTEMS.items():
            if (name, p["id"]) in done:
                continue
            payload = {"question": p["question"], "backend": cfg["backend"], "use_rag": False,
                       "stateless": True, "qwen_opts": cfg["qwen_opts"]}
            try:
                r = requests.post(f"{API_URL}/ask", json=payload, timeout=900)
                r.raise_for_status()
                ans = r.json()["answer"]
            except Exception as e:
                print("  [WARN]", name, p["id"], str(e)[:80])
                continue
            row = {"system": name, "id": p["id"], "cat": p["cat"], "algo": p["algo"], "response": ans}
            pd.DataFrame([row]).to_csv(ANS, mode="a", header=not os.path.exists(ANS), index=False)
            print(f"  {name:<20} #{p['id']:<3} {p['cat']:<17} {len(ans):>5} chars", flush=True)


ERR_RE = re.compile(r"^\s*(Groq API error|Error code: \d|Error fetching answer|⚠)", re.I)


def regen():
    """Retry empty / API-error answers (a failed generation is not an answer). Up to 5 attempts each."""
    P = {p["id"]: p for p in select(json.load(open(PROB, encoding="utf-8")))}
    a = pd.read_csv(ANS)
    for i, r in a.iterrows():
        txt = "" if pd.isna(r["response"]) else str(r["response"])
        if r["id"] not in P or (txt.strip() and not ERR_RE.search(txt)):
            continue
        cfg = SYSTEMS[r["system"]]
        for attempt in range(1, 6):
            payload = {"question": P[r["id"]]["question"], "backend": cfg["backend"], "use_rag": False,
                       "stateless": True, "qwen_opts": cfg["qwen_opts"]}
            try:
                out = requests.post(f"{API_URL}/ask", json=payload, timeout=900).json()["answer"]
            except Exception as e:
                out = ""
                print("  [WARN]", str(e)[:80])
            ok = bool(str(out).strip()) and not ERR_RE.search(str(out))
            print(f"  regen {r['system']} #{r['id']} attempt {attempt}: {len(str(out))} chars {'ok' if ok else 'FAILED'}")
            if ok:
                a.loc[i, "response"] = out
                break
    a.to_csv(ANS, index=False)


NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")


def numbers(text):
    return [float(x.replace(",", "")) for x in NUM.findall(str(text))]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def score():
    P = {p["id"]: p for p in select(json.load(open(PROB, encoding="utf-8")))}
    a = pd.read_csv(ANS)
    a = a[a.id.isin(P)]                      # only the selected problems (extra answers from the 45-problem start are ignored)
    ERR = re.compile(r"^\s*(Groq API error|Error code: \d|⚠)", re.I)
    rows = []
    for _, r in a.iterrows():
        p = P[r["id"]]
        txt = "" if pd.isna(r["response"]) else str(r["response"])
        nums = numbers(txt)
        tol = p["tol"]
        strict = bool(nums) and abs(nums[-1] - p["truth"]) <= tol          # last number in the answer
        lenient = any(abs(x - p["truth"]) <= tol for x in numbers(txt[-250:]))   # appears in the final 250 chars
        rows.append({"system": r["system"], "id": r["id"], "cat": r["cat"], "strict": strict, "lenient": lenient,
                     "api_error": bool(ERR.search(txt))})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join("eval", "num_bench_scored.csv"), index=False)
    print(f"API-error answers (excluded from nothing; counted wrong): {int(d.api_error.sum())}")
    for label, sub in [("ALL", d)] + [(c, d[d.cat == c]) for c in sorted(d.cat.unique())]:
        print(f"\n=== {label} ===   (strict = last number equals truth; lenient = truth appears in final 250 chars)")
        for s in SYSTEMS:
            g = sub[sub.system == s]
            if g.empty:
                continue
            out = []
            for m in ("strict", "lenient"):
                p, lo, hi = wilson(int(g[m].sum()), len(g))
                out.append(f"{m} {p:.2f} [{lo:.2f},{hi:.2f}]")
            print(f"  {s:<20} n={len(g):<3} " + "   ".join(out))


if __name__ == "__main__":
    {"build": build, "gen": gen, "regen": regen, "score": score}[sys.argv[1]]()
