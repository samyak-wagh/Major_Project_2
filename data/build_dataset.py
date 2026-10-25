"""
build_dataset.py — Build the final OS-Tutor fine-tuning dataset.
===================================================================
Combines:
  1. Curated conceptual examples (build_conceptual.py, from kb_conceptual.py)
  2. Correctness-guaranteed numerical examples (numerical_gen.py) — CPU
     scheduling, page replacement, disk scheduling, Banker's algorithm,
     memory addressing.

...then shuffles (fixed seed, reproducible) and splits into train/val.

Usage:
    python build_dataset.py                     # defaults: 400 numericals/category
    python build_dataset.py --per-category 800   # bigger numerical set
    python build_dataset.py --seed 7 --val-frac 0.1

Output (written to the repo root, next to the old dataset.json):
    dataset_v2.json         — full combined dataset (all examples)
    dataset_v2_train.json   — training split
    dataset_v2_val.json     — validation split
"""

import argparse
import json
import random
from pathlib import Path

from build_conceptual import build_all as build_conceptual_examples
from numerical_gen import GENERATORS


# Numericals are inherently 'applied' (a scenario + specific data the model
# must apply a known procedure to), tagged 'applied_numerical' specifically
# (not just 'applied') to keep them distinguishable downstream from
# build_conceptual.py's 'applied_conceptual' examples -- the two come from
# entirely different pipelines (simulation vs hand-authored KB) and
# collapsing them into one label made it impossible to split them apart
# later without re-deriving the split via content heuristics. Difficulty is
# a rough per-category heuristic based on how many steps/variables are
# typically involved. Floor is 'medium' -- category='applied_numerical' can
# never be 'easy' (enforced by build_conceptual.py's _make_example guard),
# since even the simplest numerical (memory addressing arithmetic) still
# requires applying a procedure to a specific scenario, not bare recall.
NUMERICAL_DIFFICULTY = {
    "memory_addressing": "medium",
    "page_replacement": "medium",
    "disk_scheduling": "medium",
    "cpu_scheduling": "medium",
    "bankers": "hard",
}


def build_numerical_examples(rng: random.Random, per_category: int) -> list:
    from build_conceptual import _make_example  # reuse the exact wrapper

    examples = []
    seen_questions = set()
    for name, gen_fn in GENERATORS.items():
        made = 0
        attempts = 0
        difficulty = NUMERICAL_DIFFICULTY.get(name, "medium")
        # Generate more than needed and de-dup on the rendered question text,
        # since random parameters can occasionally repeat.
        while made < per_category and attempts < per_category * 3:
            attempts += 1
            context, question, answer = gen_fn(rng)
            if question in seen_questions:
                continue
            seen_questions.add(question)
            examples.append(_make_example(context, question, answer, "applied_numerical", difficulty))
            made += 1
        print(f"  {name:20s}: {made} / {per_category} requested")
    return examples


def main():
    ap = argparse.ArgumentParser(description="Build the OS-Tutor fine-tuning dataset.")
    ap.add_argument("--per-category", type=int, default=400,
                     help="Numerical examples to generate per category (5 categories total).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (reproducibility).")
    ap.add_argument("--val-frac", type=float, default=0.08, help="Fraction held out for validation.")
    ap.add_argument("--out-dir", type=str, default="..", help="Output directory (default: repo root).")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("Building conceptual examples...")
    conceptual = build_conceptual_examples(seed=args.seed)
    print(f"  conceptual total: {len(conceptual)}")

    print("Building numerical examples...")
    numerical = build_numerical_examples(rng, args.per_category)
    print(f"  numerical total: {len(numerical)}")

    all_examples = conceptual + numerical
    rng.shuffle(all_examples)

    n_val = int(len(all_examples) * args.val_frac)
    val_set = all_examples[:n_val]
    train_set = all_examples[n_val:]

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "dataset_v2.json"
    train_path = out_dir / "dataset_v2_train.json"
    val_path = out_dir / "dataset_v2_val.json"

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, indent=2, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2, ensure_ascii=False)

    print()
    print(f"Total examples : {len(all_examples)}")
    print(f"  train         : {len(train_set)}")
    print(f"  val           : {len(val_set)}")
    print()
    print(f"Wrote: {full_path}")
    print(f"Wrote: {train_path}")
    print(f"Wrote: {val_path}")


if __name__ == "__main__":
    main()
