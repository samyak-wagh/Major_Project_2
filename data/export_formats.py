"""
export_formats.py — Export dataset_v2*.json into CSV and JSONL as well.
==========================================================================
The chat-messages JSON schema (dataset_v2*.json) is the canonical format —
it's what most modern chat/instruction fine-tuning frameworks (axolotl,
LLaMA-Factory, Unsloth, HF SFTTrainer, OpenAI's own fine-tuning API) expect
natively. This script additionally exports:

  - dataset_v2*.jsonl  — the same examples, one JSON object per line
                          (the more common file extension for this format;
                          required by some tools instead of a single array)
  - dataset_v2*.csv    — flattened to columns: system, prompt, completion
                          (for tools/UIs that only accept a flat table)

Run after build_dataset.py:
    python export_formats.py
"""

import csv
import json
from pathlib import Path

SOURCES = ["dataset_v2", "dataset_v2_train", "dataset_v2_val"]


def export_one(base_name: str, root: Path):
    json_path = root / f"{base_name}.json"
    if not json_path.exists():
        print(f"  skip {json_path.name} (not found)")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # ── JSONL ────────────────────────────────────────────────────────
    jsonl_path = root / f"{base_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ── CSV (category, difficulty, system, prompt, completion) ─────────
    csv_path = root / f"{base_name}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "difficulty", "system", "prompt", "completion"])
        for ex in data:
            msgs = {m["role"]: m["content"] for m in ex["messages"]}
            writer.writerow([
                ex.get("category", ""),
                ex.get("difficulty", ""),
                msgs.get("system", ""),
                msgs.get("user", ""),
                msgs.get("assistant", ""),
            ])

    print(f"  {base_name}: {len(data)} rows -> {jsonl_path.name}, {csv_path.name}")


def main():
    root = Path(__file__).parent / ".."
    print("Exporting CSV + JSONL alongside the JSON files...")
    for name in SOURCES:
        export_one(name, root)


if __name__ == "__main__":
    main()
