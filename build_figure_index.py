"""
build_figure_index.py
---------------------
Scans the OS textbook PDF and builds figure_index.json:
  { "figure 1.4": 10, "figure 9.18": 374, ... }

Run once:
  python build_figure_index.py
"""

import json
import re
import pymupdf  # PyMuPDF

PDF_PATH = "ostxtbook.pdf"
OUT_PATH = "figure_index.json"

# Match "Figure X.Y" that is at the start of a line (or string),
# followed optionally by whitespace, and then either End-Of-Line or a Capital letter.
# This strictly matches actual image captions and ignores inline text (like "Figure 1.4 shows...").
FIG_RE = re.compile(r'(?:^|\n)\s*Figure\s+(\d+\.\d+)(?:\s*$|\s*\n|\s+[A-Z])')


def build_index():
    print(f"Opening {PDF_PATH} ...")
    doc = pymupdf.open(PDF_PATH)
    index = {}

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        for m in FIG_RE.finditer(text):
            key = f"figure {m.group(1).lower()}"
            if key not in index:
                index[key] = page_num
                print(f"  Found {key!r} -> page {page_num}")

    doc.close()

    with open(OUT_PATH, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"\nDone! {len(index)} figures indexed -> {OUT_PATH}")
    return index


if __name__ == "__main__":
    build_index()
