"""
image_search.py — Keyword-based Image Retrieval for OS Tutor
=============================================================
Searches the metadata.json built by pdf_image_extractor.py to find
the most relevant images for a user query.

No ML/embeddings needed — uses TF-IDF-style keyword scoring over
image captions and figure labels extracted directly from the PDF.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Trigger words that indicate the user wants an image/diagram
IMAGE_INTENT_KEYWORDS = {
    "image", "diagram", "figure", "show", "draw", "picture",
    "illustration", "chart", "graph", "table", "visual",
    "display", "depict", "sketch", "layout", "structure",
    "architecture", "schematic",
}

# OS-specific boosted terms (appear in textbook figures)
OS_FIGURE_TERMS = {
    "page table", "inverted", "tlb", "paging", "segmentation",
    "process", "pcb", "scheduling", "memory", "frame", "disk",
    "file system", "inode", "deadlock", "banker", "semaphore",
    "thread", "kernel", "boot", "interrupt", "cache", "virtual",
}


def has_image_intent(query: str) -> bool:
    """Return True if the user's query is asking for a visual/diagram."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in IMAGE_INTENT_KEYWORDS)


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alpha, filter short tokens."""
    return [w for w in re.findall(r"[a-zA-Z]{3,}", text.lower())]


def _score(query_tokens: List[str], record: Dict) -> float:
    """
    Score an image record against the query tokens.
    Higher score = better match.
    """
    score = 0.0
    searchable = (
        record.get("caption", "") + " " +
        record.get("figure_label", "") + " " +
        " ".join(record.get("keywords", []))
    ).lower()

    for token in query_tokens:
        if token in searchable:
            # Bonus for figure label match (most reliable)
            if token in record.get("figure_label", "").lower():
                score += 3.0
            # Bonus for caption match
            count = searchable.count(token)
            score += 1.0 + math.log1p(count)

    # Penalize very tiny images (likely decorative)
    area = record.get("width", 100) * record.get("height", 100)
    if area < 10000:   # 100x100 pixels
        score *= 0.3

    return score


class ImageSearchEngine:
    """
    Lightweight keyword search over PDF image metadata.

    Usage
    -----
    engine = ImageSearchEngine("images/ostxtbook/metadata.json")
    results = engine.search("inverted page table", top_k=3)
    """

    def __init__(self, metadata_path: str):
        self.metadata_path = Path(metadata_path)
        self._records: List[Dict] = []
        self._loaded = False

    def _load(self):
        """Lazy-load metadata from disk."""
        if self._loaded:
            return
        if not self.metadata_path.exists():
            logger.warning(f"Image metadata not found: {self.metadata_path}")
            self._records = []
            self._loaded = True
            return
        with open(self.metadata_path, encoding="utf-8") as f:
            self._records = json.load(f)
        logger.info(f"Loaded {len(self._records)} image records from {self.metadata_path}")
        self._loaded = True

    def is_ready(self) -> bool:
        """Return True if the metadata file exists and has records."""
        self._load()
        return len(self._records) > 0

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for images matching the query.

        Returns list of top_k records sorted by relevance score,
        each with an added 'score' key.
        """
        self._load()
        if not self._records:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for record in self._records:
            s = _score(query_tokens, record)
            if s > 0:
                scored.append({**record, "score": round(s, 3)})

        # Sort by score descending
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def reload(self):
        """Force reload metadata from disk (call after new PDF ingestion)."""
        self._loaded = False
        self._load()


# ── Module-level singleton helpers ──────────────────────────────────────────

_engines: Dict[str, ImageSearchEngine] = {}


def get_engine(metadata_path: str) -> ImageSearchEngine:
    """Return a cached ImageSearchEngine for the given metadata file."""
    if metadata_path not in _engines:
        _engines[metadata_path] = ImageSearchEngine(metadata_path)
    return _engines[metadata_path]


def find_images_for_query(
    query: str,
    images_dir: str = "images",
    pdf_stem: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict]:
    """
    Convenience function: search all PDFs in images_dir for matching images.

    Parameters
    ----------
    query     : user's question
    images_dir: root directory containing per-PDF subdirectories
    pdf_stem  : if specified, only search that PDF's images
    top_k     : number of results to return

    Returns
    -------
    List of image metadata dicts, sorted by relevance.
    """
    images_root = Path(images_dir)
    if not images_root.exists():
        return []

    all_results = []

    # Find all metadata.json files
    if pdf_stem:
        search_dirs = [images_root / pdf_stem]
    else:
        search_dirs = [d for d in images_root.iterdir() if d.is_dir()]

    for subdir in search_dirs:
        meta = subdir / "metadata.json"
        if not meta.exists():
            continue
        engine = get_engine(str(meta))
        results = engine.search(query, top_k=top_k)
        all_results.extend(results)

    # Re-rank across all PDFs
    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results[:top_k]
