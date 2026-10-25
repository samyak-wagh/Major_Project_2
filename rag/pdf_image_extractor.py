"""
pdf_image_extractor.py — Extract all figures/diagrams from a PDF
================================================================
Uses PyMuPDF (already installed) to:
  1. Find every image on every page of the PDF
  2. Save each image as a PNG to images/<pdf_stem>/page_N_img_I.png
  3. Extract the surrounding text (±3 lines) as the image caption
  4. Build metadata.json with: page, caption, figure_label, image_path, keywords

Run standalone:
    python rag/pdf_image_extractor.py ostxtbook.pdf
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Minimum image size to bother saving (skip tiny icons/bullets)
MIN_WIDTH  = 80
MIN_HEIGHT = 80

# How many lines of surrounding text to grab as the "caption context"
CAPTION_CONTEXT_LINES = 6


def _extract_caption(page: fitz.Page, img_rect: fitz.Rect) -> str:
    """
    Extract text near the image bounding box as the figure caption.
    Searches above and below the image area on the same page.
    """
    # Search zone: expand the image rect by ~60 pts above and below
    search_rect = fitz.Rect(
        img_rect.x0,
        max(0, img_rect.y0 - 80),
        img_rect.x1,
        min(page.rect.height, img_rect.y1 + 80),
    )
    text = page.get_text("text", clip=search_rect).strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text[:500]  # cap at 500 chars


def _find_figure_label(caption: str) -> str:
    """Extract 'Figure X.Y' or 'Fig. X.Y' labels from caption text."""
    m = re.search(r"(Fig(?:ure)?\.?\s*\d+[\.\-]\d*)", caption, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _build_keywords(caption: str, page_num: int) -> List[str]:
    """
    Build searchable keywords from the caption + page number.
    Lowercased words longer than 3 chars, deduplicated.
    """
    words = re.findall(r"[a-zA-Z]{4,}", caption.lower())
    # Add page reference keyword
    words.append(f"page{page_num}")
    return list(dict.fromkeys(words))  # deduplicate while preserving order


def extract_images_from_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Extract all images from a PDF and save them to disk.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.
    output_dir : str, optional
        Directory to save images. Defaults to images/<pdf_stem>/ next to the PDF.

    Returns
    -------
    list of dict
        Each entry: {image_path, page, figure_label, caption, keywords, width, height}
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Default output dir: images/<pdf_stem>/ in the project root
    if output_dir is None:
        project_root = pdf_path.parent
        output_dir = project_root / "images" / pdf_path.stem
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    doc = fitz.open(str(pdf_path))
    records: List[Dict] = []
    total_saved = 0

    print(f"\n[PDF] Extracting images from: {pdf_path.name}")
    print(f"   Output dir : {output_dir}")
    print(f"   Pages      : {len(doc)}")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                logger.warning(f"Could not extract image xref={xref} on page {page_num}: {e}")
                continue

            width  = base_image.get("width",  0)
            height = base_image.get("height", 0)

            # Skip tiny images (decorations, bullets, etc.)
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                continue

            img_bytes = base_image["image"]
            img_ext   = base_image.get("ext", "png")

            # Save the image
            filename = f"page_{page_num:04d}_img_{img_idx:02d}.png"
            save_path = output_dir / filename

            # Always save as PNG — convert through RGB pixmap for any colorspace
            try:
                pix = fitz.Pixmap(doc, xref)
                # Convert any non-RGB colorspace (CMYK, LAB, etc.) to RGB
                if pix.colorspace and pix.colorspace.n not in (1, 3):  # not gray or RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                elif pix.alpha:  # has alpha channel
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(str(save_path))
            except Exception as e:
                # Last resort: save raw bytes with original extension
                logger.warning(f"Failed to save image as PNG (page {page_num} img {img_idx}): {e}")
                try:
                    raw_path = save_path.with_suffix(f".{img_ext}")
                    raw_path.write_bytes(img_bytes)
                    save_path = raw_path  # use raw path for metadata
                except Exception:
                    continue  # skip this image entirely

            # Get image placement rect for caption extraction
            img_rects = page.get_image_rects(xref)
            img_rect  = img_rects[0] if img_rects else fitz.Rect(0, 0, width, height)

            caption      = _extract_caption(page, img_rect)
            figure_label = _find_figure_label(caption)
            keywords     = _build_keywords(caption, page_num)

            record = {
                "image_path":   str(save_path.relative_to(pdf_path.parent)),
                "abs_path":     str(save_path),
                "page":         page_num,
                "page_human":   page_num + 1,
                "figure_label": figure_label,
                "caption":      caption,
                "keywords":     keywords,
                "width":        width,
                "height":       height,
                "pdf":          pdf_path.name,
            }
            records.append(record)
            total_saved += 1

        if (page_num + 1) % 50 == 0:
            print(f"   Processed {page_num + 1}/{len(doc)} pages, {total_saved} images so far...")

    page_count = len(doc)
    doc.close()

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Extracted {total_saved} images from {page_count} pages.")
    print(f"   Metadata saved to: {metadata_path}")
    return records


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag/pdf_image_extractor.py <path_to_pdf>")
        sys.exit(1)
    records = extract_images_from_pdf(sys.argv[1])
    print(f"\nSample record:\n{json.dumps(records[0], indent=2)}" if records else "No images found.")
