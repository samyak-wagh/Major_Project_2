"""
Image Handler
=============
Extracts text from uploaded images (handwritten notes, textbook photos,
whiteboard diagrams) using EasyOCR — fully offline after first model download.

Supports: PNG, JPG, JPEG, BMP, TIFF, WEBP
"""

import io
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-load EasyOCR reader (downloads ~100MB on first use, then cached)
_reader = None


def _get_reader():
    """Return a cached EasyOCR reader for English."""
    global _reader
    if _reader is None:
        logger.info("Loading EasyOCR model (first-time download ~100MB)...")
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR model loaded.")
    return _reader


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from an image using EasyOCR.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes (PNG, JPG, etc.)

    Returns
    -------
    str
        All text found in the image, joined into a single string.
        Returns empty string if no text is detected.
    """
    try:
        # Convert bytes → PIL Image → NumPy array (EasyOCR input format)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_image)

        reader = _get_reader()
        # detail=0 returns plain text strings (no bounding boxes needed)
        results = reader.readtext(img_array, detail=0, paragraph=True)

        extracted = " ".join(results).strip()
        logger.info(f"OCR extracted {len(extracted)} characters.")
        return extracted

    except Exception as e:
        logger.error(f"OCR error: {e}")
        raise RuntimeError(f"Could not extract text from image: {e}")
