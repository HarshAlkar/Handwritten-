"""Image preprocessing for MNIST-style digit inference."""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image

SUPPORTED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/x-png",
}

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# MNIST digits are white strokes on a dark background after normalization.
TARGET_SIZE = 28
EMPTY_THRESHOLD = 0.01  # fraction of "ink" pixels required


class PreprocessError(ValueError):
    """Raised when an image cannot be preprocessed for prediction."""


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] float32."""
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _to_grayscale(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, image.convert("RGBA"))
        return composite.convert("L")
    return image.convert("L")


def _ensure_white_on_black(arr: np.ndarray) -> np.ndarray:
    """
    Ensure digit strokes are bright on a dark background (MNIST convention).

    Uses mean intensity: light-background images (dark ink) are inverted;
    dark-background images (light ink) are left as-is.
    """
    mean_val = float(arr.mean())
    if mean_val > 127:
        return 255.0 - arr
    return arr


def _bbox_of_ink(arr: np.ndarray, ink_threshold: float = 20.0) -> Tuple[int, int, int, int] | None:
    """Return (row_min, row_max, col_min, col_max) inclusive bounds of ink, or None."""
    mask = arr > ink_threshold
    if not np.any(mask):
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return int(row_min), int(row_max), int(col_min), int(col_max)


def _center_and_resize(arr: np.ndarray) -> np.ndarray:
    """Crop ink, pad to square preserving aspect ratio, resize to 28x28."""
    bbox = _bbox_of_ink(arr)
    if bbox is None:
        raise PreprocessError("Empty image: no digit detected.")

    r0, r1, c0, c1 = bbox
    # Small padding around the digit
    pad = 2
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(arr.shape[0] - 1, r1 + pad)
    c1 = min(arr.shape[1] - 1, c1 + pad)
    cropped = arr[r0 : r1 + 1, c0 : c1 + 1]

    h, w = cropped.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off : y_off + h, x_off : x_off + w] = cropped

    # Slight extra border like classic MNIST (~20% margin via resize from padded square)
    # Add 4px equivalent margin by embedding into a larger canvas before resize.
    margin = max(1, side // 5)
    padded_side = side + 2 * margin
    padded = np.zeros((padded_side, padded_side), dtype=np.float32)
    padded[margin : margin + side, margin : margin + side] = square

    img = Image.fromarray(padded.astype(np.uint8), mode="L")
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _ink_fraction(arr_01: np.ndarray) -> float:
    return float(np.mean(arr_01 > 0.1))


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image into model input shape (1, 28, 28, 1).

    Returns float32 array normalized to [0, 1] with white digits on black.
    """
    gray = _to_grayscale(image)
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        raise PreprocessError("Invalid image: empty pixel data.")

    arr = _ensure_white_on_black(arr)
    resized = _center_and_resize(arr)
    normalized = normalize_array(resized)

    if _ink_fraction(normalized) < EMPTY_THRESHOLD:
        raise PreprocessError("Empty canvas: please draw a digit before predicting.")

    return normalized.reshape(1, TARGET_SIZE, TARGET_SIZE, 1)


def preprocess_image_bytes(
    data: bytes,
    content_type: str | None = None,
    filename: str | None = None,
) -> np.ndarray:
    """Validate and preprocess raw image bytes."""
    if not data:
        raise PreprocessError("No image data provided.")

    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct and ct not in SUPPORTED_CONTENT_TYPES and ct != "application/octet-stream":
            raise PreprocessError("Unsupported file type. Upload a PNG or JPEG image.")

    if filename:
        lower = filename.lower()
        if "." in lower:
            ext = "." + lower.rsplit(".", 1)[-1]
            if ext not in SUPPORTED_EXTENSIONS:
                raise PreprocessError("Unsupported file type. Upload a PNG or JPEG image.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 — convert to API-friendly error
        raise PreprocessError("Invalid or corrupted image file.") from exc

    fmt = (image.format or "").upper()
    if fmt not in {"PNG", "JPEG", "JPG"} and content_type not in SUPPORTED_CONTENT_TYPES:
        # Pillow may leave format unset for some streams; allow if content-type ok
        if content_type is None and filename is None:
            raise PreprocessError("Unsupported file type. Upload a PNG or JPEG image.")

    return preprocess_pil_image(image)
