"""Shared pytest fixtures."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def white_bg_digit_png() -> bytes:
    """Dark digit on white background (typical canvas drawing)."""
    img = Image.new("L", (140, 140), color=255)
    pixels = img.load()
    assert pixels is not None
    # Draw a simple vertical stroke resembling "1"
    for y in range(30, 110):
        for x in range(65, 75):
            pixels[x, y] = 0
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def black_bg_digit_png() -> bytes:
    """Light digit on black background (MNIST-like)."""
    img = Image.new("L", (140, 140), color=0)
    pixels = img.load()
    assert pixels is not None
    for y in range(30, 110):
        for x in range(65, 75):
            pixels[x, y] = 255
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def empty_canvas_png() -> bytes:
    img = Image.new("L", (140, 140), color=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def jpeg_digit_bytes(white_bg_digit_png: bytes) -> bytes:
    img = Image.open(io.BytesIO(white_bg_digit_png)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
