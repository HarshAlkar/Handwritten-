"""Unit tests for the image preprocessing pipeline."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from ML.preprocessing.pipeline import PreprocessError, preprocess_image_bytes, preprocess_pil_image


def test_white_background_produces_valid_tensor(white_bg_digit_png: bytes):
    out = preprocess_image_bytes(white_bg_digit_png, content_type="image/png", filename="d.png")
    assert out.shape == (1, 28, 28, 1)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0
    assert float(out.mean()) > 0.01  # some ink present


def test_black_background_produces_valid_tensor(black_bg_digit_png: bytes):
    out = preprocess_image_bytes(black_bg_digit_png, content_type="image/png", filename="d.png")
    assert out.shape == (1, 28, 28, 1)
    assert float(out.max()) > 0.5


def test_jpeg_supported(jpeg_digit_bytes: bytes):
    out = preprocess_image_bytes(jpeg_digit_bytes, content_type="image/jpeg", filename="d.jpg")
    assert out.shape == (1, 28, 28, 1)


def test_empty_canvas_rejected(empty_canvas_png: bytes):
    with pytest.raises(PreprocessError, match="Empty"):
        preprocess_image_bytes(empty_canvas_png, content_type="image/png", filename="empty.png")


def test_unsupported_extension_rejected(white_bg_digit_png: bytes):
    with pytest.raises(PreprocessError, match="Unsupported"):
        preprocess_image_bytes(white_bg_digit_png, filename="digit.gif")


def test_invalid_bytes_rejected():
    with pytest.raises(PreprocessError):
        preprocess_image_bytes(b"not-an-image", content_type="image/png", filename="x.png")


def test_pil_rgb_image():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    # draw black blob
    for x in range(40, 60):
        for y in range(20, 80):
            img.putpixel((x, y), (0, 0, 0))
    out = preprocess_pil_image(img)
    assert out.shape == (1, 28, 28, 1)
