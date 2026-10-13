"""
tests/conftest.py
Shared pytest fixtures and CLI options for PhotoProof tests.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from photoproof.indexer import MODEL_ID


# ---------------------------------------------------------------------------
# CLI option: --dataset-path for real labeled retrieval tests
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dataset-path",
        action="store",
        default=None,
        metavar="PATH",
        help="Path to a labeled image dataset folder for retrieval quality tests.",
    )


@pytest.fixture
def dataset_path(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--dataset-path")


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------

def _make_solid_png(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
    """Generate a minimal valid PNG of a solid colour in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tmp_image_folder(tmp_path: Path) -> Path:
    """Create a temporary folder with a small set of solid-colour PNG fixtures."""
    colours = {
        "red.png": (255, 0, 0),
        "green.png": (0, 255, 0),
        "blue.png": (0, 0, 255),
        "white.png": (255, 255, 255),
        "black.png": (0, 0, 0),
    }
    for filename, colour in colours.items():
        (tmp_path / filename).write_bytes(_make_solid_png(colour))
    return tmp_path


@pytest.fixture
def corrupt_image_file(tmp_image_folder: Path) -> Path:
    """Add one corrupt (non-image) file to the tmp_image_folder and return its path."""
    corrupt = tmp_image_folder / "corrupt.jpg"
    corrupt.write_bytes(b"this is not image data")
    return corrupt


# ---------------------------------------------------------------------------
# Mock StorageClient
# ---------------------------------------------------------------------------

class MockStorageClient:
    """In-memory StorageClient double for unit tests."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        # Expose call-tracking spies
        self.upsert_photo = MagicMock(side_effect=self._upsert_photo)
        self.get_indexed_files = MagicMock(side_effect=self._get_indexed_files)
        self.search = MagicMock(return_value=[])
        self.delete_photo = MagicMock(side_effect=self._delete_photo)

    def _upsert_photo(
        self,
        path: str,
        filename: str,
        file_size: int,
        modified_time: float,
        model_id: str,
        embedding: np.ndarray,
    ) -> None:
        self._store[path] = {
            "filename": filename,
            "file_size": file_size,
            "modified_time": modified_time,
            "model_id": model_id,
            "embedding": embedding,
        }

    def _get_indexed_files(self) -> dict[str, tuple[float, str]]:
        return {
            path: (record["modified_time"], record["model_id"])
            for path, record in self._store.items()
        }

    def _delete_photo(self, path: str) -> None:
        self._store.pop(path, None)


@pytest.fixture
def mock_db() -> MockStorageClient:
    """Return a fresh in-memory MockStorageClient."""
    return MockStorageClient()


# ---------------------------------------------------------------------------
# Stub model (returns deterministic random embeddings without loading CLIP)
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_model():
    """Return (clip_model, preprocess, tokenizer) stubs that produce random 512-d embeddings."""
    rng = np.random.default_rng(42)

    clip_model = MagicMock()
    preprocess = MagicMock(return_value=MagicMock())
    tokenizer = MagicMock(return_value=MagicMock())

    def _fake_embed_image(model, prep, image):
        vec = rng.random(512).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def _fake_embed_text(model, tok, text):
        vec = rng.random(512).astype(np.float32)
        return vec / np.linalg.norm(vec)

    return clip_model, preprocess, tokenizer, _fake_embed_image, _fake_embed_text
