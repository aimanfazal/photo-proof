"""
tests/test_search.py
Unit and integration tests for photoproof/search.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from photoproof.indexer import index_folder, MODEL_ID
from photoproof.search import search_photos


# ---------------------------------------------------------------------------
# Unit tests (no DB, no CLIP model required)
# ---------------------------------------------------------------------------

class TestSearchUnit:
    def test_returns_empty_list_when_no_photos_indexed(self, mock_db, stub_model):
        """search_photos must return [] without error when the index is empty."""
        clip_model, preprocess, tokenizer, _, fake_embed_text = stub_model
        mock_db.search.return_value = []

        with patch("photoproof.search.model_module.embed_text", side_effect=fake_embed_text):
            results = search_photos("a sunset", clip_model, tokenizer, mock_db, limit=5)

        assert results == []

    def test_passes_limit_to_db_search(self, mock_db, stub_model):
        """search_photos must forward the limit argument to db_client.search."""
        clip_model, preprocess, tokenizer, _, fake_embed_text = stub_model

        with patch("photoproof.search.model_module.embed_text", side_effect=fake_embed_text):
            search_photos("a dog", clip_model, tokenizer, mock_db, limit=7)

        _, call_kwargs = mock_db.search.call_args
        assert call_kwargs.get("limit") == 7

    def test_result_shape(self, mock_db, stub_model):
        """Each result dict must contain id, path, filename, and similarity."""
        clip_model, preprocess, tokenizer, _, fake_embed_text = stub_model
        mock_db.search.return_value = [
            {"id": 1, "path": "/photos/img.jpg", "filename": "img.jpg", "similarity": 0.91}
        ]

        with patch("photoproof.search.model_module.embed_text", side_effect=fake_embed_text):
            results = search_photos("a cat", clip_model, tokenizer, mock_db, limit=1)

        assert len(results) == 1
        result = results[0]
        assert set(result.keys()) >= {"id", "path", "filename", "similarity"}

    def test_query_embedding_is_normalized(self, mock_db, stub_model):
        """The embedding passed to db.search must be L2-normalised (norm ≈ 1)."""
        clip_model, preprocess, tokenizer, _, fake_embed_text = stub_model
        captured: list[np.ndarray] = []

        def _spy_search(embedding, limit):
            captured.append(embedding)
            return []

        mock_db.search.side_effect = _spy_search

        with patch("photoproof.search.model_module.embed_text", side_effect=fake_embed_text):
            search_photos("flowers", clip_model, tokenizer, mock_db, limit=5)

        assert len(captured) == 1
        norm = float(np.linalg.norm(captured[0]))
        assert abs(norm - 1.0) < 1e-5, f"Embedding norm was {norm}, expected ~1.0"


# ---------------------------------------------------------------------------
# Retrieval quality test (requires --dataset-path and a real CLIP model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(True, reason="Overridden below; skip guard is in the test body")
class TestRetrievalQuality:
    pass  # placeholder — real test is the function below


def test_recall_at_5(dataset_path, tmp_path):
    """
    Measure Recall@5 on a real labeled dataset.

    Requires:
      - pytest --dataset-path /path/to/labeled/dataset
      - The dataset folder must contain sub-folders named after their label,
        e.g. dataset/sunset/img1.jpg, dataset/dogs/img2.jpg.
      - Each sub-folder label is used as the query string.

    The test asserts Recall@5 >= 0.6 across all labels.

    Skipped automatically when --dataset-path is not provided.
    """
    if dataset_path is None:
        pytest.skip("--dataset-path not provided; skipping retrieval quality test.")

    # Import here so the test module can still be collected without CLIP installed.
    from photoproof.model import load_model, embed_image, embed_text
    from photoproof.indexer import index_folder
    from photoproof.search import search_photos

    # Use a real in-memory-style mock DB backed by numpy cosine search.
    # We build a real StorageClient only if DATABASE_URL is set; otherwise
    # we use the numpy-based mock to avoid a DB requirement for quality tests.
    import os
    from tests.conftest import MockStorageClient

    db = MockStorageClient()

    # Patch search to perform real cosine similarity against the in-memory store.
    import numpy as np

    def _real_search(query_embedding: np.ndarray, limit: int = 10):
        if not db._store:
            return []
        paths = list(db._store.keys())
        embeddings = np.stack([db._store[p]["embedding"] for p in paths])
        scores = embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:limit]
        return [
            {
                "id": i,
                "path": paths[i],
                "filename": db._store[paths[i]]["filename"],
                "similarity": float(scores[i]),
            }
            for i in top_indices
        ]

    db.search.side_effect = _real_search

    model, preprocess, tokenizer = load_model()

    # Index all images in the dataset folder.
    with patch("photoproof.indexer.model_module.embed_image",
               side_effect=lambda m, p, img: embed_image(model, preprocess, img)):
        index_folder(dataset_path, model, preprocess, db)

    # Build label -> [paths] mapping from sub-folder structure.
    from pathlib import Path as _Path
    dataset_root = _Path(dataset_path)
    label_map: dict[str, list[str]] = {}
    for label_dir in sorted(dataset_root.iterdir()):
        if label_dir.is_dir():
            label_map[label_dir.name] = [str(p) for p in label_dir.rglob("*") if p.is_file()]

    if not label_map:
        pytest.skip(
            f"Dataset at {dataset_path} has no sub-folders; "
            "expected structure: dataset/<label>/<image files>."
        )

    # Evaluate Recall@5 per label.
    hits = 0
    total_queries = 0

    with patch("photoproof.search.model_module.embed_text",
               side_effect=lambda m, t, text: embed_text(model, tokenizer, text)):
        for label, relevant_paths in label_map.items():
            results = search_photos(label, model, tokenizer, db, limit=5)
            result_paths = {r["path"] for r in results}
            relevant_set = set(relevant_paths)
            if result_paths & relevant_set:
                hits += 1
            total_queries += 1

    recall_at_5 = hits / total_queries if total_queries > 0 else 0.0
    assert recall_at_5 >= 0.6, (
        f"Recall@5 = {recall_at_5:.2%} ({hits}/{total_queries}) — below the 0.60 threshold."
    )
