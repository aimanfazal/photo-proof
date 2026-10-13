"""
photoproof/search.py
Convert a natural-language query to an embedding and return ranked photo results.
Also supports similar-photo search using an already-indexed image's stored embedding.
"""

from __future__ import annotations

from typing import Any

from photoproof import model as model_module
from photoproof.storage import StorageClient


def search_photos(
    query: str,
    clip_model,
    tokenizer,
    db_client: StorageClient,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search the photo index for images semantically matching query.

    Args:
        query: natural-language description, e.g. "a dog running on the beach".
        clip_model: loaded OpenCLIP model (from model.load_model).
        tokenizer: OpenCLIP tokenizer (from model.load_model).
        db_client: connected StorageClient instance.
        limit: maximum number of results to return.

    Returns:
        List of dicts ordered by descending similarity, each containing:
            - id (int)
            - path (str)
            - filename (str)
            - similarity (float): cosine similarity in [0, 1].
        Returns an empty list if no photos are indexed.
    """
    query_embedding = model_module.embed_text(clip_model, tokenizer, query)
    return db_client.search(query_embedding, limit=limit)


def search_similar(
    path: str,
    db_client: StorageClient,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find photos visually similar to an already-indexed image.

    Uses the stored embedding — no image file is opened and no CLIP inference
    is performed.

    Args:
        path: file path of an already-indexed photo.
        db_client: connected StorageClient instance.
        limit: maximum number of results to return.

    Returns:
        List of dicts ordered by descending similarity (the query photo itself
        is excluded). Returns an empty list if path is not in the index.
    """
    embedding = db_client.get_photo_embedding(path)
    if embedding is None:
        return []
    return db_client.search_excluding(embedding, exclude_path=path, limit=limit)
