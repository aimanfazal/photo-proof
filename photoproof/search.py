"""
photoproof/search.py
Convert a natural-language query to an embedding and return ranked photo results.
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
