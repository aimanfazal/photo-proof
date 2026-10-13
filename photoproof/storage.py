"""
photoproof/storage.py
Manages all PostgreSQL interactions: connection, schema setup, upsert, and search.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import psycopg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Embedding dimension must match the loaded OpenCLIP model.
EMBEDDING_DIM = 512

_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS photos (
    id           BIGSERIAL PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    filename     TEXT NOT NULL,
    file_size    BIGINT,
    modified_time DOUBLE PRECISION,
    model_id     TEXT NOT NULL,
    description  TEXT,
    embedding    VECTOR({EMBEDDING_DIM})
);
"""

_UPSERT_SQL = """
INSERT INTO photos (path, filename, file_size, modified_time, model_id, embedding)
VALUES (%s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (path) DO UPDATE SET
    filename      = EXCLUDED.filename,
    file_size     = EXCLUDED.file_size,
    modified_time = EXCLUDED.modified_time,
    model_id      = EXCLUDED.model_id,
    embedding     = EXCLUDED.embedding;
"""

_GET_INDEXED_SQL = "SELECT path, modified_time, model_id FROM photos;"

_SEARCH_SQL = """
SELECT
    id,
    path,
    filename,
    1 - (embedding <=> %s::vector) AS similarity
FROM photos
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


class StorageClient:
    """Thin wrapper around a psycopg connection for PhotoProof's DB operations."""

    def __init__(self, connection: psycopg.Connection) -> None:
        self._conn = connection

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def connect(cls) -> "StorageClient":
        """Open a connection using DATABASE_URL and SSL_MODE from the environment."""
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Copy .env.example to .env and fill in your connection string."
            )
        ssl_mode = os.environ.get("SSL_MODE", "require")
        conn = psycopg.connect(database_url, sslmode=ssl_mode, autocommit=False)
        return cls(conn)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def setup_schema(self) -> None:
        """Create the pgvector extension and photos table if they do not exist."""
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_photo(
        self,
        path: str,
        filename: str,
        file_size: int,
        modified_time: float,
        model_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Insert or update a photo record including its embedding."""
        # pgvector expects a list representation: '[0.1, 0.2, ...]'
        embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
        with self._conn.cursor() as cur:
            cur.execute(
                _UPSERT_SQL,
                (path, filename, file_size, modified_time, model_id, embedding_str),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_indexed_files(self) -> dict[str, tuple[float, str]]:
        """Return {path: (modified_time, model_id)} for every already-indexed photo."""
        with self._conn.cursor() as cur:
            cur.execute(_GET_INDEXED_SQL)
            rows = cur.fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def search(
        self, query_embedding: np.ndarray, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return top-`limit` photos ranked by cosine similarity to query_embedding."""
        embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
        with self._conn.cursor() as cur:
            cur.execute(_SEARCH_SQL, (embedding_str, embedding_str, limit))
            rows = cur.fetchall()
        return [
            {"id": r[0], "path": r[1], "filename": r[2], "similarity": float(r[3])}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def delete_photo(self, path: str) -> None:
        """Remove a photo record from the database by its file path."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM photos WHERE path = %s;", (path,))
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
