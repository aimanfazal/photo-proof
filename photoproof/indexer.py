"""
photoproof/indexer.py
Orchestrates folder scanning, embedding generation, and storage.
Skips files whose path and modification time are already indexed.
"""

from __future__ import annotations

import logging

from photoproof import files, model as model_module
from photoproof.storage import StorageClient

logger = logging.getLogger(__name__)

MODEL_ID = "ViT-B-32/openai"


def index_folder(
    folder_path: str,
    clip_model,
    preprocess,
    db_client: StorageClient,
) -> dict:
    """Scan folder_path, embed new/changed images, and persist them to the database.

    Args:
        folder_path: path to the local photo folder to index.
        clip_model: loaded OpenCLIP model (from model.load_model).
        preprocess: image transform (from model.load_model).
        db_client: connected StorageClient instance.

    Returns:
        Summary dict with keys:
            - total (int): total image files found.
            - indexed (int): files newly embedded and stored.
            - skipped (int): files skipped because they are already up-to-date.
            - failed (int): files that could not be opened or embedded.
    """
    candidates = files.scan_folder(folder_path)
    already_indexed = db_client.get_indexed_files()  # {path: modified_time}

    total = len(candidates)
    indexed = 0
    skipped = 0
    failed = 0

    for entry in candidates:
        path = entry["path"]
        modified_time = entry["modified_time"]

        # Skip if the file is already indexed with the same modification time.
        if path in already_indexed and already_indexed[path] == modified_time:
            skipped += 1
            continue

        image = files.open_image(path)
        if image is None:
            logger.warning("index_folder: skipping unreadable file %s", path)
            failed += 1
            continue

        try:
            embedding = model_module.embed_image(clip_model, preprocess, image)
        except Exception as exc:  # noqa: BLE001
            logger.warning("index_folder: embedding failed for %s — %s", path, exc)
            failed += 1
            continue

        try:
            db_client.upsert_photo(
                path=path,
                filename=entry["filename"],
                file_size=entry["file_size"],
                modified_time=modified_time,
                model_id=MODEL_ID,
                embedding=embedding,
            )
            indexed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("index_folder: DB upsert failed for %s — %s", path, exc)
            failed += 1

    return {"total": total, "indexed": indexed, "skipped": skipped, "failed": failed}
