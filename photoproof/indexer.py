"""
photoproof/indexer.py
Orchestrates folder scanning, embedding generation, and storage.
Skips files whose path and modification time are already indexed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from photoproof import files, model as model_module
from photoproof.storage import StorageClient

logger = logging.getLogger(__name__)

MODEL_ID = "ViT-B-32/openai"


def index_folder(
    folder_path: str,
    clip_model,
    preprocess,
    db_client: StorageClient,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Scan folder_path, embed new/changed images, and persist them to the database.

    Args:
        folder_path: path to the local photo folder to index.
        clip_model: loaded OpenCLIP model (from model.load_model).
        preprocess: image transform (from model.load_model).
        db_client: connected StorageClient instance.
        progress_callback: optional callable invoked after each file is processed,
            with signature ``(files_done: int, total: int, current_filename: str)``.

    Returns:
        Summary dict with keys:
            - total (int): total image files found.
            - indexed (int): files newly embedded and stored.
            - skipped (int): files skipped because they are already up-to-date.
            - failed (int): files that could not be opened or embedded.
            - errors (list[dict]): per-file failures — each entry is
              ``{"path": str, "reason": str}``.
    """
    candidates = files.scan_folder(folder_path)
    already_indexed = db_client.get_indexed_files()  # {path: (modified_time, model_id)}

    total = len(candidates)
    indexed = 0
    skipped = 0
    failed = 0
    errors: list[dict] = []
    files_done = 0

    for entry in candidates:
        path = entry["path"]
        filename = entry["filename"]
        modified_time = entry["modified_time"]

        # Skip only when BOTH modification time AND model_id match the stored record.
        if path in already_indexed:
            stored_mtime, stored_model_id = already_indexed[path]
            if stored_mtime == modified_time and stored_model_id == MODEL_ID:
                skipped += 1
                files_done += 1
                if progress_callback and total > 0:
                    progress_callback(files_done, total, filename)
                continue

        image = files.open_image(path)
        if image is None:
            reason = "Could not open file (corrupt or unsupported format)"
            logger.warning("index_folder: skipping unreadable file %s", path)
            failed += 1
            errors.append({"path": path, "reason": reason})
            files_done += 1
            if progress_callback and total > 0:
                progress_callback(files_done, total, filename)
            continue

        try:
            embedding = model_module.embed_image(clip_model, preprocess, image)
        except Exception as exc:  # noqa: BLE001
            reason = f"Embedding failed: {exc}"
            logger.warning("index_folder: embedding failed for %s — %s", path, exc)
            failed += 1
            errors.append({"path": path, "reason": reason})
            files_done += 1
            if progress_callback and total > 0:
                progress_callback(files_done, total, filename)
            continue

        try:
            db_client.upsert_photo(
                path=path,
                filename=filename,
                file_size=entry["file_size"],
                modified_time=modified_time,
                model_id=MODEL_ID,
                embedding=embedding,
            )
            indexed += 1
        except Exception as exc:  # noqa: BLE001
            reason = f"Database write failed: {exc}"
            logger.warning("index_folder: DB upsert failed for %s — %s", path, exc)
            failed += 1
            errors.append({"path": path, "reason": reason})

        files_done += 1
        if progress_callback and total > 0:
            progress_callback(files_done, total, filename)

    return {
        "total": total,
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
    }
