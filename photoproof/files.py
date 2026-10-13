"""
photoproof/files.py
Filesystem utilities: discover supported images in a folder and open them safely.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
)


def scan_folder(folder_path: str) -> list[dict]:
    """Recursively find all supported image files under folder_path.

    Args:
        folder_path: absolute or relative path to the root folder to scan.

    Returns:
        List of dicts, each containing:
            - path (str): absolute file path.
            - filename (str): file name (e.g. "IMG_1048.JPG").
            - file_size (int): file size in bytes.
            - modified_time (float): last-modified Unix timestamp.
    """
    root = Path(folder_path)
    if not root.is_dir():
        logger.warning("scan_folder: %s is not a directory — skipping.", folder_path)
        return []

    results: list[dict] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            stat = file_path.stat()
            results.append(
                {
                    "path": str(file_path),
                    "filename": file_path.name,
                    "file_size": stat.st_size,
                    "modified_time": stat.st_mtime,
                }
            )
        except OSError as exc:
            logger.warning("scan_folder: could not stat %s — %s", file_path, exc)

    return results


def file_exists(path: str) -> bool:
    """Return True if the file at path currently exists on disk."""
    return Path(path).is_file()


def open_image(path: str) -> Image.Image | None:
    """Open an image file and return it as an RGB PIL Image.

    Args:
        path: path to the image file.

    Returns:
        PIL.Image in RGB mode, or None if the file cannot be read.
    """
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("open_image: could not open %s — %s", path, exc)
        return None
