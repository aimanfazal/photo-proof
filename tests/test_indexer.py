"""
tests/test_indexer.py
Unit tests for photoproof/indexer.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from photoproof import indexer as indexer_module
from photoproof.indexer import MODEL_ID, index_folder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_index(folder, mock_db, stub_model):
    """Convenience wrapper that injects the stub embed_image into indexer."""
    clip_model, preprocess, tokenizer, fake_embed_image, _ = stub_model
    with patch("photoproof.indexer.model_module.embed_image", side_effect=fake_embed_image):
        return index_folder(str(folder), clip_model, preprocess, mock_db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSkipLogic:
    def test_skips_unchanged_files_on_second_run(self, tmp_image_folder, mock_db, stub_model):
        """After a full index run, re-running the same folder should skip all files."""
        first = _run_index(tmp_image_folder, mock_db, stub_model)
        assert first["indexed"] == first["total"]
        assert first["skipped"] == 0

        second = _run_index(tmp_image_folder, mock_db, stub_model)
        assert second["skipped"] == second["total"]
        assert second["indexed"] == 0

    def test_reindexes_when_model_id_changes(self, tmp_image_folder, mock_db, stub_model):
        """Changing MODEL_ID must trigger re-embedding of all previously indexed files."""
        _run_index(tmp_image_folder, mock_db, stub_model)

        clip_model, preprocess, tokenizer, fake_embed_image, _ = stub_model
        new_model_id = "ViT-L-14/openai"
        with (
            patch("photoproof.indexer.MODEL_ID", new_model_id),
            patch("photoproof.indexer.model_module.embed_image", side_effect=fake_embed_image),
        ):
            second = index_folder(str(tmp_image_folder), clip_model, preprocess, mock_db)

        # Every file should be re-indexed because model_id no longer matches.
        assert second["indexed"] == second["total"]
        assert second["skipped"] == 0

    def test_reindexes_modified_file(self, tmp_image_folder, mock_db, stub_model):
        """A file whose mtime changes must be re-embedded on the next run."""
        _run_index(tmp_image_folder, mock_db, stub_model)

        # Pick one file and bump its mtime in the mock store to simulate an edit.
        first_path = next(iter(mock_db._store))
        mock_db._store[first_path]["modified_time"] -= 1.0  # make stored time stale

        second = _run_index(tmp_image_folder, mock_db, stub_model)
        assert second["indexed"] >= 1


class TestCorruptFileHandling:
    def test_corrupt_file_counted_as_failed(
        self, tmp_image_folder, corrupt_image_file, mock_db, stub_model
    ):
        """A corrupt file must be counted as failed without aborting the run."""
        summary = _run_index(tmp_image_folder, mock_db, stub_model)
        assert summary["failed"] == 1
        assert summary["total"] == summary["indexed"] + summary["skipped"] + summary["failed"]

    def test_corrupt_file_appears_in_errors_list(
        self, tmp_image_folder, corrupt_image_file, mock_db, stub_model
    ):
        """The errors list must contain one entry for the corrupt file."""
        summary = _run_index(tmp_image_folder, mock_db, stub_model)
        assert len(summary["errors"]) == 1
        assert str(corrupt_image_file) == summary["errors"][0]["path"]

    def test_run_completes_despite_corrupt_file(
        self, tmp_image_folder, corrupt_image_file, mock_db, stub_model
    ):
        """The run must complete and index all valid images even when one file is corrupt."""
        summary = _run_index(tmp_image_folder, mock_db, stub_model)
        # 5 valid PNGs were created by tmp_image_folder; 1 corrupt JPG added by corrupt_image_file
        assert summary["indexed"] == 5


class TestNoRawBytesInDBCalls:
    def test_upsert_photo_never_receives_raw_image_bytes(
        self, tmp_image_folder, mock_db, stub_model
    ):
        """No argument passed to upsert_photo must be a bytes object.

        This is the network-level spy: we inspect every positional keyword
        argument captured by the MagicMock spy and assert that no raw image
        binary data was transmitted.
        """
        _run_index(tmp_image_folder, mock_db, stub_model)

        for call in mock_db.upsert_photo.call_args_list:
            args, kwargs = call
            all_values = list(args) + list(kwargs.values())
            for value in all_values:
                assert not isinstance(value, bytes), (
                    f"Raw bytes found in upsert_photo call: "
                    f"argument type {type(value).__name__!r} in call {call}"
                )


class TestProgressCallback:
    def test_callback_called_for_every_file(self, tmp_image_folder, mock_db, stub_model):
        """The progress_callback must be invoked exactly once per candidate file."""
        calls: list[tuple] = []

        def _cb(done, total, filename):
            calls.append((done, total, filename))

        clip_model, preprocess, tokenizer, fake_embed_image, _ = stub_model
        with patch("photoproof.indexer.model_module.embed_image", side_effect=fake_embed_image):
            summary = index_folder(
                str(tmp_image_folder), clip_model, preprocess, mock_db,
                progress_callback=_cb,
            )

        assert len(calls) == summary["total"]

    def test_callback_final_count_equals_total(self, tmp_image_folder, mock_db, stub_model):
        """The last callback invocation must report done == total."""
        calls: list[tuple] = []

        def _cb(done, total, filename):
            calls.append((done, total, filename))

        clip_model, preprocess, tokenizer, fake_embed_image, _ = stub_model
        with patch("photoproof.indexer.model_module.embed_image", side_effect=fake_embed_image):
            index_folder(
                str(tmp_image_folder), clip_model, preprocess, mock_db,
                progress_callback=_cb,
            )

        last_done, last_total, _ = calls[-1]
        assert last_done == last_total
