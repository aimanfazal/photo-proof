# PhotoProof — Phase 2 Reliability Plan

## Top-Level Overview

Harden the PhotoProof Phase 1 MVP across five reliability dimensions:

1. **Robust incremental indexing** — extend the skip-unchanged logic to also check `model_id`, so re-indexing is triggered whenever the embedding model changes.
2. **Missing-path handling in search results** — search results can point to files that have since been moved or deleted; the gallery must detect and communicate this gracefully rather than failing silently.
3. **Real indexing progress and per-file error reporting** — replace the single spinner with a `st.progress` bar and accumulate per-file error details shown to the user after indexing completes.
4. **Retrieval quality tests** — `tests/test_indexer.py` and `tests/test_search.py` using pytest, exercising the skip logic, corrupt-file handling, and retrieval quality against a configurable real dataset path.
5. **"No raw photo bytes uploaded" test** — a psycopg-level mock/spy that inspects every SQL parameter passed to the DB and asserts no binary image data is present.

**No new features** from Phase 3 are introduced. All changes are backward-compatible with the existing schema and module interfaces.

---

## Sub-Tasks

---

### Sub-Task 1 — Robust Incremental Indexing (model_id-aware skip)

**Intent:** The Phase 1 skip logic compares `path` + `modified_time` only. If the user switches to a different OpenCLIP model, existing embeddings are stale but `modified_time` has not changed, so they would not be re-generated. The skip check must also compare `model_id` against the stored value.

**Expected Outcomes:**
- `storage.get_indexed_files()` returns `{path: (modified_time, model_id)}` tuples instead of just `modified_time` floats.
- `indexer.index_folder()` skips a file only when both `modified_time` AND `model_id` match the stored record.
- Existing behaviour for unchanged files with the same model is preserved.
- A file that was indexed with an old model is re-embedded on the next run with a new model.

**Todo List:**
- [ ] Update `_GET_INDEXED_SQL` in `storage.py` to also select `model_id`.
- [ ] Update `get_indexed_files()` return type to `dict[str, tuple[float, str]]` — `{path: (modified_time, model_id)}`.
- [ ] Update the skip condition in `indexer.index_folder()` to check both `modified_time` and `model_id`.
- [ ] Update the docstrings for both changed functions.

**Relevant Context:**
- [`photoproof/storage.py`](photoproof/storage.py) — `_GET_INDEXED_SQL`, `get_indexed_files()`.
- [`photoproof/indexer.py`](photoproof/indexer.py) — skip condition on line 53.
- `MODEL_ID = "ViT-B-32/openai"` is the constant in `indexer.py`.

**Status:** `[x] done`

---

### Sub-Task 2 — Missing-Path Handling in Search Results

**Intent:** When a search returns results pointing to files that no longer exist on disk (moved, deleted, or on a network drive that is currently offline), the gallery currently silently shows a broken thumbnail warning. Phase 2 should detect missing paths explicitly and communicate clearly to the user, distinguishing "file moved/deleted" from "unreadable corrupt file".

**Expected Outcomes:**
- `files.py` gains a `file_exists(path: str) -> bool` helper.
- `storage.py` gains a `delete_photo(path: str) -> None` method to remove a stale record from the DB.
- In `app.py`, the `_show_gallery` helper checks `file_exists` for each result; if the file is missing it shows a distinct "⚠ File not found — removed from index" message and calls `db_client.delete_photo(path)` to clean up the stale record.
- The missing-path check happens at gallery render time, not at search query time (keeps search fast).

**Todo List:**
- [ ] Add `file_exists(path: str) -> bool` to `files.py`.
- [ ] Add `delete_photo(path: str) -> None` to `storage.py` with `DELETE FROM photos WHERE path = %s`.
- [ ] In `app.py` `_show_gallery`: check `file_exists(result["path"])` before rendering; if False, show a warning caption and call `db_client.delete_photo(result["path"])`.
- [ ] Update `_show_gallery` signature to accept `db_client` as a parameter.

**Relevant Context:**
- [`photoproof/files.py`](photoproof/files.py) — add alongside `open_image`.
- [`photoproof/storage.py`](photoproof/storage.py) — add alongside `upsert_photo`.
- [`app.py`](app.py) — `_show_gallery` helper, lines 68–87.

**Status:** `[x] done`

---

### Sub-Task 3 — Indexing Progress Bar and Per-File Error Reporting

**Intent:** The current UI shows only a spinner during indexing and a one-line summary at the end. For large folders, users have no feedback during the run and no detail about which files failed. Phase 2 adds a `st.progress` bar updated per file and an expandable error log shown after indexing completes.

**Expected Outcomes:**
- `indexer.index_folder()` accepts an optional `progress_callback: Callable[[int, int, str], None] | None` parameter — called after each file is processed with `(files_done, total, current_filename)`.
- `indexer.index_folder()` returns an augmented summary: adds `errors: list[dict]` where each entry is `{path, reason}`.
- In `app.py`, the Index Folder handler:
  - Creates a `st.progress(0)` and a `st.empty()` status text before calling `index_folder`.
  - Passes a callback that updates the progress bar and status text.
  - After completion, if `summary["errors"]`, shows an `st.expander("Failed files")` listing each `{path, reason}`.

**Todo List:**
- [ ] Add `progress_callback` optional parameter to `index_folder()` in `indexer.py`.
- [ ] Call `progress_callback(files_done, total, current_filename)` after processing each file (whether indexed, skipped, or failed).
- [ ] Add `errors: list[dict]` to the return dict — each entry is `{"path": ..., "reason": ...}`.
- [ ] In `app.py`, replace the `st.spinner` block for indexing with `st.progress` + `st.empty` status text.
- [ ] After indexing, render `st.expander("Failed files — click to expand")` if `summary["errors"]` is non-empty.
- [ ] Ensure the progress bar reaches 1.0 and the status text clears (or shows "Done") when indexing finishes.

**Relevant Context:**
- [`photoproof/indexer.py`](photoproof/indexer.py) — `index_folder()` loop.
- [`app.py`](app.py) — Index Folder handler, lines 140–157.
- Streamlit `st.progress` accepts a float 0.0–1.0; update it as `files_done / total` after each file.

**Status:** `[x] done`

---

### Sub-Task 4 — Test Suite: Indexer and Search Logic

**Intent:** Add `pytest`-based unit and integration tests for the two most critical modules — `indexer.py` (skip logic, corrupt-file handling) and `search.py` (text-to-embedding flow, empty-index guard). Tests use synthetic fixture images (generated solid-colour PNGs) so they work without a real dataset. A separate parametrised test accepts a real labeled dataset path via a `pytest` CLI option (`--dataset-path`) and measures Recall@5.

**Expected Outcomes:**
- `tests/__init__.py` created (empty, marks the package).
- `tests/conftest.py` created with:
  - A `tmp_image_folder` fixture that generates a small set of solid-colour PNG fixtures.
  - A `mock_db` fixture that provides an in-memory `StorageClient`-compatible double (using `unittest.mock`).
  - A `--dataset-path` pytest CLI option.
- `tests/test_indexer.py`:
  - `test_skips_unchanged_file` — indexes a folder, calls `index_folder` again, asserts `skipped == total`.
  - `test_reindexes_on_model_change` — patches `MODEL_ID` to a different string, calls `index_folder` again, asserts `indexed == total`.
  - `test_handles_corrupt_file` — places an invalid file in the folder, asserts `failed == 1` and run completes.
  - `test_no_raw_bytes_in_db_calls` — spies on `StorageClient.upsert_photo` and asserts no argument is a `bytes` object.
- `tests/test_search.py`:
  - `test_returns_empty_on_no_index` — with empty mock DB, `search_photos` returns `[]`.
  - `test_recall_at_5` — skipped unless `--dataset-path` is provided; indexes the dataset, runs labelled queries, asserts Recall@5 ≥ 0.6.

**Todo List:**
- [ ] Create `tests/__init__.py` (empty).
- [ ] Create `tests/conftest.py` with `tmp_image_folder`, `mock_db` fixtures and `--dataset-path` CLI option.
- [ ] Create `tests/test_indexer.py` with the four tests listed above.
- [ ] Create `tests/test_search.py` with the two tests listed above.
- [ ] Add `pytest` and `pytest-mock` to `requirements.txt`.

**Relevant Context:**
- [`photoproof/indexer.py`](photoproof/indexer.py) — `index_folder`, `MODEL_ID`.
- [`photoproof/search.py`](photoproof/search.py) — `search_photos`.
- [`photoproof/storage.py`](photoproof/storage.py) — `StorageClient` interface to mock.
- The "no raw bytes" test is the psycopg-level spy requested: inspect every positional argument passed to `upsert_photo` and assert `isinstance(arg, bytes)` is False for all.

**Status:** `[x] done`

---

## Implementation Order

Sub-Tasks 1 and 2 are independent code changes and can be done in either order.
Sub-Task 3 depends on Sub-Task 1 (because it documents the updated return shape).
Sub-Task 4 depends on Sub-Tasks 1–3 being complete (tests cover the final behaviour).

```
Sub-Task 1 (model_id-aware skip)
Sub-Task 2 (missing-path handling)    ← independent of ST1
        |
        v
Sub-Task 3 (progress bar + error log) ← needs ST1 return shape
        |
        v
Sub-Task 4 (test suite)               ← tests final behaviour of ST1-3
```
