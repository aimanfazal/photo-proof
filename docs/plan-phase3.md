# PhotoProof — Phase 3 Plan

## Top-Level Overview

Phase 3 adds three features that deliver direct user value without introducing new heavy dependencies:

1. **Similar-photo search** — select any already-indexed photo by path and find visually similar images, using its stored embedding as the query vector (no re-encoding).
2. **Filters + batch export** — filter search results by filename substring and minimum similarity threshold; export results as a downloadable CSV.
3. **Improved packaging** — `pyproject.toml`, a `Makefile` with standard targets (`install`, `run`, `test`), and a complete `README.md`.

**Scope boundary:** Hybrid keyword search, model-generated captions, and a local SQLite backend are explicitly excluded from this phase.

---

## Sub-Tasks

---

### Sub-Task 1 — Similar-Photo Search

**Intent:** Allow users to pick any already-indexed photo (by its file path) and retrieve visually similar images from the index. The stored embedding is fetched directly from the database — no image file needs to be re-opened and no CLIP inference is run.

**Expected Outcomes:**
- `storage.py` gains `get_photo_embedding(path: str) -> np.ndarray | None` — fetches the stored embedding for a given path, returns `None` if not found.
- `search.py` gains `search_similar(path: str, db_client, limit=10) -> list[dict]` — fetches the embedding and delegates to `db_client.search()`. Returns `[]` if path is not in the index. Excludes the query photo itself from results.
- `app.py` sidebar gains a "Find similar" section: a text input for a file path + a "Find Similar" button. Results rendered in the main gallery with a label "Similar to: <filename>".
- The query photo itself is not returned in results.

**Todo List:**
- [ ] Add `_GET_EMBEDDING_SQL` constant to `storage.py`: `SELECT embedding FROM photos WHERE path = %s`.
- [ ] Implement `get_photo_embedding(path)` in `storage.py` — parses the pgvector string result back to a numpy array.
- [ ] Add `_SEARCH_EXCLUDING_SQL` to `storage.py` — same as `_SEARCH_SQL` but with `AND path != %s` to exclude the query photo.
- [ ] Implement `search_similar(path, db_client, limit=10)` in `search.py`.
- [ ] Add "Find similar" sidebar section to `app.py` with a path text input and button.
- [ ] On "Find Similar" click: call `search_similar` and render results with `_show_gallery`, labelled "Similar to: <filename>".
- [ ] Show `st.warning("Photo not found in index — index the folder first.")` if `get_photo_embedding` returns `None`.

**Relevant Context:**
- [`photoproof/storage.py`](photoproof/storage.py) — `search()` method and `_SEARCH_SQL` pattern to follow.
- [`photoproof/search.py`](photoproof/search.py) — `search_photos()` as the pattern to follow.
- [`app.py`](app.py) — sidebar structure, `_show_gallery` helper.

**Status:** `[x] done`

---

### Sub-Task 2 — Filters and Batch Export (CSV)

**Intent:** Let users narrow results after a search and export them. Two filters are added: a filename substring filter (client-side, no DB query change) and a minimum similarity threshold slider. Export produces a downloadable CSV containing path, filename, and similarity for all displayed results.

**Expected Outcomes:**
- After any search (text query, image upload, or similar-photo), two filter controls appear above the gallery:
  - A text input: "Filter by filename" — hides results whose filename does not contain the substring (case-insensitive).
  - A slider: "Minimum similarity" — hides results below the threshold (default: 0.0, range 0.0–1.0, step 0.01).
- Both filters apply client-side to the current result list (no new DB query).
- A "Download results as CSV" button appears below the gallery whenever filtered results are non-empty.
- The CSV contains columns: `rank`, `filename`, `similarity`, `path`.
- Filters and export are implemented as a reusable `_apply_filters_and_export(results)` helper in `app.py` that returns the filtered list and renders the CSV button.

**Todo List:**
- [ ] Add `_apply_filters_and_export(results: list[dict]) -> list[dict]` to `app.py`.
- [ ] Inside the helper: render `st.text_input("Filter by filename")` and `st.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.01)`.
- [ ] Apply filename substring filter (case-insensitive) and similarity threshold to produce `filtered`.
- [ ] Build CSV string from `filtered` using Python's `csv` module (stdlib only — no new dependency).
- [ ] Render `st.download_button("Download results as CSV", ...)` if `filtered` is non-empty.
- [ ] Call `_apply_filters_and_export(results)` before every `_show_gallery(results, db_client)` call in `app.py`; pass filtered results to `_show_gallery`.

**Relevant Context:**
- [`app.py`](app.py) — three result display points: image upload search (line ~204), text search (line ~233), and the upcoming similar-photo search from Sub-Task 1.
- No changes to `storage.py` or `search.py` needed — purely a UI-layer concern.

**Status:** `[x] done`

---

### Sub-Task 3 — Packaging (`pyproject.toml`, `Makefile`, `README.md`)

**Intent:** Make the project installable as a Python package and provide standard developer entry points. A complete `README.md` replaces the placeholder note in `MVP.md`'s Getting Started section.

**Expected Outcomes:**
- `pyproject.toml` present with `[project]` metadata, dependencies matching `requirements.txt`, and a `[project.scripts]` entry `photoproof = "app:main"` (a thin `main()` wrapper that calls `streamlit run`).
- `Makefile` with targets:
  - `make install` — creates `.venv` and runs `pip install -e .`
  - `make run` — activates venv and runs `streamlit run app.py`
  - `make test` — runs `pytest tests/ -v`
  - `make clean` — removes `.venv`, `__pycache__`, and `.pytest_cache`
- `README.md` — complete, standalone document covering: project description, prerequisites, installation (`make install`), configuration (`.env`), running (`make run`), running tests (`make test`), project structure table, and a "How it works" section.

**Todo List:**
- [ ] Create `pyproject.toml` with `[build-system]`, `[project]` (name, version, description, requires-python, dependencies), and `[project.optional-dependencies]` for `dev = ["pytest", "pytest-mock"]`.
- [ ] Create `Makefile` with `install`, `run`, `test`, `clean` targets.
- [ ] Create `README.md` with all sections listed above.

**Relevant Context:**
- [`requirements.txt`](requirements.txt) — authoritative list of runtime dependencies to mirror in `pyproject.toml`.
- [`MVP.md`](MVP.md) §Getting Started — content to adapt for `README.md`.
- [`MVP.md`](MVP.md) §Project Structure — module table to include in `README.md`.

**Status:** `[x] done`

---

## Implementation Order

Sub-Tasks 1 and 3 are independent. Sub-Task 2 depends on Sub-Task 1 being complete (it must hook into the similar-photo search result path as well as the existing two).

```
Sub-Task 1 (similar-photo search)
Sub-Task 3 (packaging)                ← independent of ST1
        |
        v
Sub-Task 2 (filters + CSV export)     ← needs all three result paths to exist
```
