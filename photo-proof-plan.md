# PhotoProof — Phase 1 MVP Implementation Plan

## Top-Level Overview

Build the complete PhotoProof Phase 1 MVP from a blank repository.

The goal is a locally runnable Streamlit application that:
1. Indexes a local photo folder (typed path) and optionally processes a single uploaded image.
2. Generates image embeddings via OpenCLIP (locally, on CPU/GPU).
3. Stores embeddings and photo metadata in a Tiger Data PostgreSQL + `pgvector` database.
4. Accepts natural-language search queries, converts them to text embeddings, and returns ranked photo results.
5. Displays matching photos in a Streamlit gallery with filenames and original paths.

**Model default:** `ViT-B-32` with `openai` pretrained weights — smallest OpenCLIP model, 512-dimensional embeddings, runs comfortably on a laptop CPU. Embedding dimension is `512`.

**PostgreSQL driver:** `psycopg[binary]` (psycopg v3) — modern async-friendly driver; binary package avoids needing libpq installed locally.

**SSL mode:** Configurable via `SSL_MODE` environment variable (default: `require`). Tiger Data requires TLS; local dev instances can override with `SSL_MODE=disable` in `.env`.

**Single uploaded image:** Treated as a search query only — not upserted into the database. The file uploader has no meaningful local path to persist, and keeping indexing and search as separate operations keeps the data model clean.

**Tests:** Deferred to Phase 2. No test files in this plan.

**Scope boundary:** Only Phase 1 roadmap items. Phase 2/3 features (hybrid search, model-generated descriptions, local SQLite backend, etc.) are explicitly excluded.

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffold

**Intent:** Create the project skeleton so all subsequent sub-tasks have a home. Establishes `.gitignore`, `.env.example`, `requirements.txt`, the `photoproof/` package directory with an empty `__init__.py`, and the `.vscode/mcp.json` Tiger Data MCP config.

**Expected Outcomes:**
- Repository contains the full directory structure described in `MVP.md`.
- `requirements.txt` pins all runtime dependencies.
- `.env.example` shows the required `DATABASE_URL` variable.
- `.gitignore` excludes `.env`, `.venv/`, `__pycache__/`, `*.py[cod]`, `.streamlit/secrets.toml`.
- `.vscode/mcp.json` contains the Tiger Data MCP stdio configuration.
- `photoproof/__init__.py` is empty (marks the package).

**Todo List:**
- [ ] Create `.gitignore` with the entries listed in `MVP.md`.
- [ ] Create `.env.example` with `DATABASE_URL=your_postgresql_connection_string`.
- [ ] Create `requirements.txt` with: `streamlit`, `open-clip-torch`, `torch` (CPU build reference note), `Pillow`, `numpy`, `psycopg[binary]`, `python-dotenv`.
- [ ] Create `photoproof/__init__.py` (empty).
- [ ] Create `.vscode/mcp.json` with the Tiger Data MCP stdio config from `MVP.md`.

**Relevant Context:**
- `MVP.md` §Getting Started (prerequisites, `.gitignore` entries, `.env` format).
- `MVP.md` §Technology Stack (full dependency list).
- `MVP.md` §Tiger Data Integration / Development-time MCP integration (`.vscode/mcp.json` format).

**Status:** `[ ] pending`

---

### Sub-Task 2 — Database Storage Module (`storage.py`)

**Intent:** Implement `photoproof/storage.py`, which manages all PostgreSQL interactions: connection setup, schema creation/migration, upserting photo records, and querying stored embeddings.

**Expected Outcomes:**
- `storage.py` exports a `StorageClient` class (or equivalent functions) with:
  - `connect()` — reads `DATABASE_URL` from environment and opens a `psycopg` connection with `sslmode=require`.
  - `setup_schema()` — creates the `pgvector` extension and the `photos` table if they do not exist.
  - `upsert_photo(path, filename, file_size, modified_time, model_id, embedding)` — inserts or updates a photo record.
  - `get_indexed_files()` — returns a dict of `{path: modified_time}` for all indexed photos (used to skip unchanged files).
  - `search(query_embedding, limit=10)` — returns the top-`limit` rows ordered by cosine distance, including `id`, `path`, `filename`, and `similarity`.
- `DATABASE_URL` and `SSL_MODE` are loaded via `python-dotenv` so a `.env` file is picked up automatically.
- No raw credentials appear in log output.

**Todo List:**
- [ ] Read `DATABASE_URL` and `SSL_MODE` (default `require`) from environment using `python-dotenv`.
- [ ] Implement `connect()` using `psycopg.connect` with `sslmode` set from the `SSL_MODE` env var.
- [ ] Implement `setup_schema()` running the `CREATE EXTENSION` and `CREATE TABLE` SQL from `MVP.md`.
- [ ] Implement `upsert_photo()` using `INSERT ... ON CONFLICT (path) DO UPDATE`.
- [ ] Implement `get_indexed_files()` selecting `path` and `modified_time`.
- [ ] Implement `search()` using the cosine-distance query from `MVP.md`, passing the query embedding as a `pgvector` vector parameter.
- [ ] Use `VECTOR(512)` as the embedding column dimension (matching `ViT-B-32`).

**Relevant Context:**
- `MVP.md` §Tiger Data Integration — schema SQL and search query SQL.
- `MVP.md` §Privacy and Security — no logging of credentials or sensitive paths.
- `.env.example` should include both `DATABASE_URL=your_postgresql_connection_string` and `SSL_MODE=require`.
- Embedding dimension: `512` (ViT-B-32 / openai weights).

**Status:** `[ ] pending`

---

### Sub-Task 3 — Model Module (`model.py`)

**Intent:** Implement `photoproof/model.py`, which loads the OpenCLIP model once and exposes functions to generate image embeddings from PIL Images and text embeddings from strings.

**Expected Outcomes:**
- `model.py` exports:
  - `load_model()` — loads `ViT-B-32` with `openai` pretrained weights, returns `(model, preprocess, tokenizer)`. Model is moved to CPU (or CUDA if available). Model is set to `eval()`.
  - `embed_image(model, preprocess, image: PIL.Image) -> np.ndarray` — returns a normalized 512-d float32 numpy vector.
  - `embed_text(model, tokenizer, text: str) -> np.ndarray` — returns a normalized 512-d float32 numpy vector.
- Embeddings are L2-normalized so cosine similarity equals dot product.
- Model is loaded lazily / cached using `@st.cache_resource` when called from the Streamlit app (the module itself does not import Streamlit; caching is the app's responsibility).

**Todo List:**
- [ ] Import `open_clip`, `torch`, `numpy`, `PIL.Image`.
- [ ] Implement `load_model()` using `open_clip.create_model_and_transforms` and `open_clip.get_tokenizer`.
- [ ] Implement `embed_image()`: preprocess image, run `model.encode_image`, normalize with `numpy`.
- [ ] Implement `embed_text()`: tokenize text, run `model.encode_text`, normalize with `numpy`.
- [ ] Ensure both embed functions run under `torch.no_grad()` and return CPU numpy arrays.

**Relevant Context:**
- `MVP.md` §How It Works — OpenCLIP image and text encoder roles.
- Model: `ViT-B-32`, pretrained: `openai`, embedding dim: `512`.

**Status:** `[ ] pending`

---

### Sub-Task 4 — File Utilities Module (`files.py`)

**Intent:** Implement `photoproof/files.py`, which handles filesystem concerns: discovering supported image files in a folder and safely opening original images.

**Expected Outcomes:**
- `files.py` exports:
  - `SUPPORTED_EXTENSIONS` — a frozenset of lowercase extensions: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`.
  - `scan_folder(folder_path: str) -> list[dict]` — returns a list of dicts with `path`, `filename`, `file_size`, `modified_time` for each supported image found (recursive scan).
  - `open_image(path: str) -> PIL.Image` — opens an image and converts it to `RGB`. Returns `None` and logs a warning if the file cannot be read (corrupt or missing).

**Todo List:**
- [ ] Implement `scan_folder()` using `pathlib.Path.rglob` filtered by `SUPPORTED_EXTENSIONS`.
- [ ] Populate each dict with `str(path)`, `path.name`, `path.stat().st_size`, `path.stat().st_mtime`.
- [ ] Implement `open_image()` with a try/except that catches `PIL.UnidentifiedImageError` and `OSError`.

**Relevant Context:**
- `MVP.md` §Module responsibilities — `files.py` validates image paths and opens original images.
- `MVP.md` §Phase 2 Reliability — corrupt file handling noted as future work; `open_image` should still be defensive.

**Status:** `[ ] pending`

---

### Sub-Task 5 — Indexer Module (`indexer.py`)

**Intent:** Implement `photoproof/indexer.py`, which orchestrates folder scanning, embedding generation, and storage — skipping files whose path and modification time are already indexed.

**Expected Outcomes:**
- `indexer.py` exports:
  - `index_folder(folder_path, model, preprocess, db_client) -> dict` — scans the folder, skips unchanged files, generates embeddings for new/changed files, upserts them in the database, and returns a summary dict `{total, indexed, skipped, failed}`.
- Unchanged files (same path + `modified_time`) are skipped without re-embedding.
- Failed files (unreadable images) are counted but do not abort the whole run.

**Todo List:**
- [ ] Call `files.scan_folder()` to get all candidate images.
- [ ] Call `db_client.get_indexed_files()` to get already-indexed paths with their `modified_time`.
- [ ] For each candidate, skip if `path` is in indexed files and `modified_time` matches.
- [ ] For new/changed files: call `files.open_image()`, then `model.embed_image()`, then `db_client.upsert_photo()`.
- [ ] Count and return `{total, indexed, skipped, failed}`.
- [ ] Use `MODEL_ID = "ViT-B-32/openai"` as the `model_id` value stored in the database.

**Relevant Context:**
- `MVP.md` §Search Pipeline step 2–3 — embeddings generated once and reused.
- `MVP.md` §Phase 1 Roadmap — "Scan and index local images" and "Generate and persist image embeddings".

**Status:** `[ ] pending`

---

### Sub-Task 6 — Search Module (`search.py`)

**Intent:** Implement `photoproof/search.py`, which converts a user's text query to an embedding and delegates similarity search to the storage client.

**Expected Outcomes:**
- `search.py` exports:
  - `search_photos(query: str, model, tokenizer, db_client, limit=10) -> list[dict]` — returns a ranked list of dicts with `id`, `path`, `filename`, `similarity`.

**Todo List:**
- [ ] Call `model.embed_text(model_obj, tokenizer, query)` to get the query embedding.
- [ ] Call `db_client.search(query_embedding, limit=limit)` and return the result.
- [ ] Return an empty list (not an error) if the database has no indexed photos.

**Relevant Context:**
- `MVP.md` §Search Pipeline steps 4–7.
- `MVP.md` §Tiger Data Integration — the cosine-distance SQL query.

**Status:** `[ ] pending`

---

### Sub-Task 7 — Streamlit Application (`app.py`)

**Intent:** Implement the root-level `app.py`, which wires all modules together into the Streamlit user interface.

**Expected Outcomes:**
- Running `streamlit run app.py` opens a browser UI with:
  - **Sidebar:** typed folder-path input for indexing + an "Index Folder" button + a single-image file-uploader fallback.
  - **Main area:** a text input for natural-language search + a "Search" button.
  - **Gallery:** matching photo thumbnails displayed in a grid (using `st.columns`), each labelled with filename and similarity score; clicking/hovering shows the full file path.
  - **Status messages:** indexing summary (`total / indexed / skipped / failed`) and error messages shown via `st.info` / `st.error`.
- `load_model()` and `StorageClient` are cached with `@st.cache_resource`.
- `setup_schema()` is called once on startup.
- Single uploaded image is embedded and searched against the existing index (no upsert — it is a search query, not an indexing operation; the file has no persistent local path to store).

**Todo List:**
- [ ] Import all `photoproof` modules.
- [ ] Cache `load_model()` result with `@st.cache_resource`.
- [ ] Cache `StorageClient` (connected + schema set up) with `@st.cache_resource`.
- [ ] Build sidebar: folder-path `st.text_input`, "Index Folder" `st.button`, `st.file_uploader` for single image.
- [ ] On "Index Folder" click: call `index_folder()` and display summary via `st.info`.
- [ ] On file upload: open the uploaded bytes as a `PIL.Image`, call `model.embed_image()`, and run `db_client.search()` directly — no upsert.
- [ ] Build search bar: `st.text_input` for query + "Search" button.
- [ ] On search: call `search_photos()` and display results in a `st.columns(4)` grid with thumbnail, filename, similarity, and path tooltip.
- [ ] Handle empty results gracefully with `st.info("No results found.")`.
- [ ] Show `st.error` for any exception that reaches the UI layer.

**Relevant Context:**
- `MVP.md` §Planned MVP Workflow (5-step user journey).
- `MVP.md` §Module Responsibilities — `app.py` provides the Streamlit interface.
- Streamlit `@st.cache_resource` is the correct decorator for shared connections and models (not `@st.cache_data`).

**Status:** `[ ] pending`

---

## Implementation Order

The sub-tasks should be implemented in this sequence, as each builds on the previous:

```
Sub-Task 1 (Scaffold) → Sub-Task 2 (storage.py) → Sub-Task 3 (model.py)
    → Sub-Task 4 (files.py) → Sub-Task 5 (indexer.py)
    → Sub-Task 6 (search.py) → Sub-Task 7 (app.py)
```

Sub-Tasks 3 and 4 have no dependency on each other and can be implemented in either order after Sub-Task 2.
