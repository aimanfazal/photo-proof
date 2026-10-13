# PhotoProof 📸

> **Find the photo you remember, not the filename you forgot.**

PhotoProof is a local-first, AI-powered photo search application. Instead of browsing thousands of files or manually tagging images, describe what you're looking for in plain English — *"a dog running on the beach at sunset"* — and PhotoProof retrieves visually relevant images from your library.

It combines [OpenCLIP](https://github.com/mlfoundations/open_clip) embeddings with [Tiger Data](https://www.tigerdata.com/) PostgreSQL + `pgvector` for persistent semantic search, all through a [Streamlit](https://streamlit.io/) browser interface.

---

## Features

- **Natural-language search** — describe a photo in plain English.
- **Search by image** — upload a photo to find visually similar indexed images.
- **Similar-photo search** — pick any already-indexed photo and find its visual neighbours.
- **Incremental indexing** — unchanged photos are skipped; re-indexing is only triggered when files change or the model is swapped.
- **Missing-path cleanup** — deleted or moved files are detected at query time and removed from the index automatically.
- **Result filtering** — filter by filename substring and minimum similarity threshold.
- **CSV export** — download any set of search results as a CSV.
- **Local AI inference** — embeddings are generated on your machine; no per-query model API fees.

---

## Prerequisites

- Python 3.10 or later
- Git
- A [Tiger Data](https://www.tigerdata.com/) PostgreSQL service with `pgvector` enabled
  *(or any PostgreSQL ≥ 14 instance with the `pgvector` extension)*
- Enough RAM to load ViT-B-32 (~350 MB); a GPU is optional but speeds up indexing

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd photo-proof
```

### 2. Install with Make (recommended)

```bash
make install
```

This creates a `.venv` virtual environment and installs all dependencies including dev tools.

**Manual alternative:**

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

---

## Configuration

Copy the example environment file and fill in your database connection string:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
DATABASE_URL=your_postgresql_connection_string
SSL_MODE=require          # use 'disable' for a plain local PostgreSQL instance
```

> **Never commit `.env` or share your connection string publicly.**

---

## Running

```bash
make run
```

Or manually:

```bash
.venv/Scripts/python -m streamlit run photoproof/app.py   # Windows
# source .venv/bin/activate && streamlit run photoproof/app.py  # macOS/Linux
```

The browser opens automatically at `http://localhost:8501`.

---

## Usage

### Indexing photos

1. Enter the **absolute path** to a folder containing photos in the sidebar.
2. Click **Index Folder**.
3. A progress bar tracks each file. A summary shows how many were indexed, skipped, or failed.
4. Unchanged files are skipped automatically on subsequent runs.

### Searching

- **By description**: type a natural-language query in the main search bar and click **Search**.
- **By image upload**: upload any image in the sidebar — it is used as a visual query (not stored).
- **By similar photo**: enter the path of any indexed photo in the sidebar and click **Find Similar**.

### Filtering and exporting

After any search, use the filter controls to narrow results by filename or minimum similarity. Click **Download results as CSV** to save the current results.

---

## Running Tests

```bash
make test
```

To run the retrieval quality test against a real labeled dataset:

```bash
.venv/Scripts/python -m pytest tests/test_search.py --dataset-path /path/to/dataset -v
```

The dataset folder should contain sub-folders named after their label:

```
dataset/
├── sunset/
│   ├── img1.jpg
│   └── img2.jpg
└── dogs/
    └── img3.jpg
```

---

## How It Works

```
Local Photo Folder
        │
        ▼
Image Scanner (files.py)
        │
        ▼
OpenCLIP Image Encoder (model.py)
        │
        ▼
512-d L2-normalised embedding
        │
        ▼
Tiger Data PostgreSQL + pgvector (storage.py)
        ▲
        │
OpenCLIP Text Encoder
        ▲
        │
Natural-language query  ──►  Vector similarity search  ──►  Ranked gallery
```

1. `files.py` scans the folder and discovers supported images.
2. `model.py` encodes each image into a 512-dimensional embedding using ViT-B-32 / OpenAI weights.
3. `storage.py` upserts embeddings and metadata into PostgreSQL with `pgvector`.
4. At query time, the text query (or uploaded/indexed image) is encoded and ranked by cosine distance.
5. `app.py` renders the top results in a Streamlit gallery.

---

## Project Structure

```
photo-proof/
├── .vscode/mcp.json          Tiger Data MCP config for VS Code
├── docs/
│   ├── mvp.md                Original MVP spec and project rationale
│   ├── plan-phase1.md        Phase 1 implementation plan
│   ├── plan-phase2.md        Phase 2 reliability plan
│   └── plan-phase3.md        Phase 3 features plan
├── photoproof/
│   ├── __init__.py
│   ├── _cli.py               Console script entry point
│   ├── app.py                Streamlit application
│   ├── model.py              Load OpenCLIP; embed images and text
│   ├── indexer.py            Scan, embed, and persist photos
│   ├── search.py             Text search and similar-photo search
│   ├── storage.py            PostgreSQL / pgvector interactions
│   └── files.py              Filesystem utilities
├── tests/
│   ├── conftest.py           Fixtures and CLI options
│   ├── test_indexer.py       Indexer unit tests
│   └── test_search.py        Search unit and quality tests
├── pyproject.toml            Package metadata and dependencies
├── requirements.txt          Pip-compatible dependency list
├── Makefile                  Developer shortcuts
├── .env.example              Environment variable template
└── .gitignore
```

---

## Privacy and Security

- Original photos stay in their local folders — only embeddings and metadata are sent to the database.
- Embeddings are generated locally; no image data leaves your machine.
- Database credentials are read from environment variables; never hard-coded.
- The database connection uses TLS by default (`SSL_MODE=require`).

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Application logic |
| Streamlit | Browser UI and gallery |
| OpenCLIP (ViT-B-32) | Image and text embeddings |
| PyTorch | Model inference |
| Pillow | Image loading and thumbnails |
| NumPy | Vector normalisation |
| PostgreSQL + pgvector | Persistent vector search |
| Tiger Data | Managed PostgreSQL infrastructure |
| psycopg v3 | PostgreSQL driver |
| python-dotenv | Environment variable loading |

---

## License

A project license has not yet been selected. Before public release, choose a license for the application code and verify the licenses of all model weights and dependencies.

---

## Acknowledgements

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [Tiger Data](https://www.tigerdata.com/)
- [pgvector](https://github.com/pgvector/pgvector)

---

*PhotoProof — Find the moment, not the filename.*
