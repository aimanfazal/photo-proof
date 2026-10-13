# PhotoProof 📸

**Find the photo you remember, not the filename you forgot.**

PhotoProof is a local-first, AI-powered photo search application that helps photographers find images using natural-language descriptions. Instead of manually browsing thousands of files or tagging every photo, users can search for things like *"a dog running on the beach at sunset"* and retrieve visually relevant images.

PhotoProof combines open-weight AI with Tiger Data's PostgreSQL and `pgvector` capabilities to build a semantic image search engine.

> **Project status:** MVP in development.

## ✨ Features

The initial MVP focuses on four essential capabilities:

- **Natural-language photo search:** Search images using ordinary text descriptions.
- **AI-powered image matching:** Use OpenCLIP embeddings to rank images by semantic similarity.
- **Persistent vector search:** Store embeddings and photo metadata in Tiger Data using PostgreSQL and `pgvector`.
- **Simple photo gallery:** Browse matching thumbnails, filenames, and original file paths through a Streamlit interface.

### Planned MVP workflow

1. Select a local folder containing photos.
2. Index supported images and generate embeddings.
3. Enter a natural-language search query.
4. Retrieve the most relevant photos.
5. Open a matching image using its original file path.

## 🎯 The Problem

Photographers and small studios often maintain large photo libraries containing thousands of images with unhelpful filenames, such as `IMG_1048.JPG`.

Finding a particular moment can require manually opening folders, scanning thumbnails, or tagging images individually.

PhotoProof aims to make photo retrieval faster by searching for visual meaning instead of relying solely on filenames and manually maintained metadata.

**Target users:** Freelance photographers, photography studios, event photographers, and people managing large personal photo archives.

## 🧠 How It Works

PhotoProof uses a vision-language model to represent photos and search queries in a shared embedding space.

```text
Local Photo Folder
        |
        v
Image Scanner and Preprocessor
        |
        v
OpenCLIP Image Encoder
        |
        v
Image Embeddings + Metadata
        |
        v
Tiger Data PostgreSQL + pgvector
        ^
        |
OpenCLIP Text Encoder
        ^
        |
Natural-Language Search Query
        |
        v
Vector Similarity Search
        |
        v
Ranked Photo Gallery
```

### Search pipeline

1. The application scans supported images in the selected folder.
2. OpenCLIP generates an embedding for each image.
3. PhotoProof stores each embedding alongside its file path and metadata in Tiger Data.
4. The user enters a text query.
5. OpenCLIP converts the query into a compatible text embedding.
6. PostgreSQL and `pgvector` rank image embeddings by cosine distance.
7. Streamlit displays the top matching photos.

Image embeddings should be generated once and reused when the source files have not changed.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | User interface and photo gallery |
| OpenCLIP | Image and text embeddings |
| PyTorch | Model inference |
| Pillow | Image loading and thumbnail processing |
| NumPy | Vector normalization and supporting operations |
| PostgreSQL | Persistent storage |
| Tiger Data | Managed PostgreSQL infrastructure |
| pgvector | Vector storage and similarity search |
| VS Code MCP | Development-time interaction with Tiger Data |

The first version will use direct vector similarity search. Hybrid keyword and vector search may be added later if it improves retrieval quality.

## 🐅 Tiger Data Integration

Tiger Data is a core component of PhotoProof's retrieval architecture.

The application uses PostgreSQL with `pgvector` to store image embeddings and associated metadata. This enables persistent vector search without requiring a separate vector database service.

The database will store information such as:

- Image identifier
- Original file path
- Filename
- File size and modification time
- Embedding model identifier
- Image embedding
- Optional image description

A simplified example schema:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE photos (
    id BIGSERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_size BIGINT,
    modified_time DOUBLE PRECISION,
    model_id TEXT NOT NULL,
    description TEXT,
    embedding VECTOR(512)
);
```

The embedding dimension shown above is illustrative. It must match the output dimension of the selected OpenCLIP model.

A simplified semantic search query:

```sql
SELECT
    id,
    path,
    filename,
    1 - (embedding <=> $1::vector) AS similarity
FROM photos
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

The application will generate the query embedding locally and pass it to PostgreSQL for retrieval.

### Development-time MCP integration

PhotoProof can use Tiger Data's MCP integration through VS Code to help a coding agent inspect database services, explore the schema, and perform supported development tasks.

Example project configuration:

```text
.vscode/
└── mcp.json
```

Example `.vscode/mcp.json` configuration, subject to the installed Tiger CLI version and its authentication requirements:

```json
{
  "servers": {
    "tiger": {
      "type": "stdio",
      "command": "tiger",
      "args": ["mcp", "start"]
    }
  }
}
```

Refer to the [Tiger CLI repository](https://github.com/timescale/tiger-cli) for the current installation and MCP setup instructions. If the official installer generates a different configuration, use its supported configuration rather than maintaining conflicting entries.

MCP is a development tool; the running PhotoProof application connects directly to PostgreSQL through a Python database driver.

## 🔓 Why Open Innovation Matters

PhotoProof is built around open-weight AI and an interoperable database architecture.

### Local AI inference

OpenCLIP generates image and text embeddings locally. The application does not need a hosted model inference API for every search.

### Model flexibility

The embedding model can be replaced or evaluated against alternatives. A model change requires regenerating image embeddings to ensure they are compatible with the new text encoder.

### No per-query model API fees

Local inference avoids a hosted model provider's per-request inference charges. Hardware, electricity, storage, and database hosting may still incur costs.

### Privacy-conscious design

Original photos can remain on the user's computer while their embeddings and metadata are stored in Tiger Data.

This is a hybrid architecture, not a fully offline system. The hosted database requires network access, and embeddings and metadata may reveal sensitive information. Users should understand what leaves their device.

### Open-source ecosystem

The project builds on openly available software and model implementations, allowing developers to inspect, adapt, and extend the system.

Licensing varies between code and pretrained weights. Verify the selected model's license before redistributing it or using it commercially.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the chosen dependencies
- Git
- VS Code (recommended)
- A Tiger Data PostgreSQL database with `pgvector` available
- Sufficient RAM or GPU resources to run the selected OpenCLIP model

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd photoproof
```

Replace `<your-repository-url>` with your repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Once `requirements.txt` has been created:

```bash
pip install -r requirements.txt
```

The project will require compatible versions of Streamlit, OpenCLIP, PyTorch, Pillow, NumPy, and a PostgreSQL driver such as `psycopg`.

### 4. Configure Tiger Data

Create a Tiger Data PostgreSQL service and enable `pgvector` if it is not already available.

Set the database connection string as an environment variable. For example, create a local `.env` file:

```dotenv
DATABASE_URL=your_postgresql_connection_string
```

Never commit this file or expose its contents in logs, screenshots, or public repositories.

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.streamlit/secrets.toml
```

The application should read the connection string from the environment and use TLS settings appropriate for the database provider.

### 5. Start the application

Once `app.py` is implemented:

```bash
streamlit run app.py
```

The browser interface should allow the user to index a local photo folder and search the resulting library.

**Note:** These are setup instructions for the intended MVP architecture. A complete runnable application, dependency lockfile, and database migration must be committed before the repository can be considered ready for users.

## 📁 Project Structure

```text
photoproof/
├── .vscode/
│   └── mcp.json
├── photoproof/
│   ├── __init__.py
│   ├── model.py
│   ├── indexer.py
│   ├── search.py
│   ├── storage.py
│   └── files.py
├── tests/
│   ├── test_search.py
│   └── test_indexer.py
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── mvp.md
└── README.md
```

### Module responsibilities

- `model.py`: Load OpenCLIP and generate image and text embeddings.
- `indexer.py`: Scan folders, preprocess images, and index files.
- `search.py`: Execute similarity search and return ranked results.
- `storage.py`: Manage PostgreSQL connections and database operations.
- `files.py`: Validate image paths and open original images.
- `app.py`: Provide the Streamlit interface.

## 🔐 Privacy and Security

PhotoProof aims to minimize unnecessary exposure of personal photo libraries.

- Original images remain in their local folders.
- Image and text embeddings are generated locally.
- Embeddings and metadata are sent to Tiger Data for hosted vector search.
- Database credentials are kept in environment variables.
- The database should use authenticated, encrypted connections.
- Database access should follow least-privilege principles.
- The application should not expose arbitrary database operations to end users.
- Avoid logging sensitive paths, queries, or database credentials unnecessarily.

Because the database is hosted, PhotoProof requires an internet connection for database-backed search. A future local-only mode could use SQLite and NumPy or a local vector index instead.

## 🧪 Testing and Evaluation

The MVP should be tested against a small collection of varied images and manually reviewed search queries.

Example queries:

- "A dog playing in the grass"
- "A sunset over the ocean"
- "A person holding a red umbrella"
- "A building at night"

Evaluation should measure:

- **Recall@5:** Whether a relevant image appears among the top five results.
- **Precision@5:** How many of the top five results are relevant.
- **Search latency:** Time from submitting a query to displaying results.
- **Indexing time:** Time required to index a known image collection.
- **Incremental indexing:** Whether unchanged images avoid unnecessary re-embedding.

These tests help establish whether the system is useful rather than relying on a few handpicked examples.

## 🗺️ Development Roadmap

### Phase 1 — Core MVP

- [ ] Load a compatible OpenCLIP model.
- [ ] Scan and index local images.
- [ ] Generate and persist image embeddings.
- [ ] Configure Tiger Data and `pgvector`.
- [ ] Implement text-to-image similarity search.
- [ ] Build the Streamlit gallery.
- [ ] Display correct filenames and original paths.

### Phase 2 — Reliability

- [ ] Cache embeddings and avoid unnecessary re-indexing.
- [ ] Handle corrupt files and missing paths.
- [ ] Add indexing progress and error reporting.
- [ ] Test retrieval quality on a labeled dataset.
- [ ] Verify that original photos are not uploaded.

### Phase 3 — Possible Extensions

- [ ] Hybrid keyword and vector search.
- [ ] Optional model-generated image descriptions.
- [ ] Similar-photo search.
- [ ] Additional filters and batch export.
- [ ] A separate local-only storage backend.
- [ ] Improved packaging and installation.

These extensions are intentionally excluded from the initial MVP.

## 🏆 Hacktoberfest Open-Source AI Challenge

PhotoProof is being developed for the Hacktoberfest Open-Source AI Challenge — Week 2: Side Hustle.

The project explores how open-weight AI and vector search can solve a practical problem for photographers.

### Intended partner category

**Tiger Data — Best Use of Tiger Data**

The planned integration uses Tiger Data as the application's PostgreSQL data layer and `pgvector` as the semantic retrieval engine.

The final submission should demonstrate the actual database integration, show representative search results, and explain the trade-offs between local inference and hosted vector storage.

No prize eligibility or award is implied by the planned integration.

## 🤝 Contributing

Contributions, ideas, bug reports, and feedback are welcome.

Potential contribution areas include:
- Improving retrieval accuracy
- Testing different compatible OpenCLIP models
- Optimizing indexing performance
- Improving accessibility and usability
- Expanding automated tests
- Improving documentation

Please open an issue before making substantial changes.

## 📄 License

A project license has not yet been selected. Before public release, choose an appropriate license for the application code and verify the licenses of all model weights and dependencies.

## 🙌 Acknowledgements

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Tiger Data](https://www.tigerdata.com/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Tiger CLI](https://github.com/timescale/tiger-cli)

---

**PhotoProof — Find the moment, not the filename.**
