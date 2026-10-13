"""
app.py — PhotoProof Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging

import streamlit as st
from PIL import Image

from photoproof import indexer, search as search_module
from photoproof.files import file_exists
from photoproof.model import embed_image, load_model
from photoproof.storage import StorageClient

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhotoProof",
    page_icon="📸",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources — loaded once per Streamlit session
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading AI model…")
def get_model():
    """Load OpenCLIP model, preprocess transform, and tokenizer."""
    return load_model()


@st.cache_resource(show_spinner="Connecting to database…")
def get_db() -> StorageClient:
    """Connect to Tiger Data and ensure the schema exists."""
    client = StorageClient.connect()
    client.setup_schema()
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THUMBNAIL_SIZE = (300, 300)
GALLERY_COLUMNS = 4


def _make_thumbnail(path: str) -> Image.Image | None:
    """Open a local image and resize it to a thumbnail for display."""
    try:
        img = Image.open(path)
        img.thumbnail(THUMBNAIL_SIZE)
        return img
    except Exception:  # noqa: BLE001
        return None


def _show_gallery(results: list[dict], db_client: StorageClient) -> None:
    """Render search results as a photo grid.

    Detects missing files, shows a clear message, and removes stale DB rows.
    """
    if not results:
        st.info("No results found. Try a different query or index more photos.")
        return

    cols = st.columns(GALLERY_COLUMNS)
    for idx, result in enumerate(results):
        col = cols[idx % GALLERY_COLUMNS]
        path = result["path"]
        with col:
            if not file_exists(path):
                st.warning(
                    f"⚠ File not found — removed from index  \n`{path}`"
                )
                try:
                    db_client.delete_photo(path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not remove stale record %s — %s", path, exc)
                continue

            thumb = _make_thumbnail(path)
            if thumb:
                st.image(thumb, use_container_width=True)
            else:
                st.warning("⚠ Could not load thumbnail")
            st.caption(
                f"**{result['filename']}**  \n"
                f"Similarity: {result['similarity']:.2%}  \n"
                f"`{path}`"
            )


# ---------------------------------------------------------------------------
# Sidebar — indexing controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📸 PhotoProof")
    st.markdown("*Find the photo you remember, not the filename you forgot.*")
    st.divider()

    # ---- Folder indexing ----
    st.subheader("Index a folder")
    folder_path = st.text_input(
        "Folder path",
        placeholder="/Users/you/Photos",
        help="Enter the absolute path to a local folder containing photos.",
    )
    index_button = st.button("Index Folder", type="primary", use_container_width=True)

    st.divider()

    # ---- Single-image upload (search-by-image) ----
    st.subheader("Search by image")
    uploaded_file = st.file_uploader(
        "Upload a photo to find similar images",
        type=["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"],
        help="The uploaded image is used as a search query — it is not stored.",
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.header("PhotoProof")

# ---- Initialise resources (errors surfaced early) ----
try:
    clip_model, preprocess, tokenizer = get_model()
    db_client = get_db()
except Exception as exc:
    st.error(
        f"**Startup error:** {exc}\n\n"
        "Check that `DATABASE_URL` and `SSL_MODE` are set in your `.env` file "
        "and that your Tiger Data instance is reachable."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Handle: Index Folder
# ---------------------------------------------------------------------------

if index_button:
    if not folder_path:
        st.warning("Please enter a folder path first.")
    else:
        progress_bar = st.progress(0.0, text="Scanning folder…")
        status_text = st.empty()
        try:
            def _on_progress(done: int, total: int, filename: str) -> None:
                frac = done / total if total > 0 else 1.0
                progress_bar.progress(frac, text=f"Indexing {done}/{total}: {filename}")
                status_text.text(filename)

            summary = indexer.index_folder(
                folder_path, clip_model, preprocess, db_client,
                progress_callback=_on_progress,
            )
            progress_bar.progress(1.0, text="Done")
            status_text.empty()
            st.info(
                f"✅ Indexing complete — "
                f"**{summary['total']}** found, "
                f"**{summary['indexed']}** indexed, "
                f"**{summary['skipped']}** skipped (unchanged), "
                f"**{summary['failed']}** failed."
            )
            if summary.get("errors"):
                with st.expander(f"⚠ Failed files ({len(summary['errors'])}) — click to expand"):
                    for err in summary["errors"]:
                        st.markdown(f"- `{err['path']}`  \n  _{err['reason']}_")
        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Indexing error: {exc}")

# ---------------------------------------------------------------------------
# Handle: Search by uploaded image
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    st.subheader("Results — image search")
    try:
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(pil_image, caption="Query image", width=200)

        with st.spinner("Searching for similar photos…"):
            query_embedding = embed_image(clip_model, preprocess, pil_image)
            results = db_client.search(query_embedding, limit=10)

        _show_gallery(results, db_client)
    except Exception as exc:
        st.error(f"Image search error: {exc}")

# ---------------------------------------------------------------------------
# Handle: Natural-language search
# ---------------------------------------------------------------------------

st.subheader("Search by description")

col_input, col_btn = st.columns([5, 1])
with col_input:
    query_text = st.text_input(
        "Search query",
        placeholder="A dog running on the beach at sunset",
        label_visibility="collapsed",
    )
with col_btn:
    search_button = st.button("Search", type="primary", use_container_width=True)

if search_button:
    if not query_text.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching…"):
            try:
                results = search_module.search_photos(
                    query_text, clip_model, tokenizer, db_client, limit=12
                )
                _show_gallery(results, db_client)
            except Exception as exc:
                st.error(f"Search error: {exc}")
