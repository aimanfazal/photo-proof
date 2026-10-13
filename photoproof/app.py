"""
app.py — PhotoProof Streamlit application.

Run with:
    streamlit run photoproof/app.py
"""

from __future__ import annotations

import csv
import io as _io
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so `from photoproof import …` resolves
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from PIL import Image

from photoproof import indexer, search as search_module
from photoproof.files import file_exists
from photoproof.model import embed_image, load_model
from photoproof.storage import StorageClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page configuration & custom styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhotoProof — Semantic Photo Search",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern CSS styling for cards, badges, and layout
st.markdown(
    """
    <style>
    /* Metric and card tweaks */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .photo-card {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 8px;
        background: rgba(255, 255, 255, 0.02);
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
        margin-bottom: 12px;
    }
    .photo-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.15);
    }
    .badge-sim {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #2e7d32;
        color: white;
    }
    .badge-sim-mid {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #f57f17;
        color: white;
    }
    .badge-sim-low {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #616161;
        color: white;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .hero-subtitle {
        color: gray;
        font-size: 1.05rem;
        margin-top: 2px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_source_type" not in st.session_state:
    st.session_state.search_source_type = None  # "text", "upload", "similar"
if "search_context" not in st.session_state:
    st.session_state.search_context = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "text"
if "selected_photo_path" not in st.session_state:
    st.session_state.selected_photo_path = None
if "selected_photo_info" not in st.session_state:
    st.session_state.selected_photo_info = None

# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading OpenCLIP model (ViT-B-32)…")
def get_model():
    """Load OpenCLIP model, preprocess transform, and tokenizer."""
    return load_model()


@st.cache_resource(show_spinner="Connecting to Tiger Data PostgreSQL…")
def get_db() -> StorageClient:
    """Connect to Tiger Data and ensure the schema exists."""
    client = StorageClient.connect()
    client.setup_schema()
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
THUMBNAIL_SIZE = (350, 350)


def _make_thumbnail(path: str) -> Image.Image | None:
    """Open a local image and resize it to a thumbnail for display."""
    try:
        img = Image.open(path)
        img.thumbnail(THUMBNAIL_SIZE)
        return img
    except Exception:  # noqa: BLE001
        return None


def _get_similarity_badge_html(similarity: float) -> str:
    """Render an HTML badge color-coded by similarity score."""
    pct = similarity * 100
    if similarity >= 0.70:
        cls = "badge-sim"
    elif similarity >= 0.50:
        cls = "badge-sim-mid"
    else:
        cls = "badge-sim-low"
    return f'<span class="{cls}">{pct:.1f}% match</span>'


def _format_size(size_bytes: int | None) -> str:
    """Format file size in human-readable units."""
    if size_bytes is None:
        return "Unknown size"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Resource Setup
# ---------------------------------------------------------------------------
try:
    clip_model, preprocess, tokenizer = get_model()
    db_client = get_db()
except Exception as exc:
    st.error(
        f"### ⚠ Startup Error\n\n**Details:** `{exc}`\n\n"
        "Ensure your `.env` file contains a valid `DATABASE_URL` with SSL mode set, "
        "and that your database instance is active."
    )
    st.stop()

# Get library stats
try:
    indexed_files = db_client.get_indexed_files()
    total_indexed_count = len(indexed_files)
except Exception:
    indexed_files = {}
    total_indexed_count = 0


# ---------------------------------------------------------------------------
# Sidebar: Library Management & Indexing
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📸 **PhotoProof**")
    st.markdown(
        "<p style='color: gray; font-size: 0.9rem; margin-top: -10px;'>"
        "Semantic AI photo search powered by OpenCLIP & Tiger Data."
        "</p>",
        unsafe_allow_html=True,
    )

    # Database quick metric card
    st.metric(label="Total Photos Indexed", value=f"{total_indexed_count:,}")

    st.divider()

    # Indexing Accordion / Controls
    st.subheader("📁 Index Photo Folder")
    with st.expander("Folder Ingestion", expanded=total_indexed_count == 0):
        st.write(
            "Scan a folder to extract embeddings. Files already indexed with the same timestamp will be automatically skipped."
        )
        folder_input = st.text_input(
            "Folder directory path",
            placeholder="/path/to/photos or E:\\Photos",
            help="Enter the absolute path to your local photo collection.",
            key="input_folder_path",
        )

        col_idx_btn, col_clear_btn = st.columns([1, 1])
        with col_idx_btn:
            start_index = st.button("🚀 Start Indexing", type="primary", use_container_width=True)

        if start_index:
            if not folder_input.strip():
                st.warning("Please provide a valid folder path.")
            else:
                progress_bar = st.progress(0.0, text="Scanning directory…")
                status_text = st.empty()
                try:
                    def _on_progress(done: int, total: int, filename: str) -> None:
                        frac = done / total if total > 0 else 1.0
                        progress_bar.progress(frac, text=f"Indexing ({done}/{total})")
                        status_text.caption(f"Current: `{filename}`")

                    summary = indexer.index_folder(
                        folder_input.strip(),
                        clip_model,
                        preprocess,
                        db_client,
                        progress_callback=_on_progress,
                    )
                    progress_bar.progress(1.0, text="Completed!")
                    status_text.empty()

                    st.success(
                        f"**Scan Complete!**\n\n"
                        f"- 🔎 **{summary['total']}** discovered\n"
                        f"- 📥 **{summary['indexed']}** newly indexed\n"
                        f"- ⏭ **{summary['skipped']}** unchanged\n"
                        f"- ❌ **{summary['failed']}** failed"
                    )

                    if summary.get("errors"):
                        with st.expander(f"Failed files ({len(summary['errors'])})"):
                            for err in summary["errors"]:
                                st.markdown(f"- `{err['path']}`\n  *{err['reason']}*")

                    # Force refresh of stats
                    st.rerun()

                except Exception as exc:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Error during indexing: {exc}")

    st.divider()
    st.caption("PhotoProof • Local-first Semantic Search")


# ---------------------------------------------------------------------------
# Main Area Header
# ---------------------------------------------------------------------------
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown('<div class="hero-title">PhotoProof 📸</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Find the moment you remember, not the filename you forgot.</div>',
        unsafe_allow_html=True,
    )
with header_col2:
    if st.session_state.search_results:
        if st.button("🔄 Clear Search", use_container_width=True):
            st.session_state.search_results = []
            st.session_state.search_context = ""
            st.session_state.search_source_type = None
            st.session_state.selected_photo_path = None
            st.session_state.selected_photo_info = None
            st.rerun()


# ---------------------------------------------------------------------------
# Search Modalities (Tabs)
# ---------------------------------------------------------------------------
tab_text, tab_image, tab_similar = st.tabs([
    "💬 Search by Description",
    "🖼 Search by Image",
    "🔗 Find Similar to Indexed Photo",
])

# ---- TAB 1: Natural Language Text Search ----
with tab_text:
    # Use a form so pressing [Enter] triggers the search immediately!
    with st.form("text_search_form", clear_on_submit=False):
        col_q, col_limit = st.columns([4, 1])
        with col_q:
            query_input = st.text_input(
                "Query description",
                placeholder="e.g. A dog playing frisbee in golden hour light, or family birthday party",
                label_visibility="collapsed",
            )
        with col_limit:
            limit_val = st.number_input(
                "Max results", min_value=1, max_value=100, value=16, step=4, label_visibility="collapsed"
            )

        submit_text = st.form_submit_button("🔍 Search Library", type="primary", use_container_width=True)

    # Prompt chips / suggestions
    st.markdown(
        "<p style='font-size:0.85rem; color:gray; margin-top:-6px;'>"
        "💡 Suggestions: <code>sunset over mountains</code> • <code>person smiling in sunglasses</code> • "
        "<code>latte art on a wooden table</code> • <code>vintage car on empty road</code></p>",
        unsafe_allow_html=True,
    )

    if submit_text:
        if not query_input.strip():
            st.warning("Please enter a description to search.")
        else:
            with st.spinner(f"Searching photos matching '{query_input.strip()}'…"):
                try:
                    res = search_module.search_photos(
                        query_input.strip(),
                        clip_model,
                        tokenizer,
                        db_client,
                        limit=int(limit_val),
                    )
                    st.session_state.search_results = res
                    st.session_state.search_source_type = "text"
                    st.session_state.search_context = query_input.strip()
                    st.session_state.selected_photo_path = None
                    st.session_state.selected_photo_info = None
                except Exception as exc:
                    st.error(f"Search failed: {exc}")

# ---- TAB 2: Reverse Image Search ----
with tab_image:
    st.markdown("Upload any photo to find visually or semantically similar images in your library.")
    upload_col1, upload_col2 = st.columns([3, 1])
    with upload_col1:
        uploaded_file = st.file_uploader(
            "Upload image query",
            type=["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"],
            label_visibility="collapsed",
        )
    with upload_col2:
        img_limit = st.number_input(
            "Max results (image)", min_value=1, max_value=100, value=16, step=4, label_visibility="collapsed"
        )
        submit_upload = st.button("🔎 Find Matches", type="primary", use_container_width=True)

    if submit_upload and uploaded_file is not None:
        with st.spinner("Analyzing uploaded image & searching library…"):
            try:
                img_bytes = uploaded_file.read()
                pil_image = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
                query_embedding = embed_image(clip_model, preprocess, pil_image)
                res = db_client.search(query_embedding, limit=int(img_limit))
                st.session_state.search_results = res
                st.session_state.search_source_type = "upload"
                st.session_state.search_context = uploaded_file.name
                st.session_state.selected_photo_path = None
                st.session_state.selected_photo_info = None
            except Exception as exc:
                st.error(f"Image search error: {exc}")

# ---- TAB 3: Similar to Existing Photo ----
with tab_similar:
    st.markdown("Select or type the path of an indexed photo to retrieve similar images without re-encoding.")
    with st.form("similar_search_form"):
        col_sim_path, col_sim_limit = st.columns([4, 1])
        with col_sim_path:
            sim_path_input = st.text_input(
                "Indexed file path",
                placeholder="e.g. /home/user/Pictures/IMG_0042.jpg",
                label_visibility="collapsed",
            )
        with col_sim_limit:
            sim_limit_val = st.number_input(
                "Max similar results", min_value=1, max_value=100, value=16, step=4, label_visibility="collapsed"
            )
        submit_sim = st.form_submit_button("✨ Find Visually Similar", type="primary", use_container_width=True)

    if submit_sim:
        if not sim_path_input.strip():
            st.warning("Please enter a file path.")
        else:
            with st.spinner("Finding visually similar photos…"):
                try:
                    res = search_module.search_similar(
                        sim_path_input.strip(), db_client, limit=int(sim_limit_val)
                    )
                    if not res:
                        st.warning("Photo not found in index. Ensure the photo was indexed first.")
                    else:
                        st.session_state.search_results = res
                        st.session_state.search_source_type = "similar"
                        st.session_state.search_context = Path(sim_path_input.strip()).name
                        st.session_state.selected_photo_path = None
                        st.session_state.selected_photo_info = None
                except Exception as exc:
                    st.error(f"Similar search error: {exc}")


# ---------------------------------------------------------------------------
# Results Section & Toolbar
# ---------------------------------------------------------------------------
st.divider()

results = st.session_state.search_results
source_type = st.session_state.search_source_type
context_label = st.session_state.search_context

if not results:
    if total_indexed_count == 0:
        st.info("👋 **Welcome to PhotoProof!** Start by entering a local photo folder in the sidebar to index your library.")
    else:
        st.info(f"✨ Ready to search across **{total_indexed_count:,}** indexed photos. Enter a description or upload a photo above.")
else:
    # Context Header
    if source_type == "text":
        st.subheader(f"Results for: “{context_label}”")
    elif source_type == "upload":
        st.subheader(f"Results matching uploaded image: “{context_label}”")
    elif source_type == "similar":
        st.subheader(f"Photos visually similar to: “{context_label}”")

    # Filter & Refinement Bar
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2.5, 2, 1.2, 1.3])

    with filter_col1:
        name_filter = st.text_input(
            "Filter by filename",
            placeholder="e.g. DSC_ or .png",
            key="ui_name_filter",
        )

    with filter_col2:
        min_sim = st.slider(
            "Minimum similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key="ui_min_sim",
        )

    with filter_col3:
        gallery_cols = st.selectbox(
            "Grid columns",
            options=[3, 4, 5, 6],
            index=1,
            key="ui_grid_cols",
        )

    # Apply client-side filters
    filtered_results = [
        r for r in results
        if (not name_filter or name_filter.lower() in r["filename"].lower())
        and r["similarity"] >= min_sim
    ]

    with filter_col4:
        st.write("")  # alignment spacer
        if filtered_results:
            buf = _io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["rank", "filename", "similarity", "path"])
            writer.writeheader()
            for rank, row in enumerate(filtered_results, start=1):
                writer.writerow({
                    "rank": rank,
                    "filename": row["filename"],
                    "similarity": f"{row['similarity']:.6f}",
                    "path": row["path"],
                })
            st.download_button(
                label="⬇ Export CSV",
                data=buf.getvalue(),
                file_name="photoproof_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.caption(f"Showing **{len(filtered_results)}** of {len(results)} matches")

    # -----------------------------------------------------------------------
    # Inspect / Details Modal / Expander if a photo was selected
    # -----------------------------------------------------------------------
    if st.session_state.selected_photo_path:
        sel_path = st.session_state.selected_photo_path
        sel_info = st.session_state.selected_photo_info or {}
        with st.expander(f"🔍 Inspect Photo: {Path(sel_path).name}", expanded=True):
            col_modal_img, col_modal_meta = st.columns([1, 1])
            with col_modal_img:
                if file_exists(sel_path):
                    st.image(sel_path, use_container_width=True)
                else:
                    st.warning("File no longer exists on disk.")
            with col_modal_meta:
                st.markdown(f"### {Path(sel_path).name}")
                st.markdown(f"**Similarity score:** {sel_info.get('similarity', 0):.2%}")
                st.code(sel_path, language="text")

                btn_find_more, btn_close = st.columns(2)
                with btn_find_more:
                    if st.button("✨ Search Similar to This", key=f"inspect_sim_{sel_path}"):
                        try:
                            sim_res = search_module.search_similar(sel_path, db_client, limit=16)
                            st.session_state.search_results = sim_res
                            st.session_state.search_source_type = "similar"
                            st.session_state.search_context = Path(sel_path).name
                            st.session_state.selected_photo_path = None
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Search similar error: {exc}")
                with btn_close:
                    if st.button("✖ Close Inspector", key=f"inspect_close_{sel_path}"):
                        st.session_state.selected_photo_path = None
                        st.session_state.selected_photo_info = None
                        st.rerun()

    # -----------------------------------------------------------------------
    # Gallery Grid
    # -----------------------------------------------------------------------
    if not filtered_results:
        st.warning("No photos match the current filter criteria. Adjust the filename or similarity slider.")
    else:
        cols = st.columns(gallery_cols)
        for idx, item in enumerate(filtered_results):
            col = cols[idx % gallery_cols]
            path = item["path"]
            sim = item["similarity"]
            filename = item["filename"]

            with col:
                st.markdown('<div class="photo-card">', unsafe_allow_html=True)

                if not file_exists(path):
                    st.warning(f"⚠ Missing from disk:\n`{filename}`")
                    try:
                        db_client.delete_photo(path)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not delete missing photo %s: %s", path, exc)
                    st.markdown('</div>', unsafe_allow_html=True)
                    continue

                thumb = _make_thumbnail(path)
                if thumb:
                    st.image(thumb, use_container_width=True)
                else:
                    st.warning("⚠ Failed to load image")

                # Badge + File Info
                badge_html = _get_similarity_badge_html(sim)
                st.markdown(
                    f"<div style='margin-top: 4px; display: flex; justify-content: space-between; align-items: center;'>"
                    f"<span style='font-size: 0.85rem; font-weight: 600; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;'>{filename}</span>"
                    f"{badge_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Card Actions
                act_col1, act_col2 = st.columns(2)
                with act_col1:
                    if st.button("🔍 Inspect", key=f"view_{idx}_{path}", use_container_width=True):
                        st.session_state.selected_photo_path = path
                        st.session_state.selected_photo_info = item
                        st.rerun()
                with act_col2:
                    if st.button("✨ Similar", key=f"sim_{idx}_{path}", use_container_width=True):
                        try:
                            sim_res = search_module.search_similar(path, db_client, limit=16)
                            st.session_state.search_results = sim_res
                            st.session_state.search_source_type = "similar"
                            st.session_state.search_context = filename
                            st.session_state.selected_photo_path = None
                            st.session_state.selected_photo_info = None
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Search similar error: {exc}")

                st.markdown('</div>', unsafe_allow_html=True)
