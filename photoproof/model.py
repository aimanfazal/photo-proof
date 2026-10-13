"""
photoproof/model.py
Load OpenCLIP and generate L2-normalised image and text embeddings.

Default model: ViT-B-32 / openai pretrained weights (512-d embeddings).
The caller is responsible for caching the loaded model (e.g. @st.cache_resource).
"""

from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image

# Model identifiers — change here if you want to swap to a different checkpoint.
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def load_model() -> tuple:
    """Load the OpenCLIP model, image preprocessor, and text tokenizer.

    Returns:
        (model, preprocess, tokenizer)
        - model: OpenCLIP model in eval mode on the best available device.
        - preprocess: torchvision transform to apply to PIL Images before encoding.
        - tokenizer: callable that converts text strings to token tensors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


def embed_image(model, preprocess, image: Image.Image) -> np.ndarray:
    """Return an L2-normalised 512-d float32 embedding for a PIL Image.

    Args:
        model: loaded OpenCLIP model (from load_model).
        preprocess: image transform (from load_model).
        image: a PIL Image (will be converted to RGB internally if needed).

    Returns:
        1-D float32 numpy array of shape (512,), L2-normalised.
    """
    device = next(model.parameters()).device
    image_rgb = image.convert("RGB")
    tensor = preprocess(image_rgb).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        features = model.encode_image(tensor)  # (1, 512)
    vec = features.squeeze(0).cpu().float().numpy()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_text(model, tokenizer, text: str) -> np.ndarray:
    """Return an L2-normalised 512-d float32 embedding for a text string.

    Args:
        model: loaded OpenCLIP model (from load_model).
        tokenizer: tokenizer callable (from load_model).
        text: natural-language query string.

    Returns:
        1-D float32 numpy array of shape (512,), L2-normalised.
    """
    device = next(model.parameters()).device
    tokens = tokenizer([text]).to(device)  # (1, context_length)
    with torch.no_grad():
        features = model.encode_text(tokens)  # (1, 512)
    vec = features.squeeze(0).cpu().float().numpy()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec
