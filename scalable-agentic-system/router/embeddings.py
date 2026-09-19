"""
Thin wrapper around the embedding model used for BOTH:
  1. Indexing tool descriptions into pgvector at registration time.
  2. Embedding the user's (rewritten) query at request time.

Kept as its own module so the embedding backend (OpenAI, local
sentence-transformers, Cohere, etc.) can be swapped without touching
the router logic.
"""
from __future__ import annotations
import numpy as np
from openai import OpenAI
from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def embed_text(text: str) -> list[float]:
    resp = _client.embeddings.create(model=settings.embedding_model, input=text)
    return resp.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batched embedding — used when bulk-registering tools (e.g. loading
    a 500-endpoint OpenAPI/Postman collection)."""
    resp = _client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) or 1e-9
    return float(np.dot(a_arr, b_arr) / denom)
