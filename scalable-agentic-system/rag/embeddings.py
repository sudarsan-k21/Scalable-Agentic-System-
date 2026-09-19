"""Configures the embedding model LlamaIndex uses for the RAG pipeline
(kept separate from router/embeddings.py so the RAG corpus and the tool
registry can use different embedding models/dimensions if needed)."""
from __future__ import annotations
from llama_index.embeddings.openai import OpenAIEmbedding
from app.config import settings


def get_embed_model() -> OpenAIEmbedding:
    return OpenAIEmbedding(model=settings.embedding_model, api_key=settings.openai_api_key)
