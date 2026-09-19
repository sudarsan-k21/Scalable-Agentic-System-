"""Lazily-initialized retriever singleton used by tools/rag_tool.py."""
from __future__ import annotations
from functools import lru_cache
from llama_index.core import VectorStoreIndex
from rag.loader import get_vector_store
from rag.embeddings import get_embed_model


@lru_cache(maxsize=1)
def get_retriever(top_k: int = 5):
    index = VectorStoreIndex.from_vector_store(
        vector_store=get_vector_store(), embed_model=get_embed_model()
    )
    return index.as_retriever(similarity_top_k=top_k)
