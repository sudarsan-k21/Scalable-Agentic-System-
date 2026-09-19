"""Ingests documents (product docs, guides, PDFs) into the RAG vector store."""
from __future__ import annotations
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from app.config import settings


def get_vector_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        database=settings.postgres_url.rsplit("/", 1)[-1],
        host="localhost",
        password="postgres",
        port=5432,
        user="postgres",
        table_name="rag_documents",
        embed_dim=1536,
    )


def ingest_directory(path: str) -> VectorStoreIndex:
    documents = SimpleDirectoryReader(path).load_data()
    storage_context = StorageContext.from_defaults(vector_store=get_vector_store())
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index
