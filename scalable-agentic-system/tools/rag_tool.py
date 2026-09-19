"""
RAG Pipeline Tool — exposed to the agent as exactly ONE tool
("knowledge_base.search"), regardless of how many documents sit behind
it. This is deliberate: the router selects a TOOL, not a document, so
adding 10,000 more pages to the knowledge base never grows the tool
catalog or degrades tool-selection accuracy. Retrieval happens *inside*
this tool via LlamaIndex, after the tool has already been chosen.
"""
from __future__ import annotations
from rag.retriever import get_retriever


def search_knowledge_base(query: str, top_k: int = 5) -> dict:
    retriever = get_retriever()
    nodes = retriever.retrieve(query)[:top_k]
    return {
        "query": query,
        "results": [
            {"text": n.node.get_content(), "score": n.score, "source": n.node.metadata.get("source")}
            for n in nodes
        ],
    }


TOOL_HANDLERS = {
    "rag.knowledge_base.search": search_knowledge_base,
}

TOOL_METADATA = {
    "tool_id": "rag.knowledge_base.search",
    "name": "Knowledge Base Search",
    "description": (
        "Search product documentation, guides, FAQs, and policy pages to answer "
        "questions or provide grounded context. Use for 'how do I...' or "
        "'what is...' style questions, not for taking actions."
    ),
    "category": "rag",
    "permissions_required": [],
    "parameters_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
}
