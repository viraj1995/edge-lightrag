"""
Edge-LightRAG: Lightweight RAG for Edge Devices

A stripped-down RAG system optimized for resource-constrained environments
like NVIDIA Jetson AGX Orin.
"""

from .rag import EdgeRAG
from .storage import JSONStorage
from .chunking import chunk_text
from .embedding import get_embedding

__all__ = [
    "EdgeRAG",
    "JSONStorage", 
    "chunk_text",
    "get_embedding",
]
