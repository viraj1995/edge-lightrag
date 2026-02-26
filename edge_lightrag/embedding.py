"""
Embedding utilities for Edge-LightRAG.

Supports sentence-transformers for local embedding generation.
Can be extended to use Ollama, OpenAI, or other providers.
"""

import numpy as np
from typing import List, Optional
from functools import lru_cache


# Default embedding model - lightweight and fast
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingFunc:
    """Wrapper for embedding functions."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, embedding_dim: int = 384):
        """
        Initialize embedding function.
        
        Args:
            model_name: Name of sentence-transformers model
            embedding_dim: Dimension of embeddings
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model = None
        
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
        
    async def __call__(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            2D numpy array of shape (len(texts), embedding_dim)
        """
        model = self._load_model()
        
        # Handle empty input
        if not texts:
            return np.array([])
            
        # Encode with mean pooling
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        return embeddings


# Global embedding function instance
_embedding_func: Optional[EmbeddingFunc] = None


async def get_embedding(
    texts: List[str],
    model_name: str = DEFAULT_MODEL
) -> np.ndarray:
    """
    Get embeddings for texts using sentence-transformers.
    
    Args:
        texts: List of text strings
        model_name: Model to use
        
    Returns:
        2D numpy array of embeddings
    """
    global _embedding_func
    
    if _embedding_func is None:
        _embedding_func = EmbeddingFunc(model_name)
        
    return await _embedding_func(texts)


async def get_embedding_single(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Get embedding for a single text."""
    embeddings = await get_embedding([text], model_name)
    return embeddings[0] if len(embeddings) > 0 else np.array([])


def clear_embedding_cache():
    """Clear the embedding function cache."""
    global _embedding_func
    _embedding_func = None
