"""
Simple JSON + NumPy storage layer for Edge-LightRAG.

Replaces complex database backends (Neo4j, MongoDB, PostgreSQL) with
lightweight file-based storage suitable for edge devices.
"""

import json
import os
import aiofiles
import numpy as np
from pathlib import Path
from typing import Any, Optional


class JSONStorage:
    """Simple JSON file storage with numpy array support."""
    
    def __init__(self, working_dir: str, namespace: str = "default"):
        """
        Initialize JSON storage.
        
        Args:
            working_dir: Directory for storage files
            namespace: Subdirectory for this storage instance
        """
        self.working_dir = Path(working_dir) / namespace
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_file = self.working_dir / "data.json"
        self.vectors_file = self.working_dir / "vectors.npy"
        
        self._data: dict = {}
        self._vectors: Optional[np.ndarray] = None
        
    async def initialize(self) -> None:
        """Initialize storage - load existing data if present."""
        # Load JSON data
        if self.data_file.exists():
            async with aiofiles.open(self.data_file, 'r') as f:
                content = await f.read()
                self._data = json.loads(content) if content else {}
        else:
            self._data = {}
            
        # Load vectors
        if self.vectors_file.exists():
            self._vectors = np.load(str(self.vectors_file))
        else:
            self._vectors = np.array([])
            
    async def finalize(self) -> None:
        """Persist data to disk."""
        # Save JSON
        async with aiofiles.open(self.data_file, 'w') as f:
            await f.write(json.dumps(self._data, indent=2))
            
        # Save vectors
        if self._vectors is not None and len(self._vectors) > 0:
            np.save(str(self.vectors_file), self._vectors)
            
    def get(self, key: str) -> Any:
        """Get value by key."""
        return self._data.get(key)
        
    def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        self._data[key] = value
        
    def delete(self, key: str) -> bool:
        """Delete key. Returns True if existed."""
        if key in self._data:
            del self._data[key]
            return True
        return False
        
    def keys(self) -> list:
        """Get all keys."""
        return list(self._data.keys())
        
    def values(self) -> list:
        """Get all values."""
        return list(self._data.values())
        
    def items(self) -> list:
        """Get all key-value pairs."""
        return list(self._data.items())
        
    def __len__(self) -> int:
        return len(self._data)
    
    # Vector operations
    def add_vectors(self, vectors: np.ndarray) -> int:
        """
        Add vectors to storage.
        
        Args:
            vectors: 2D numpy array of shape (n, embedding_dim)
            
        Returns:
            Starting index of added vectors
        """
        if self._vectors is None or len(self._vectors) == 0:
            self._vectors = vectors
            return 0
            
        start_idx = len(self._vectors)
        self._vectors = np.vstack([self._vectors, vectors])
        return start_idx
        
    def get_vectors(self, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        """Get vectors in range [start, end)."""
        if self._vectors is None or len(self._vectors) == 0:
            return np.array([])
        return self._vectors[start:end]
        
    def search_vectors(self, query: np.ndarray, top_k: int = 5) -> tuple:
        """
        Search for most similar vectors using cosine similarity.
        
        Args:
            query: 1D query vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (indices, similarities)
        """
        if self._vectors is None or len(self._vectors) == 0:
            return (), ()
            
        # Normalize vectors
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        vectors_norm = self._vectors / (
            np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-8
        )
        
        # Cosine similarity
        similarities = np.dot(vectors_norm, query_norm)
        
        # Top-k
        if top_k >= len(similarities):
            top_k = len(similarities)
            
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_sims = similarities[top_indices]
        
        return top_indices, top_sims
        
    def clear(self) -> None:
        """Clear all data."""
        self._data = {}
        self._vectors = np.array([])
