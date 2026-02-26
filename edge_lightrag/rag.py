"""
EdgeRAG: Core RAG class for edge devices.

A simplified but functional RAG implementation optimized for
resource-constrained environments.
"""

import asyncio
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from .storage import JSONStorage
from .chunking import chunk_text, extract_entities_simple
from .embedding import get_embedding, DEFAULT_MODEL


class QueryParam:
    """Parameters for querying."""
    
    def __init__(
        self,
        mode: str = "naive",
        top_k: int = 5,
        include_entities: bool = True
    ):
        self.mode = mode  # "naive" or "local"
        self.top_k = top_k
        self.include_entities = include_entities


class EdgeRAG:
    """
    Lightweight RAG system for edge devices.
    
    Supports:
    - Naive retrieval (pure vector similarity)
    - Local retrieval (entity-based context)
    """
    
    def __init__(
        self,
        working_dir: str = "./rag_storage",
        embedding_model: str = DEFAULT_MODEL,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize EdgeRAG.
        
        Args:
            working_dir: Directory for storing index data
            embedding_model: Sentence-transformers model name
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap
        """
        self.working_dir = working_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Storage instances
        self.chunk_storage: Optional[JSONStorage] = None
        self.entity_storage: Optional[JSONStorage] = None
        
        # In-memory cache
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize storage backends."""
        if self._initialized:
            return
            
        self.chunk_storage = JSONStorage(self.working_dir, "chunks")
        self.entity_storage = JSONStorage(self.working_dir, "entities")
        
        await asyncio.gather(
            self.chunk_storage.initialize(),
            self.entity_storage.initialize()
        )
        
        self._initialized = True
        
    async def finalize(self) -> None:
        """Finalize and persist storage."""
        if not self._initialized:
            return
            
        await asyncio.gather(
            self.chunk_storage.finalize(),
            self.entity_storage.finalize()
        )
        
    async def insert(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Insert a document into the RAG system.
        
        Args:
            text: Document text
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            Document ID
        """
        await self._ensure_initialized()
        
        if not doc_id:
            doc_id = str(uuid.uuid4())
            
        # Chunk the text
        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        if not chunks:
            return doc_id
            
        # Generate embeddings
        embeddings = await get_embedding(chunks, self.embedding_model)
        
        # Store chunks with embeddings
        chunk_start_idx = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Extract entities from this chunk
            entities = extract_entities_simple(chunk)
            
            chunk_data = {
                "id": chunk_id,
                "doc_id": doc_id,
                "text": chunk,
                "chunk_index": i,
                "entities": entities,
                "embedding_idx": chunk_start_idx + i
            }
            
            self.chunk_storage.set(chunk_id, chunk_data)
            
        # Add embeddings
        if len(embeddings) > 0:
            self.chunk_storage.add_vectors(embeddings)
            
        # Extract and store entities
        all_entities = extract_entities_simple(text)
        for entity in all_entities:
            existing = self.entity_storage.get(entity)
            if existing:
                if doc_id not in existing["doc_ids"]:
                    existing["doc_ids"].append(doc_id)
                    existing["count"] = len(existing["doc_ids"])
                self.entity_storage.set(entity, existing)
            else:
                self.entity_storage.set(entity, {
                    "name": entity,
                    "doc_ids": [doc_id],
                    "count": 1
                })
                
        return doc_id
        
    async def insert_batch(self, texts: List[str]) -> List[str]:
        """
        Insert multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        for text in texts:
            doc_id = await self.insert(text)
            doc_ids.append(doc_id)
        return doc_ids
        
    async def query(
        self,
        query_text: str,
        param: Optional[QueryParam] = None
    ) -> str:
        """
        Query the RAG system.
        
        Args:
            query_text: Query string
            param: Query parameters
            
        Returns:
            Retrieved context string
        """
        await self._ensure_initialized()
        
        if param is None:
            param = QueryParam()
            
        if param.mode == "local":
            return await self._local_retrieval(query_text, param)
        else:
            return await self._naive_retrieval(query_text, param)
            
    async def _naive_retrieval(
        self,
        query_text: str,
        param: QueryParam
    ) -> str:
        """Pure vector similarity retrieval."""
        # Get query embedding
        query_embedding = await get_embedding(
            [query_text],
            self.embedding_model
        )
        
        if len(query_embedding) == 0:
            return ""
            
        query_vec = query_embedding[0]
        
        # Search vectors
        indices, similarities = self.chunk_storage.search_vectors(
            query_vec,
            top_k=param.top_k
        )
        
        # Get corresponding chunks
        results = []
        for idx, sim in zip(indices, similarities):
            # Find chunk with this embedding index
            for key in self.chunk_storage.keys():
                chunk = self.chunk_storage.get(key)
                if chunk and chunk.get("embedding_idx") == int(idx):
                    results.append({
                        "text": chunk["text"],
                        "score": float(sim)
                    })
                    break
                    
        # Format results
        context = "\n\n".join([
            f"[Score: {r['score']:.3f}] {r['text']}"
            for r in results
        ])
        
        return context
        
    async def _local_retrieval(
        self,
        query_text: str,
        param: QueryParam
    ) -> str:
        """Entity-based local retrieval."""
        # Extract entities from query
        query_entities = extract_entities_simple(query_text)
        
        # Find relevant documents via entities
        relevant_doc_ids = set()
        for entity in query_entities:
            entity_data = self.entity_storage.get(entity)
            if entity_data:
                relevant_doc_ids.update(entity_data.get("doc_ids", []))
                
        if not relevant_doc_ids:
            # Fall back to naive if no entities found
            return await self._naive_retrieval(query_text, param)
            
        # Get chunks from relevant docs
        relevant_chunks = []
        for key in self.chunk_storage.keys():
            chunk = self.chunk_storage.get(key)
            if chunk and chunk.get("doc_id") in relevant_doc_ids:
                relevant_chunks.append(chunk)
                
        if not relevant_chunks:
            return ""
            
        # Re-rank by similarity
        query_embedding = await get_embedding(
            [query_text],
            self.embedding_model
        )
        
        if len(query_embedding) == 0:
            return ""
            
        query_vec = query_embedding[0]
        
        # Get embeddings for relevant chunks
        chunk_texts = [c["text"] for c in relevant_chunks]
        chunk_embeddings = await get_embedding(chunk_texts, self.embedding_model)
        
        # Calculate similarities
        similarities = np.dot(
            chunk_embeddings,
            query_vec
        )
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[-param.top_k * 2:][::-1]
        
        # Format results
        results = []
        for idx in sorted_indices[:param.top_k]:
            results.append({
                "text": relevant_chunks[int(idx)]["text"],
                "score": float(similarities[int(idx)])
            })
            
        context = "\n\n".join([
            f"[Score: {r['score']:.3f}] {r['text']}"
            for r in results
        ])
        
        return context
        
    async def _ensure_initialized(self) -> None:
        """Ensure storage is initialized."""
        if not self._initialized:
            await self.initialize()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics."""
        if not self._initialized:
            return {"initialized": False}
            
        return {
            "initialized": True,
            "total_chunks": len(self.chunk_storage) if self.chunk_storage else 0,
            "total_entities": len(self.entity_storage) if self.entity_storage else 0,
            "working_dir": self.working_dir,
            "embedding_model": self.embedding_model
        }
