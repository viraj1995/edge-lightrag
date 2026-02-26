"""
Text chunking utilities for Edge-LightRAG.

Simple but effective chunking strategies for RAG pipelines.
"""

import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    split_by: str = "word"
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Maximum chunk size (in words or sentences)
        chunk_overlap: Overlap between chunks
        split_by: Unit to split by ("word", "sentence", "paragraph")
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
        
    if split_by == "sentence":
        return _chunk_by_sentence(text, chunk_size, chunk_overlap)
    elif split_by == "paragraph":
        return _chunk_by_paragraph(text, chunk_size, chunk_overlap)
    else:
        return _chunk_by_word(text, chunk_size, chunk_overlap)


def _chunk_by_word(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split by words with overlap."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
            
    return chunks


def _chunk_by_sentence(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split by sentences with overlap."""
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= chunk_size:
        return [text] if text.strip() else []
        
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(sentences):
            break
            
    return chunks


def _chunk_by_paragraph(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split by paragraphs with overlap."""
    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if len(paragraphs) <= chunk_size:
        return paragraphs if paragraphs else [text]
        
    # Merge multiple paragraphs into chunks
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) > 0:
            test_chunk = current_chunk + "\n\n" + para
        else:
            test_chunk = para
            
        # Rough word count estimate
        word_count = len(test_chunk.split())
        
        if word_count > chunk_size and len(current_chunk) > 0:
            chunks.append(current_chunk)
            # Keep last part for overlap
            current_chunk = para[-overlap * 5:] if len(para) > overlap * 5 else para
        else:
            current_chunk = test_chunk
            
    if current_chunk.strip():
        chunks.append(current_chunk)
        
    return chunks


def extract_entities_simple(text: str) -> List[str]:
    """
    Simple named entity extraction using capitalization.
    
    This is a lightweight alternative to full NER.
    For production, consider spacy or transformers-based NER.
    """
    # Capitalized words (likely proper nouns)
    # Exclude common sentence starters
    patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
    ]
    
    entities = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        # Filter out common non-entities
        for match in matches:
            # Skip if it's just the first word of a sentence
            if match.lower() not in {'the', 'a', 'an', 'this', 'that', 'in', 'on', 'at', 'to', 'for'}:
                entities.add(match)
                
    return list(entities)[:20]  # Limit to top 20
