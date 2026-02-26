"""
Example usage of Edge-LightRAG.
"""

import asyncio
from edge_lightrag import EdgeRAG


async def main():
    # Initialize RAG
    rag = EdgeRAG(
        working_dir="./rag_storage",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Initialize storage
    await rag.initialize()
    
    # Insert sample documents
    print("Inserting documents...")
    docs = [
        "Python is a high-level programming language known for its simplicity and readability. "
        "It supports multiple programming paradigms including procedural, object-oriented, and functional programming. "
        "Python is widely used in data science, machine learning, and web development.",
        
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data. "
        "Deep learning is a specialized form of machine learning using neural networks with many layers. "
        "Popular frameworks include PyTorch, TensorFlow, and Keras.",
        
        "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge bases. "
        "It helps reduce hallucinations and provides up-to-date information. "
        "RAG systems typically use vector databases for efficient similarity search.",
        
        "NVIDIA Jetson is a series of embedded computing boards designed for AI applications. "
        "Jetson AGX Orin is the flagship model offering high performance for edge AI. "
        "It supports CUDA and various AI frameworks out of the box.",
    ]
    
    for doc in docs:
        await rag.insert(doc)
        
    # Query
    print("\nQuerying...")
    
    # Naive mode
    result = await rag.query(
        "What is Python used for?",
        mode="naive",
        top_k=2
    )
    print(f"\n[Naive Retrieval]\n{result}")
    
    # Local mode (entity-based)
    result = await rag.query(
        "How does RAG reduce hallucinations?",
        mode="local",
        top_k=2
    )
    print(f"\n[Local Retrieval]\n{result}")
    
    # Get stats
    stats = rag.get_stats()
    print(f"\n[Stats] {stats}")
    
    # Cleanup
    await rag.finalize()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
