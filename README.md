# Edge-LightRAG: Lightweight RAG for Edge Devices

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-4ecdc4?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/github/stars/viraj1995/edge-lightrag?style=for-the-badge" alt="Stars">
  <img src="https://img.shields.io/badge/Jetson-AGX%20Orin-00d9ff?style=for-the-badge&logo=nvidia">
</p>

A stripped-down, memory-efficient Retrieval-Augmented Generation (RAG) system optimized for edge devices like NVIDIA Jetson AGX Orin. Built on LightRAG concepts but with minimal dependencies and resource usage.

## Why Edge-LightRAG?

| Feature | LightRAG | Edge-LightRAG |
|---------|----------|---------------|
| Dependencies | 20+ (Neo4j, MongoDB, etc.) | 5 core only |
| Memory Usage | 500MB+ | ~100MB |
| Storage Options | Multiple DBs | JSON + NumPy only |
| API Server | FastAPI included | Optional |
| Graph Operations | Full KG traversal | Simplified |
| Best For | Cloud/Server | Jetson, Raspberry Pi, Mobile |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Edge-LightRAG                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │   Document   │───▶│   Indexing   │───▶│   Storage   │ │
│  │   Input      │    │   Pipeline   │    │   Layer     │ │
│  └──────────────┘    └──────────────┘    └──────┬──────┘ │
│                                                  │         │
│                     ┌───────────────────────────┘         │
│                     ▼                                     │
│            ┌────────────────┐                              │
│            │   JSON Files   │  (lightweight persistence)  │
│            │  - chunks.json │                              │
│            │  - entities.json                             │
│            │  - vectors.npy  │                              │
│            └────────────────┘                              │
│                          │                                 │
│                     ┌────▼─────┐                           │
│                     │  Query   │                           │
│                     │  Layer   │                           │
│                     └────┬─────┘                           │
│                          │                                 │
│         ┌────────────────┼────────────────┐               │
│         ▼                ▼                ▼               │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐        │
│  │    Naive   │   │   Local   │   │  Hybrid    │        │
│  │  Retrieval │   │  Retrieval│   │  (future)  │        │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘        │
│        │                │                 │               │
│        └────────────────┴─────────────────┘               │
│                         │                                  │
│                    ┌────▼────┐                             │
│                    │ Response│                             │
│                    └─────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/viraj1995/edge-lightrag.git
cd edge-lightrag
pip install -r requirements.txt
```

## Requirements

```
numpy>=1.24.0
sentence-transformers>=2.2.0
aiofiles>=23.0.0
```

## Quick Start

```python
import asyncio
from edge_lightrag import EdgeRAG

async def main():
    rag = EdgeRAG(
        working_dir="./rag_storage",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Index documents
    await rag.insert("Your document text here...")
    
    # Query
    result = await rag.query("Your question?")
    print(result)

asyncio.run(main())
```

## Project Status

- [x] Repo created
- [ ] Core RAG class implemented
- [ ] Chunking & embedding pipeline
- [ ] Naive retrieval mode
- [ ] Local retrieval mode (entity-based)
- [ ] JSON storage layer
- [ ] Documentation & README
- [ ] Jetson deployment test

## Roadmap (Week 1)

| Day | Task |
|-----|------|
| 1 | Analyze LightRAG architecture, plan strip-down |
| 2 | Core EdgeRAG class + storage layer |
| 3 | Indexing pipeline (chunking + embedding) |
| 4 | Naive retrieval (vector search) |
| 5 | Local retrieval (entity-based) |
| 6 | Testing + documentation |
| 7 | Jetson deployment verification |

## References

- [LightRAG](https://github.com/HKUDS/LightRAG) - Original project (28k stars)
- [Paper](https://arxiv.org/abs/2410.05779) - LightRAG paper

## License

MIT

---

*Built by [Viraj](https://github.com/viraj1995) as part of AI engineering learning journey.*
