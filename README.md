

# OpenQueryAgent

**Open-source, database-agnostic query agent for vector databases.**

Translate natural language into precise vector database operations across multiple backends — with a single unified API.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Type Checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue)](http://mypy-lang.org/)
[![Code Style: ruff](https://img.shields.io/badge/code--style-ruff-purple)](https://docs.astral.sh/ruff/)
[![Tests: 407 passing](https://img.shields.io/badge/tests-407%20passing-brightgreen)]()

---

## Features

* **8 Vector Database Adapters** — Qdrant, Milvus, pgvector, Weaviate, Pinecone, Chroma, Elasticsearch, AWS S3 Vectors
* **LLM-Powered Query Planning** — Decomposes complex queries into optimized sub-queries with automatic fallback
* **Universal Filter DSL** — Write filters once, compile to any backend's native format
* **Streaming Responses** — Token-by-token answer generation with citation extraction
* **Multi-DB Federation** — Query across multiple databases in a single request
* **Pluggable Pipeline** — Swap any component: planner, reranker, synthesizer, LLM, embedding model
* **Conversation Memory** — Multi-turn dialogue with automatic token budget management
* **Security Hardened** — Input validation, SQL injection prevention, credential redaction, prompt injection protection
* **Fully Typed** — Strict mypy with `py.typed` marker for downstream consumers

---

## Quick Start

### Installation

```bash
# Core + Qdrant adapter + OpenAI LLM & embeddings
pip install openqueryagent[qdrant,openai]

# Or with Milvus
pip install openqueryagent[milvus,openai]

# Or with pgvector
pip install openqueryagent[pgvector,openai]

# Or with Elasticsearch
pip install openqueryagent[elasticsearch,openai]

# Everything
pip install openqueryagent[all]
```

### Basic Usage

```python
import asyncio
from openqueryagent.core.agent import QueryAgent
from openqueryagent.adapters.qdrant import QdrantAdapter
from openqueryagent.llm.openai import OpenAIProvider
from openqueryagent.embeddings.openai import OpenAIEmbedding

async def main():
    # Create components
    adapter = QdrantAdapter(url="localhost", port=6333)
    await adapter.connect()

    llm = OpenAIProvider(model="gpt-4o-mini")
    embedding = OpenAIEmbedding(model="text-embedding-3-small")

    # Create agent
    agent = QueryAgent(
        adapters={"qdrant": adapter},
        llm=llm,
        embedding=embedding,
    )
    await agent.initialize()

    # Ask a question — gets synthesized answer with citations
    response = await agent.ask("What are the best products under $50?")
    print(response.answer)
    for citation in response.citations:
        print(f"  [{citation.document_id}]: {citation.text_snippet}")

    # Search — returns ranked documents without synthesis
    results = await agent.search("machine learning papers", limit=5)
    for doc in results.documents:
        print(f"  {doc.document.id}: {doc.score:.3f}")

asyncio.run(main())
```

### With MiniMax

```python
from openqueryagent.core.agent import QueryAgent
from openqueryagent.adapters.qdrant import QdrantAdapter
from openqueryagent.llm.minimax import MiniMaxProvider

async def main():
    adapter = QdrantAdapter(url="localhost", port=6333)
    await adapter.connect()

    # MiniMax-M2.5 with 204K context window — no extra install needed
    llm = MiniMaxProvider(model="MiniMax-M2.5")  # uses MINIMAX_API_KEY env var

    agent = QueryAgent(adapters={"qdrant": adapter}, llm=llm)
    await agent.initialize()

    response = await agent.ask("What are the latest trends in AI?")
    print(response.answer)
```

### Without LLM (Zero-Cost Search)

```python
from openqueryagent.core.agent import QueryAgent
from openqueryagent.adapters.qdrant import QdrantAdapter
from openqueryagent.core.planner import SimpleQueryPlanner

agent = QueryAgent(
    adapters={"qdrant": adapter},
    planner=SimpleQueryPlanner(default_collection="products"),
)

# Uses SimpleQueryPlanner — no LLM calls, just vector search
results = await agent.search("wireless headphones")
```

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                  QueryAgent                           │
│                                                       │
│  ┌──────────┐   ┌────────┐   ┌──────────┐            │
│  │  Schema   │──▶│ Query  │──▶│  Router  │            │
│  │ Inspector │   │Planner │   │          │            │
│  └──────────┘   └────────┘   └────┬─────┘            │
│                                    │                  │
│                               ┌────▼─────┐            │
│                               │ Executor │ (parallel)  │
│                               └────┬─────┘            │
│                                    │                  │
│                               ┌────▼─────┐            │
│                               │ Reranker │ (RRF)      │
│                               └────┬─────┘            │
│                                    │                  │
│                               ┌────▼──────┐           │
│                               │Synthesizer│ (LLM)     │
│                               └───────────┘           │
└──────────────────────────────────────────────────────┘
    │
    ▼
AskResponse { answer, citations, query_plan }
```

---

## Pipeline

| Stage      | Component                                                     | Purpose                                                  |
| ---------- | ------------------------------------------------------------- | -------------------------------------------------------- |
| Plan       | `LLMQueryPlanner` / `SimpleQueryPlanner` / `RuleBasedPlanner` | Decompose query into sub-queries with intent detection   |
| Route      | `QueryRouter`                                                 | Resolve collections, compile filters to native format    |
| Execute    | `QueryExecutor`                                               | Parallel execution with timeouts and dependency ordering |
| Rerank     | `RRFReranker` / `NoopReranker`                                | Reciprocal Rank Fusion for multi-source results          |
| Synthesize | `LLMSynthesizer`                                              | Generate answer with `[N]` citation extraction           |

---

## Supported Backends

### Vector Databases

| Backend        | Extra                           | Search Types                         | Aggregation          |
| -------------- | ------------------------------- | ------------------------------------ | -------------------- |
| Qdrant         | `openqueryagent[qdrant]`        | Vector, Keyword, Hybrid (prefetch)   | Client-side scroll   |
| Milvus         | `openqueryagent[milvus]`        | Vector, Keyword, Hybrid              | Client-side query    |
| pgvector       | `openqueryagent[pgvector]`      | Vector, Keyword, Hybrid              | Native SQL           |
| Weaviate       | `openqueryagent[weaviate]`      | Vector, Keyword (BM25), Hybrid       | Client-side          |
| Pinecone       | `openqueryagent[pinecone]`      | Vector, Keyword, Hybrid              | Client-side          |
| Chroma         | `openqueryagent[chroma]`        | Vector, Keyword                      | Client-side          |
| Elasticsearch  | `openqueryagent[elasticsearch]` | Vector (kNN), Keyword (BM25), Hybrid | Native agg framework |
| AWS S3 Vectors | `openqueryagent[s3vectors]`     | Vector                               | Client-side          |

### LLM Providers

| Provider    | Extra                       | Features                                            |
| ----------- | --------------------------- | --------------------------------------------------- |
| OpenAI      | `openqueryagent[openai]`    | GPT-4o, JSON mode, streaming, Azure support         |
| Anthropic   | `openqueryagent[anthropic]` | Claude, JSON extraction, streaming                  |
| MiniMax     | Built-in                    | MiniMax-M2.5 (204K context), JSON mode, streaming   |
| Ollama      | Built-in                    | Local models (Llama 3, Mistral, Mixtral), streaming |
| AWS Bedrock | `openqueryagent[bedrock]`   | Claude, Titan, Llama via Bedrock                    |

### Embedding Providers

| Provider    | Models                                                                       |
| ----------- | ---------------------------------------------------------------------------- |
| OpenAI      | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| Cohere      | `embed-english-v3.0`, `embed-multilingual-v3.0`                              |
| HuggingFace | Any `sentence-transformers` model                                            |
| AWS Bedrock | Titan Embed, Cohere Embed via Bedrock                                        |

---

## Universal Filter DSL

Write filters once, compile to any backend:

```python
from openqueryagent.core.filters import F

# Build a filter
f = (F.price < 50) & (F.category == "electronics") & ~(F.status == "discontinued")

results = await agent.search("wireless headphones", filters=f)
```

### Supported Operators

| Category   | Operators                                                       |                 |
| ---------- | --------------------------------------------------------------- | --------------- |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=`                                |                 |
| Collection | `in_`, `not_in`, `between`                                      |                 |
| Text       | `contains`, `not_contains`, `starts_with`, `ends_with`, `regex` |                 |
| Geo        | `geo_radius`                                                    |                 |
| Existence  | `exists`                                                        |                 |
| Boolean    | `&` (AND), `                                                    | `(OR),`~` (NOT) |

---

## Conversation Memory

Multi-turn dialogue with automatic context management:

```python
response1 = await agent.ask("What products do you have in electronics?")
response2 = await agent.ask("Which of those are under $30?")

agent.memory.get_messages()
agent.memory.clear()
```

---

## Development

### Setup

```bash
git clone https://github.com/thirukguru/openqueryagent.git
cd openqueryagent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,qdrant,milvus,pgvector,openai,anthropic]"
```

### Quality Checks

```bash
ruff check openqueryagent/ tests/
mypy openqueryagent/
pytest tests/ -v
```

---

## License

Apache 2.0 — see LICENSE for details.

