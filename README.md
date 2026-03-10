# Consulting Research Explorer

Consulting Research Explorer is a citation-first question answering app for public consulting-firm research.

It is built for a small internal reviewer group that wants to:

- ask questions across a multi-firm document set
- compare how firms frame the same topic
- inspect the exact passages used to support each answer
- keep Neo4j as the retrieval layer instead of hiding everything inside a flat vector store

The current corpus is the local `papers/` directory used during ingestion, while the deployed app runs graph-first against Neo4j Aura.

## What the app does

- Ingests Markdown and PDF research documents
- Extracts and cleans document metadata, including PDF metadata with OpenAI-assisted cleanup
- Builds a parent-child chunk structure for better grounded retrieval
- Loads documents, sections, chunks, and relationships into Neo4j Aura
- Uses hybrid graph retrieval:
  - vector search over child chunks
  - full-text search over chunk content
  - section-level context recovery before synthesis
- Generates a concise answer with citations to the supporting sources

## Product behavior

The Streamlit app is designed to answer questions like:

- What themes are showing up in AI transformation for financial services?
- How does Oliver Wyman frame AI compared with McKinsey?
- Which firms emphasize operating model change versus technology modernization?

The UI exposes:

- answer generation with citations
- source expanders with supporting excerpts
- organization, industry, topic, and year filters
- optional restricted-access simulation for future access-control work

## Architecture

Core runtime flow:

1. User asks a question in Streamlit.
2. The app embeds the query with OpenAI.
3. Neo4j Aura retrieves relevant child chunks using hybrid search.
4. The retriever rolls those results up to parent sections for better context.
5. The app synthesizes a cited answer using the retrieved evidence only.

Main components:

- `app.py`: Streamlit Cloud entrypoint
- `src/knowledge_graph_tool/demo.py`: Streamlit UI
- `src/knowledge_graph_tool/ingest.py`: ingestion and index build
- `src/knowledge_graph_tool/graph.py`: Neo4j load, health, metrics, retrieval support
- `src/knowledge_graph_tool/search.py`: graph-backed retrieval and answer assembly
- `src/knowledge_graph_tool/llm.py`: embeddings, answer synthesis, PDF metadata cleanup

## Deployment model

This repository is hardened for Streamlit Community Cloud for a small reviewer pilot.

Design choices:

- secrets can be provided through Streamlit secrets or environment variables
- the deployed app is graph-first and prefers Neo4j Aura
- local index loading is fallback-only
- admin rebuild tools are disabled by default
- timing and connection health are surfaced in the UI for debugging reviewer issues

This is suitable for a small internal evaluation group. It is not the final production hosting model.

## Required secrets

Use Streamlit secrets or environment variables for:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_CHAT_MODEL = "gpt-5-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_TIMEOUT_SECONDS = "60"

NEO4J_URI = "neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

KG_REQUIRE_GRAPH = "true"
KG_ENABLE_ADMIN_TOOLS = "false"
KG_TOP_K = "6"
```

See `.streamlit/secrets.toml.example`.

## Local development

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
PYTHONPATH=src .venv/bin/streamlit run app.py
```

## Loading or rebuilding data

Local maintenance commands:

```bash
./scripts/kg_tool.sh build-index --rebuild
./scripts/kg_tool.sh load-neo4j
./scripts/kg_tool.sh ask "What is Oliver Wyman's take on AI compared with McKinsey?"
```

For reviewer deployments, keep `KG_ENABLE_ADMIN_TOOLS=false` so those controls stay hidden in the UI.

## Streamlit Community Cloud setup

1. Create a new app from this repository.
2. Select branch `main`.
3. Set the app file to `app.py`.
4. Paste the required secrets into Streamlit Cloud.
5. Deploy.

Recommended reviewer settings:

- `KG_REQUIRE_GRAPH = "true"`
- `KG_ENABLE_ADMIN_TOOLS = "false"`

## Current limitations

- The pilot corpus is still public-paper based rather than real client decks.
- PDF extraction quality is improved, but some titles and publication dates can still be noisy.
- Streamlit Community Cloud is acceptable for a reviewer pilot, not for broad consultant rollout.
- Access control is still simulated; real identity and authorization are a future integration.

## Status

The repository currently supports:

- Neo4j Aura-backed retrieval
- OpenAI synthesis with citations
- Streamlit deployment for internal reviewers
- parent-child chunking aligned with modern GraphRAG retrieval patterns
