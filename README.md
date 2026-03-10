# Knowledge Graph Pilot

Consultant-facing document search over a public consulting-paper corpus with:

- Neo4j Aura as the primary retrieval layer
- hybrid vector + full-text graph retrieval
- parent-child chunking for better answer grounding
- OpenAI-based synthesis with citations
- a Streamlit interface suitable for a small internal reviewer pilot

## Deployment stance

This repository is hardened for `Streamlit Community Cloud` style deployment:

- secrets can come from `Streamlit secrets` or environment variables
- the app prefers `Neo4j Aura` for catalog, filters, metrics, and retrieval
- local index loading is fallback-only rather than the default app boot path
- admin ingestion controls are disabled by default

The intended Cloud entrypoint is `app.py`.

## Local setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
PYTHONPATH=src .venv/bin/streamlit run app.py
```

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

## Runtime behavior

- If `Neo4j` is reachable, the app uses graph-backed retrieval.
- If `KG_REQUIRE_GRAPH=true`, the app stops instead of silently degrading when Neo4j is unavailable.
- If `KG_REQUIRE_GRAPH=false`, the app can fall back to a local index when one exists.
- The sidebar shows connection health and timing for embedding, retrieval, and synthesis.

## Admin and ingestion

For pilot reviewers, the deployed app should not rebuild the corpus or reload Neo4j. Those controls are hidden unless:

```bash
KG_ENABLE_ADMIN_TOOLS=true
```

Local maintenance commands remain available:

```bash
./scripts/kg_tool.sh build-index --rebuild
./scripts/kg_tool.sh load-neo4j
./scripts/kg_tool.sh ask "What is Oliver Wyman's take on AI compared with McKinsey?"
```

## Streamlit Community Cloud notes

- Use `requirements.txt` as the dependency source.
- Set the app file to `app.py`.
- Put credentials in Streamlit secrets, not in `.env`.
- Expect cold starts and sleeping on the free tier; this repo is tuned for a small reviewer pilot, not broad production usage.
