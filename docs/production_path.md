# Production Path

## What is implemented now

- Local-first ingestion for `md`, `html`, and `pdf` documents in `papers/`
- Chunking, metadata extraction, and lightweight taxonomy tagging
- Simulated access tiers with `all_access` and `limited_access`
- Citation-aware retrieval with a local search fallback
- Optional OpenAI-backed embeddings and answer synthesis
- Neo4j loading utilities and a local Docker Compose definition
- Streamlit demo for search, citations, and access simulation

## What is not implemented yet

- PPTX ingestion and slide-level extraction
- Real Okta authentication and authorization
- Human review queue for low-confidence or restricted documents
- Production-grade observability, retries, and job orchestration
- Real entity resolution against enterprise systems
- Deployment automation and managed secrets
- Final frontend in React

## What I still need from you

1. `OPENAI_API_KEY` that we should use going forward, if you want me to stop relying on any ambient environment key.
2. Neo4j target credentials, or confirmation that you want me to keep using a local Docker instance.
3. The first real PPTX corpus when you want the original PoC spec implemented.
4. The final rule for which documents should be treated as restricted in the pilot.
5. The preferred frontend handoff point, after which I should write the Lovable prompt for a React UI.

## Recommended next implementation order

1. Add a `pptx` ingestion adapter with slide extraction, speaker notes, and embedded asset detection.
2. Load chunk embeddings and metadata into Neo4j and switch retrieval to graph-backed search when configured.
3. Add a review state model for restricted and low-confidence documents.
4. Replace the demo access selector with a real auth adapter and role mapping.
5. Split the pipeline into ingest, index, and query services with logging and retries.
6. Hand off the backend contract to a React frontend.

## Current validation status

- Local index build succeeded on `62` documents and `543` chunks.
- Local query path succeeded with citation-grounded answers.
- Automated tests pass.
- Neo4j load path is coded but not validated yet because the local Docker daemon was not running during verification.
