from pathlib import Path

from knowledge_graph_tool.config import Settings
from knowledge_graph_tool.models import (
    AccessPrincipal,
    ChunkRecord,
    CorpusIndex,
    DocumentRecord,
    ParentChunkRecord,
    SearchFilters,
)
from knowledge_graph_tool.search import LocalSearchEngine
from knowledge_graph_tool.search import answer_question


def make_settings() -> Settings:
    root = Path("/tmp/kg")
    return Settings(
        root_dir=root,
        corpus_dir=root,
        data_dir=root,
        local_index_path=root / "index.json",
        restricted_manifest_path=root / "restricted.yaml",
        openai_api_key="",
        openai_chat_model="gpt-4.1-mini",
        openai_embedding_model="text-embedding-3-small",
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="",
        neo4j_database="neo4j",
        top_k=5,
        parent_chunk_size=500,
        child_chunk_size=250,
        child_chunk_overlap=60,
        require_graph=False,
        enable_admin_tools=False,
        openai_timeout_seconds=30,
    )


def test_limited_access_hides_restricted_document():
    public_doc = DocumentRecord(
        doc_id="public-doc",
        path="/tmp/public.md",
        file_name="public.md",
        file_type="md",
        title="Public AI note",
        organization="Firm",
        published_date="2025-01-01",
        access_tier="all",
    )
    restricted_doc = DocumentRecord(
        doc_id="restricted-doc",
        path="/tmp/restricted.md",
        file_name="restricted.md",
        file_type="md",
        title="Restricted banking note",
        organization="Firm",
        published_date="2025-01-02",
        access_tier="restricted",
    )
    index = CorpusIndex(
        generated_at="2026-03-08T00:00:00Z",
        documents=[public_doc, restricted_doc],
        parent_chunks=[
            ParentChunkRecord(
                parent_chunk_id="public-doc:p0",
                doc_id="public-doc",
                parent_index=0,
                heading="Public AI note",
                text="AI transformation and operating model work.",
                token_estimate=6,
            ),
            ParentChunkRecord(
                parent_chunk_id="restricted-doc:p0",
                doc_id="restricted-doc",
                parent_index=0,
                heading="Restricted banking note",
                text="Private banking and compliance program.",
                token_estimate=5,
            ),
        ],
        chunks=[
            ChunkRecord(
                chunk_id="public-doc:p0:c0",
                doc_id="public-doc",
                parent_chunk_id="public-doc:p0",
                chunk_index=0,
                child_index=0,
                heading="Public AI note",
                text="AI transformation and operating model work.",
                token_estimate=6,
            ),
            ChunkRecord(
                chunk_id="restricted-doc:p0:c0",
                doc_id="restricted-doc",
                parent_chunk_id="restricted-doc:p0",
                chunk_index=0,
                child_index=0,
                heading="Restricted banking note",
                text="Private banking and compliance program.",
                token_estimate=5,
            ),
        ],
    )

    engine = LocalSearchEngine(index, make_settings())
    limited_hits = engine.search(
        "banking compliance",
        AccessPrincipal(mode="limited_access", allowed_document_ids=[]),
    )
    full_hits = engine.search(
        "banking compliance",
        AccessPrincipal(mode="limited_access", allowed_document_ids=["restricted-doc"]),
    )

    assert [hit.document.doc_id for hit in limited_hits] == []
    assert [hit.document.doc_id for hit in full_hits] == ["restricted-doc"]


def test_organization_filter_limits_results():
    public_doc = DocumentRecord(
        doc_id="public-doc",
        path="/tmp/public.md",
        file_name="public.md",
        file_type="md",
        title="Public AI note",
        organization="McKinsey",
        published_date="2025-01-01",
        access_tier="all",
    )
    other_doc = DocumentRecord(
        doc_id="other-doc",
        path="/tmp/other.md",
        file_name="other.md",
        file_type="md",
        title="Other AI note",
        organization="Oliver Wyman",
        published_date="2025-01-02",
        access_tier="all",
    )
    index = CorpusIndex(
        generated_at="2026-03-08T00:00:00Z",
        documents=[public_doc, other_doc],
        parent_chunks=[
            ParentChunkRecord(
                parent_chunk_id="public-doc:p0",
                doc_id="public-doc",
                parent_index=0,
                heading="Public AI note",
                text="AI transformation and operating model work.",
                token_estimate=6,
            ),
            ParentChunkRecord(
                parent_chunk_id="other-doc:p0",
                doc_id="other-doc",
                parent_index=0,
                heading="Other AI note",
                text="AI transformation in financial services.",
                token_estimate=6,
            ),
        ],
        chunks=[
            ChunkRecord(
                chunk_id="public-doc:p0:c0",
                doc_id="public-doc",
                parent_chunk_id="public-doc:p0",
                chunk_index=0,
                child_index=0,
                heading="Public AI note",
                text="AI transformation and operating model work.",
                token_estimate=6,
            ),
            ChunkRecord(
                chunk_id="other-doc:p0:c0",
                doc_id="other-doc",
                parent_chunk_id="other-doc:p0",
                chunk_index=0,
                child_index=0,
                heading="Other AI note",
                text="AI transformation in financial services.",
                token_estimate=6,
            ),
        ],
    )

    engine = LocalSearchEngine(index, make_settings())
    hits = engine.search(
        "AI transformation",
        AccessPrincipal(mode="all_access"),
        filters=SearchFilters(organizations=["Oliver Wyman"]),
    )

    assert [hit.document.doc_id for hit in hits] == ["other-doc"]


def test_comparison_query_preserves_organization_coverage():
    documents = [
        DocumentRecord(
            doc_id="mc-1",
            path="/tmp/mc-1.md",
            file_name="mc-1.md",
            file_type="md",
            title="What Is AI?",
            organization="McKinsey",
            published_date="2025-01-01",
            access_tier="all",
        ),
        DocumentRecord(
            doc_id="mc-2",
            path="/tmp/mc-2.md",
            file_name="mc-2.md",
            file_type="md",
            title="What Is AI? PDF variant",
            organization="McKinsey",
            published_date="2025-01-01",
            access_tier="all",
        ),
        DocumentRecord(
            doc_id="ow-1",
            path="/tmp/ow-1.md",
            file_name="ow-1.md",
            file_type="md",
            title="Meet LenAI",
            organization="Oliver Wyman",
            published_date="2025-01-01",
            access_tier="all",
        ),
    ]
    chunks = [
        ChunkRecord(
            chunk_id="mc-1:p0:c0",
            doc_id="mc-1",
            parent_chunk_id="mc-1:p0",
            chunk_index=0,
            child_index=0,
            heading="What Is AI?",
            text="McKinsey AI perspective and generative AI outlook.",
            token_estimate=8,
            embedding=[0.9, 0.1],
        ),
        ChunkRecord(
            chunk_id="mc-2:p0:c0",
            doc_id="mc-2",
            parent_chunk_id="mc-2:p0",
            chunk_index=0,
            child_index=0,
            heading="What Is AI? PDF variant",
            text="McKinsey AI perspective and generative AI outlook.",
            token_estimate=8,
            embedding=[0.88, 0.12],
        ),
        ChunkRecord(
            chunk_id="ow-1:p0:c0",
            doc_id="ow-1",
            parent_chunk_id="ow-1:p0",
            chunk_index=0,
            child_index=0,
            heading="Meet LenAI",
            text="Oliver Wyman describes its AI tool and view on enterprise AI adoption.",
            token_estimate=11,
            embedding=[0.75, 0.25],
        ),
    ]
    parent_chunks = [
        ParentChunkRecord(
            parent_chunk_id="mc-1:p0",
            doc_id="mc-1",
            parent_index=0,
            heading="What Is AI?",
            text="McKinsey AI perspective and generative AI outlook.",
            token_estimate=8,
        ),
        ParentChunkRecord(
            parent_chunk_id="mc-2:p0",
            doc_id="mc-2",
            parent_index=0,
            heading="What Is AI? PDF variant",
            text="McKinsey AI perspective and generative AI outlook.",
            token_estimate=8,
        ),
        ParentChunkRecord(
            parent_chunk_id="ow-1:p0",
            doc_id="ow-1",
            parent_index=0,
            heading="Meet LenAI",
            text="Oliver Wyman describes its AI tool and view on enterprise AI adoption.",
            token_estimate=11,
        ),
    ]
    engine = LocalSearchEngine(
        CorpusIndex(
            generated_at="2026-03-08T00:00:00Z",
            documents=documents,
            parent_chunks=parent_chunks,
            chunks=chunks,
        ),
        make_settings(),
    )

    hits = engine.search(
        "What is Oliver Wyman's take on AI compared with McKinsey?",
        AccessPrincipal(mode="all_access"),
        question_embedding=[1.0, 0.0],
        top_k=4,
    )

    organizations = [hit.document.organization for hit in hits]
    assert "McKinsey" in organizations
    assert "Oliver Wyman" in organizations


def test_explicit_organization_mention_scopes_results_to_that_firm():
    documents = [
        DocumentRecord(
            doc_id="ow-1",
            path="/tmp/ow-1.md",
            file_name="ow-1.md",
            file_type="md",
            title="Oliver Wyman on AI and workforce change",
            organization="Oliver Wyman",
            published_date="2025-01-01",
            access_tier="all",
        ),
        DocumentRecord(
            doc_id="mc-1",
            path="/tmp/mc-1.md",
            file_name="mc-1.md",
            file_type="md",
            title="McKinsey on AI productivity",
            organization="McKinsey",
            published_date="2025-01-01",
            access_tier="all",
        ),
    ]
    parent_chunks = [
        ParentChunkRecord(
            parent_chunk_id="ow-1:p0",
            doc_id="ow-1",
            parent_index=0,
            heading="Oliver Wyman on AI and workforce change",
            text="Oliver Wyman argues AI will reshape work through role redesign and expert augmentation.",
            token_estimate=12,
        ),
        ParentChunkRecord(
            parent_chunk_id="mc-1:p0",
            doc_id="mc-1",
            parent_index=0,
            heading="McKinsey on AI productivity",
            text="McKinsey emphasizes productivity gains from generative AI across occupations.",
            token_estimate=10,
        ),
    ]
    chunks = [
        ChunkRecord(
            chunk_id="ow-1:p0:c0",
            doc_id="ow-1",
            parent_chunk_id="ow-1:p0",
            chunk_index=0,
            child_index=0,
            heading="Oliver Wyman on AI and workforce change",
            text="Oliver Wyman argues AI will reshape work through role redesign and expert augmentation.",
            token_estimate=12,
            embedding=[0.9, 0.1],
        ),
        ChunkRecord(
            chunk_id="mc-1:p0:c0",
            doc_id="mc-1",
            parent_chunk_id="mc-1:p0",
            chunk_index=0,
            child_index=0,
            heading="McKinsey on AI productivity",
            text="McKinsey emphasizes productivity gains from generative AI across occupations.",
            token_estimate=10,
            embedding=[0.91, 0.09],
        ),
    ]
    index = CorpusIndex(
        generated_at="2026-03-08T00:00:00Z",
        documents=documents,
        parent_chunks=parent_chunks,
        chunks=chunks,
    )

    bundle = answer_question(
        question="What is Oliver Wyman's take on how AI will impact jobs?",
        principal=AccessPrincipal(mode="all_access"),
        index=index,
        settings=make_settings(),
        use_llm=False,
    )

    assert bundle.hits
    assert {hit.document.organization for hit in bundle.hits} == {"Oliver Wyman"}
