from __future__ import annotations

from typing import Optional

import streamlit as st

from knowledge_graph_tool.config import load_settings
from knowledge_graph_tool.graph import (
    fetch_graph_catalog,
    fetch_graph_metrics,
    graph_available,
    graph_health,
    load_index_to_neo4j,
)
from knowledge_graph_tool.ingest import build_or_load_index, load_restricted_manifest
from knowledge_graph_tool.models import AccessPrincipal, CorpusIndex, DocumentRecord, SearchFilters
from knowledge_graph_tool.search import answer_question


st.set_page_config(page_title="Consulting Research Explorer", layout="wide")


@st.cache_resource
def app_settings():
    return load_settings()


@st.cache_data(show_spinner=False)
def cached_index(rebuild: bool = False) -> Optional[CorpusIndex]:
    settings = app_settings()
    try:
        return build_or_load_index(settings, rebuild=rebuild)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def cached_graph_catalog():
    return fetch_graph_catalog(app_settings())


@st.cache_data(show_spinner=False)
def cached_graph_metrics():
    return fetch_graph_metrics(app_settings())


@st.cache_data(show_spinner=False, ttl=60)
def cached_graph_health():
    return graph_health(app_settings())


@st.cache_data(show_spinner=False)
def cached_search(
    question: str,
    access_mode: str,
    allowed_document_ids: tuple[str, ...],
    organizations: tuple[str, ...],
    industries: tuple[str, ...],
    topics: tuple[str, ...],
    years: tuple[str, ...],
    use_llm: bool,
    catalog_signature: tuple[str, ...],
    index_generated_at: str,
):
    settings = app_settings()
    index = None if graph_available(settings) else cached_index(False)
    principal = AccessPrincipal(mode=access_mode, allowed_document_ids=list(allowed_document_ids))
    filters = SearchFilters(
        organizations=list(organizations),
        practices=[],
        industries=list(industries),
        topics=list(topics),
        years=list(years),
    )
    documents = list(cached_graph_catalog()) if graph_available(settings) else (index.documents if index else [])
    return answer_question(
        question=question,
        principal=principal,
        index=index,
        settings=settings,
        filters=filters,
        use_llm=use_llm,
        documents=documents,
    )


def active_documents(settings) -> list[DocumentRecord]:
    if graph_available(settings):
        catalog = cached_graph_catalog()
        if catalog:
            return catalog
    index = cached_index(False)
    return index.documents if index else []


def active_metrics(settings, documents: list[DocumentRecord]) -> dict:
    if graph_available(settings):
        try:
            metrics = cached_graph_metrics()
            if metrics["documents"]:
                return metrics
        except Exception:
            pass
    index = cached_index(False)
    return {
        "documents": len(documents),
        "sections": len(index.parent_chunks) if index else 0,
        "organizations": len({document.organization for document in documents if document.organization != "Unknown"}),
        "restricted_documents": len([document for document in documents if document.access_tier == "restricted"]),
    }


def main() -> None:
    settings = app_settings()
    manifest = load_restricted_manifest(settings.restricted_manifest_path)
    documents = active_documents(settings)
    metrics = active_metrics(settings, documents)
    documents_by_id = {document.doc_id: document for document in documents}
    restricted_documents = [
        document for document in documents if document.file_name in manifest["restricted_documents"]
    ]
    organizations = sorted({document.organization for document in documents if document.organization != "Unknown"})
    industries = sorted({industry for document in documents for industry in document.industries})
    topics = sorted({topic for document in documents for topic in document.topics})
    years = sorted({(document.published_date or "")[:4] for document in documents if document.published_date})
    health = cached_graph_health()
    catalog_signature = tuple(document.doc_id for document in documents)
    fallback_index = cached_index(False) if not graph_available(settings) else None
    index_generated_at = fallback_index.generated_at if fallback_index else ""

    st.title("Consulting Research Explorer")
    st.caption(
        "Ask across consulting-firm research, compare viewpoints, and inspect the cited passages behind each answer."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Documents", metrics["documents"])
    metric_columns[1].metric("Sections", metrics["sections"])
    metric_columns[2].metric("Organizations", metrics["organizations"])
    metric_columns[3].metric("Restricted docs", metrics["restricted_documents"])

    with st.sidebar:
        st.subheader("Session")
        st.write(f"OpenAI configured: `{'yes' if settings.openai_api_key else 'no'}`")
        st.write(f"Graph retrieval: `{'connected' if health['ok'] else 'degraded'}`")
        if not health["ok"]:
            st.warning(str(health["message"]))
        use_llm = st.checkbox("AI synthesis", value=True, help="Neo4j retrieval plus LLM synthesis.")

        if settings.enable_admin_tools:
            st.subheader("Admin")
            if st.button("Rebuild local index", use_container_width=True):
                cached_index.clear()
                cached_search.clear()
                rebuilt = cached_index(True)
                if rebuilt:
                    st.success(f"Rebuilt {len(rebuilt.documents)} documents.")
                else:
                    st.error("Local index rebuild failed.")

            if st.button("Load current index into Neo4j", use_container_width=True):
                index = cached_index(False)
                if not index:
                    st.error("No local index is available to load.")
                else:
                    try:
                        load_index_to_neo4j(index, settings)
                        cached_graph_catalog.clear()
                        cached_graph_metrics.clear()
                        cached_graph_health.clear()
                        st.success("Neo4j load complete.")
                    except Exception as error:  # pragma: no cover
                        st.error(str(error))

        st.subheader("Access scope")
        allowed_document_ids = []
        if restricted_documents:
            access_mode = st.radio(
                "Select access mode",
                options=["all_access", "limited_access"],
                format_func=lambda value: value.replace("_", " ").title(),
            )
            if access_mode == "limited_access":
                allowed_document_ids = st.multiselect(
                    "Allowed restricted documents",
                    options=[document.doc_id for document in restricted_documents],
                    format_func=lambda doc_id: documents_by_id[doc_id].title,
                )
        else:
            access_mode = "all_access"
            st.caption("No restricted documents are configured in this pilot.")

        st.subheader("Narrow the corpus")
        selected_organizations = st.multiselect("Organization", organizations)
        selected_industries = st.multiselect("Industry", industries)
        selected_topics = st.multiselect("Topic", topics)
        selected_years = st.multiselect("Year", years)

    if settings.require_graph and not health["ok"]:
        st.error("Neo4j is required for this deployment and is not currently reachable.")
        st.stop()

    st.subheader("Ask across the research")
    question = st.text_area(
        "Question",
        value="What themes are showing up in AI transformation for financial services?",
        height=100,
    )
    st.caption(
        "Try: Compare Oliver Wyman and McKinsey on AI in financial services. Or: What operating-model themes show up in transformation work?"
    )

    if st.button("Generate answer", type="primary"):
        with st.spinner("Retrieving cited passages and composing the answer..."):
            bundle = cached_search(
                question=question,
                access_mode=access_mode,
                allowed_document_ids=tuple(allowed_document_ids),
                organizations=tuple(selected_organizations),
                industries=tuple(selected_industries),
                topics=tuple(selected_topics),
                years=tuple(selected_years),
                use_llm=use_llm,
                catalog_signature=catalog_signature,
                index_generated_at=index_generated_at,
            )

        st.markdown("### Answer")
        st.caption(f"Grounded in {len(bundle.hits)} source(s)")
        st.caption("Retriever: Neo4j hybrid parent-child" if health["ok"] else "Retriever: local fallback")
        st.caption("Answer mode: AI synthesis" if bundle.used_llm else "Answer mode: extractive fallback")
        if bundle.timings:
            stage_bits = []
            if "embedding_seconds" in bundle.timings:
                stage_bits.append(f"embed {bundle.timings['embedding_seconds']:.1f}s")
            if "retrieval_seconds" in bundle.timings:
                stage_bits.append(f"retrieve {bundle.timings['retrieval_seconds']:.1f}s")
            if "synthesis_seconds" in bundle.timings:
                stage_bits.append(f"synthesize {bundle.timings['synthesis_seconds']:.1f}s")
            if "total_seconds" in bundle.timings:
                stage_bits.append(f"total {bundle.timings['total_seconds']:.1f}s")
            if stage_bits:
                st.caption(" | ".join(stage_bits))
        st.write(bundle.answer)

        st.markdown("### Sources")
        if not bundle.hits:
            st.info("No sources matched the current query and filters.")
        for index_number, hit in enumerate(bundle.hits, start=1):
            expander_label = (
                f"[{index_number}] {hit.document.title} | "
                f"{hit.document.organization} | {hit.document.published_date or 'Unknown'}"
            )
            with st.expander(expander_label):
                st.caption(f"Relevance score {hit.score:.2f}")
                if hit.parent_chunk and hit.parent_chunk.heading != hit.document.title:
                    st.write(f"Section: {hit.parent_chunk.heading}")
                if hit.document.industries:
                    st.write(f"Industry: {', '.join(hit.document.industries)}")
                if hit.document.topics:
                    st.write(f"Topic: {', '.join(hit.document.topics)}")
                st.write(hit.excerpt)
                st.caption(f"Source file: {hit.document.file_name}")

if __name__ == "__main__":
    main()
