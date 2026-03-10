from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from knowledge_graph_tool.config import Settings
from knowledge_graph_tool.llm import OpenAIService
from knowledge_graph_tool.models import ChunkRecord, CorpusIndex, DocumentRecord, ParentChunkRecord
from knowledge_graph_tool.parsers import (
    build_document_id,
    extract_text,
    split_into_parent_child_chunks,
    summarize_text,
)
from knowledge_graph_tool.taxonomy import extract_matches, INDUSTRY_KEYWORDS, PRACTICE_KEYWORDS, TOPIC_KEYWORDS


def load_restricted_manifest(path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not path.exists():
        return {"restricted_documents": [], "access_groups": {}}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "restricted_documents": payload.get("restricted_documents", []),
        "access_groups": payload.get("access_groups", {}),
    }


def load_pdf_metadata_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pdf_cache_key(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"


def iter_source_files(corpus_dir: Path) -> List[Path]:
    supported_extensions = {".md", ".html", ".pdf"}
    return sorted(
        path for path in corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported_extensions
    )


def maybe_enrich_pdf_metadata(
    path: Path,
    metadata: Dict[str, str],
    text: str,
    service: OpenAIService,
    cache: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    if path.suffix.lower() != ".pdf" or not service.enabled:
        return metadata

    cache_key = pdf_cache_key(path)
    if cache_key in cache:
        cached = cache[cache_key]
        metadata.update(cached)
        return metadata

    try:
        enriched = service.infer_pdf_metadata(
            file_name=path.name,
            sample_text=text,
            current_title=metadata.get("title", path.stem),
            current_organization=metadata.get("organization", "Unknown"),
            current_published_date=metadata.get("published_date"),
        )
        metadata.update({key: value for key, value in enriched.items() if value})
        cache[cache_key] = {
            "title": metadata.get("title", path.stem),
            "organization": metadata.get("organization", "Unknown"),
            "published_date": metadata.get("published_date"),
        }
    except Exception:
        return metadata

    return metadata


def build_records(
    path: Path,
    settings: Settings,
    restricted_files: List[str],
    service: OpenAIService,
    pdf_metadata_cache: Dict[str, Dict[str, str]],
) -> Tuple[DocumentRecord, List[ParentChunkRecord], List[ChunkRecord]]:
    text, metadata = extract_text(path)
    metadata = maybe_enrich_pdf_metadata(path, metadata, text, service, pdf_metadata_cache)
    doc_id = build_document_id(path)
    title = metadata.get("title") or path.stem
    organization = metadata.get("organization") or "Unknown"
    published_date = metadata.get("published_date")
    joined_text = f"{title}\n\n{text}"

    document = DocumentRecord(
        doc_id=doc_id,
        path=str(path.resolve()),
        file_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
        title=title,
        organization=organization,
        published_date=published_date,
        access_tier="restricted" if path.name in restricted_files else "all",
        practices=extract_matches(joined_text, PRACTICE_KEYWORDS),
        industries=extract_matches(joined_text, INDUSTRY_KEYWORDS),
        topics=extract_matches(joined_text, TOPIC_KEYWORDS),
        summary=summarize_text(text),
        metadata={"source_type": path.suffix.lower().lstrip(".")},
    )

    parent_chunks: List[ParentChunkRecord] = []
    chunks = []
    hierarchy = split_into_parent_child_chunks(
        text=text,
        default_heading=title,
        parent_max_chars=settings.parent_chunk_size,
        child_max_chars=settings.child_chunk_size,
        child_overlap_chars=settings.child_chunk_overlap,
    )

    chunk_index = 0
    for parent_index, parent_payload in enumerate(hierarchy):
        parent_chunk_id = f"{doc_id}:p{parent_index}"
        parent_text = str(parent_payload["text"])
        heading = str(parent_payload["heading"])
        parent_chunks.append(
            ParentChunkRecord(
                parent_chunk_id=parent_chunk_id,
                doc_id=doc_id,
                parent_index=parent_index,
                heading=heading,
                text=parent_text,
                token_estimate=max(1, len(parent_text.split())),
                metadata={"path": str(path.resolve())},
            )
        )

        for child_index, chunk_text in enumerate(parent_payload["children"]):
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{parent_chunk_id}:c{child_index}",
                    doc_id=doc_id,
                    parent_chunk_id=parent_chunk_id,
                    chunk_index=chunk_index,
                    child_index=child_index,
                    heading=heading,
                    text=str(chunk_text),
                    token_estimate=max(1, len(str(chunk_text).split())),
                    metadata={"path": str(path.resolve())},
                )
            )
            chunk_index += 1
    return document, parent_chunks, chunks


def maybe_attach_embeddings(index: CorpusIndex, settings: Settings) -> None:
    service = OpenAIService(settings)
    if not service.enabled:
        return

    batch_size = 32
    for start in range(0, len(index.chunks), batch_size):
        batch = index.chunks[start : start + batch_size]
        vectors = service.embed_texts(chunk.text for chunk in batch)
        for chunk, vector in zip(batch, vectors):
            chunk.embedding = vector


def build_index(settings: Settings, persist: bool = True) -> CorpusIndex:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_restricted_manifest(settings.restricted_manifest_path)
    restricted_files = manifest["restricted_documents"]
    pdf_metadata_cache_path = settings.data_dir / "pdf_metadata_cache.json"
    pdf_metadata_cache = load_pdf_metadata_cache(pdf_metadata_cache_path)
    service = OpenAIService(settings)
    documents: List[DocumentRecord] = []
    parent_chunks: List[ParentChunkRecord] = []
    chunks: List[ChunkRecord] = []

    for path in iter_source_files(settings.corpus_dir):
        document, parent_chunk_records, chunk_records = build_records(
            path,
            settings,
            restricted_files,
            service,
            pdf_metadata_cache,
        )
        documents.append(document)
        parent_chunks.extend(parent_chunk_records)
        chunks.extend(chunk_records)

    index = CorpusIndex(
        generated_at=datetime.now(timezone.utc).isoformat(),
        documents=documents,
        parent_chunks=parent_chunks,
        chunks=chunks,
    )
    maybe_attach_embeddings(index, settings)

    if persist:
        settings.local_index_path.write_text(
            json.dumps(index.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        pdf_metadata_cache_path.write_text(
            json.dumps(pdf_metadata_cache, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    return index


def load_index(settings: Settings) -> CorpusIndex:
    payload = json.loads(settings.local_index_path.read_text(encoding="utf-8"))
    return CorpusIndex.from_dict(payload)


def build_or_load_index(settings: Settings, rebuild: bool = False) -> CorpusIndex:
    if rebuild or not settings.local_index_path.exists():
        return build_index(settings)
    return load_index(settings)
