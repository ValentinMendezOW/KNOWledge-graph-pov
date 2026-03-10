from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentRecord:
    doc_id: str
    path: str
    file_name: str
    file_type: str
    title: str
    organization: str
    published_date: Optional[str]
    access_tier: str
    practices: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentRecord":
        return cls(**payload)


@dataclass
class ParentChunkRecord:
    parent_chunk_id: str
    doc_id: str
    parent_index: int
    heading: str
    text: str
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ParentChunkRecord":
        return cls(**payload)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    parent_chunk_id: str
    chunk_index: int
    child_index: int
    heading: str
    text: str
    token_estimate: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChunkRecord":
        return cls(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            parent_chunk_id=payload.get("parent_chunk_id", ""),
            chunk_index=payload["chunk_index"],
            child_index=payload.get("child_index", payload["chunk_index"]),
            heading=payload["heading"],
            text=payload["text"],
            token_estimate=payload["token_estimate"],
            embedding=payload.get("embedding"),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class CorpusIndex:
    generated_at: str
    documents: List[DocumentRecord]
    parent_chunks: List[ParentChunkRecord]
    chunks: List[ChunkRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "documents": [document.to_dict() for document in self.documents],
            "parent_chunks": [parent_chunk.to_dict() for parent_chunk in self.parent_chunks],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CorpusIndex":
        return cls(
            generated_at=payload["generated_at"],
            documents=[DocumentRecord.from_dict(item) for item in payload["documents"]],
            parent_chunks=[
                ParentChunkRecord.from_dict(item) for item in payload.get("parent_chunks", [])
            ],
            chunks=[ChunkRecord.from_dict(item) for item in payload["chunks"]],
        )


@dataclass
class AccessPrincipal:
    mode: str
    allowed_document_ids: List[str] = field(default_factory=list)


@dataclass
class SearchFilters:
    organizations: List[str] = field(default_factory=list)
    practices: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    years: List[str] = field(default_factory=list)


@dataclass
class SearchHit:
    score: float
    document: DocumentRecord
    chunk: ChunkRecord
    excerpt: str
    parent_chunk: Optional[ParentChunkRecord] = None


@dataclass
class AnswerBundle:
    answer: str
    hits: List[SearchHit]
    used_llm: bool
    timings: Dict[str, float] = field(default_factory=dict)
