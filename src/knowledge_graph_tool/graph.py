from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from knowledge_graph_tool.config import Settings
from knowledge_graph_tool.models import (
    AccessPrincipal,
    ChunkRecord,
    CorpusIndex,
    DocumentRecord,
    ParentChunkRecord,
    SearchFilters,
    SearchHit,
)
from knowledge_graph_tool.taxonomy import normalize_for_matching


VECTOR_INDEX_NAME = "chunk_embeddings"
FULLTEXT_INDEX_NAME = "chunk_fulltext"


def _driver(settings: Settings):
    if not settings.neo4j_password:
        raise RuntimeError("NEO4J_PASSWORD is not configured.")

    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def graph_available(settings: Settings) -> bool:
    return bool(settings.neo4j_uri and settings.neo4j_username and settings.neo4j_password)


def graph_health(settings: Settings) -> Dict[str, object]:
    if not graph_available(settings):
        return {"ok": False, "message": "Neo4j credentials are not configured."}

    try:
        with _driver(settings) as driver:
            with driver.session(database=settings.neo4j_database) as session:
                record = session.run("RETURN 1 AS ok").single()
        return {"ok": bool(record and record["ok"] == 1), "message": "Connected"}
    except Exception as error:
        return {"ok": False, "message": str(error)}


def fetch_graph_catalog(settings: Settings) -> List[DocumentRecord]:
    if not graph_available(settings):
        return []

    query = """
    MATCH (d:Document)
    RETURN
        d.doc_id AS doc_id,
        d.path AS path,
        d.file_name AS file_name,
        d.file_type AS file_type,
        d.title AS title,
        d.organization AS organization,
        d.published_date AS published_date,
        d.access_tier AS access_tier,
        d.practices AS practices,
        d.industries AS industries,
        d.topics AS topics,
        d.summary AS summary
    ORDER BY coalesce(d.organization, ''), coalesce(d.published_date, ''), coalesce(d.title, '')
    """

    with _driver(settings) as driver:
        with driver.session(database=settings.neo4j_database) as session:
            return [
                DocumentRecord(
                    doc_id=row["doc_id"],
                    path=row["path"] or "",
                    file_name=row["file_name"] or "",
                    file_type=row["file_type"] or "",
                    title=row["title"] or "Untitled",
                    organization=row["organization"] or "Unknown",
                    published_date=row["published_date"],
                    access_tier=row["access_tier"] or "all",
                    practices=row["practices"] or [],
                    industries=row["industries"] or [],
                    topics=row["topics"] or [],
                    summary=row["summary"] or "",
                )
                for row in session.run(query)
            ]


def fetch_graph_metrics(settings: Settings) -> Dict[str, int]:
    if not graph_available(settings):
        return {"documents": 0, "sections": 0, "organizations": 0, "restricted_documents": 0}

    query = """
    CALL {
        MATCH (d:Document)
        RETURN count(d) AS documents
    }
    CALL {
        MATCH (p:ParentChunk)
        RETURN count(p) AS sections
    }
    CALL {
        MATCH (o:Organization)
        RETURN count(o) AS organizations
    }
    CALL {
        MATCH (d:Document {access_tier: 'restricted'})
        RETURN count(d) AS restricted_documents
    }
    RETURN documents, sections, organizations, restricted_documents
    """

    with _driver(settings) as driver:
        with driver.session(database=settings.neo4j_database) as session:
            row = session.run(query).single()
            return {
                "documents": int(row["documents"]),
                "sections": int(row["sections"]),
                "organizations": int(row["organizations"]),
                "restricted_documents": int(row["restricted_documents"]),
            }


def ensure_schema(settings: Settings, embedding_dimensions: int = 1536) -> None:
    statements = [
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        "CREATE CONSTRAINT parent_chunk_id IF NOT EXISTS FOR (p:ParentChunk) REQUIRE p.parent_chunk_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT organization_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
        "CREATE CONSTRAINT practice_name IF NOT EXISTS FOR (p:Practice) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        (
            f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
            f"FOR (c:Chunk) ON (c.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dimensions}, "
            "`vector.similarity_function`: 'cosine'}}"
        ),
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.heading, c.text]",
    ]

    with _driver(settings) as driver:
        with driver.session(database=settings.neo4j_database) as session:
            for statement in statements:
                session.run(statement)


def reset_search_indexes(settings: Settings, embedding_dimensions: int = 1536) -> None:
    with _driver(settings) as driver:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS")
            session.run(f"DROP INDEX {FULLTEXT_INDEX_NAME} IF EXISTS")

    ensure_schema(settings, embedding_dimensions=embedding_dimensions)


def _batched(rows: List[dict], size: int = 200):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _delete_existing_content(session) -> None:
    for label in ["Chunk", "ParentChunk", "Document", "Organization", "Practice", "Topic"]:
        session.run(f"MATCH (n:{label}) DETACH DELETE n")


def load_index_to_neo4j(index: CorpusIndex, settings: Settings) -> None:
    embedding_dimensions = 1536
    for chunk in index.chunks:
        if chunk.embedding:
            embedding_dimensions = len(chunk.embedding)
            break

    reset_search_indexes(settings, embedding_dimensions=embedding_dimensions)

    with _driver(settings) as driver:
        with driver.session(database=settings.neo4j_database) as session:
            _delete_existing_content(session)

            document_rows = [document.to_dict() for document in index.documents]
            for batch in _batched(document_rows):
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (d:Document {doc_id: row.doc_id})
                    SET d.title = row.title,
                        d.path = row.path,
                        d.file_name = row.file_name,
                        d.file_type = row.file_type,
                        d.organization = row.organization,
                        d.published_date = row.published_date,
                        d.access_tier = row.access_tier,
                        d.summary = row.summary,
                        d.practices = row.practices,
                        d.industries = row.industries,
                        d.topics = row.topics
                    MERGE (o:Organization {name: row.organization})
                    MERGE (o)-[:PUBLISHED]->(d)
                    FOREACH (practice IN coalesce(row.practices, []) |
                        MERGE (p:Practice {name: practice})
                        MERGE (d)-[:IN_PRACTICE]->(p)
                    )
                    FOREACH (topic IN coalesce(row.topics, []) |
                        MERGE (t:Topic {name: topic})
                        MERGE (d)-[:HAS_TOPIC]->(t)
                    )
                    """,
                    rows=batch,
                )

            parent_rows = [parent_chunk.to_dict() for parent_chunk in index.parent_chunks]
            for batch in _batched(parent_rows):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (d:Document {doc_id: row.doc_id})
                    MERGE (p:ParentChunk {parent_chunk_id: row.parent_chunk_id})
                    SET p.parent_index = row.parent_index,
                        p.heading = row.heading,
                        p.text = row.text,
                        p.token_estimate = row.token_estimate
                    MERGE (d)-[:HAS_PARENT_CHUNK {order: row.parent_index}]->(p)
                    """,
                    rows=batch,
                )

            chunk_rows = [chunk.to_dict() for chunk in index.chunks]
            for batch in _batched(chunk_rows):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (d:Document {doc_id: row.doc_id})
                    MATCH (p:ParentChunk {parent_chunk_id: row.parent_chunk_id})
                    MERGE (c:Chunk {chunk_id: row.chunk_id})
                    SET c.chunk_index = row.chunk_index,
                        c.child_index = row.child_index,
                        c.heading = row.heading,
                        c.text = row.text,
                        c.token_estimate = row.token_estimate,
                        c.embedding = row.embedding
                    MERGE (d)-[:HAS_CHUNK {order: row.chunk_index}]->(c)
                    MERGE (p)-[:HAS_CHILD {order: row.child_index}]->(c)
                    """,
                    rows=batch,
                )

            parents_by_document: Dict[str, List[ParentChunkRecord]] = defaultdict(list)
            for parent_chunk in index.parent_chunks:
                parents_by_document[parent_chunk.doc_id].append(parent_chunk)

            parent_edge_rows: List[dict] = []
            for doc_parent_chunks in parents_by_document.values():
                doc_parent_chunks.sort(key=lambda item: item.parent_index)
                for current_parent, next_parent in zip(doc_parent_chunks, doc_parent_chunks[1:]):
                    parent_edge_rows.append(
                        {
                            "left_parent_chunk_id": current_parent.parent_chunk_id,
                            "right_parent_chunk_id": next_parent.parent_chunk_id,
                        }
                    )
            for batch in _batched(parent_edge_rows):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (left:ParentChunk {parent_chunk_id: row.left_parent_chunk_id})
                    MATCH (right:ParentChunk {parent_chunk_id: row.right_parent_chunk_id})
                    MERGE (left)-[:NEXT_PARENT]->(right)
                    """,
                    rows=batch,
                )

            chunks_by_document: Dict[str, List[ChunkRecord]] = defaultdict(list)
            for chunk in index.chunks:
                chunks_by_document[chunk.doc_id].append(chunk)

            child_edge_rows: List[dict] = []
            for doc_chunks in chunks_by_document.values():
                doc_chunks.sort(key=lambda item: item.chunk_index)
                for current_chunk, next_chunk in zip(doc_chunks, doc_chunks[1:]):
                    child_edge_rows.append(
                        {
                            "left_chunk_id": current_chunk.chunk_id,
                            "right_chunk_id": next_chunk.chunk_id,
                        }
                    )
            for batch in _batched(child_edge_rows):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (left:Chunk {chunk_id: row.left_chunk_id})
                    MATCH (right:Chunk {chunk_id: row.right_chunk_id})
                    MERGE (left)-[:NEXT_CHUNK]->(right)
                    """,
                    rows=batch,
                )


def _keyword_query(question: str) -> str:
    tokens = [token for token in normalize_for_matching(question).split() if len(token) > 2]
    if not tokens:
        return question
    return " OR ".join(tokens[:12])


def _vector_candidates(session, question_embedding: List[float], candidate_k: int) -> Dict[str, float]:
    results = session.run(
        f"""
        CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $limit, $embedding)
        YIELD node, score
        RETURN node.chunk_id AS chunk_id, score
        """,
        limit=candidate_k,
        embedding=question_embedding,
    )
    return {record["chunk_id"]: float(record["score"]) for record in results}


def _keyword_candidates(session, question: str, candidate_k: int) -> Dict[str, float]:
    results = session.run(
        f"""
        CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX_NAME}', $query_string)
        YIELD node, score
        RETURN node.chunk_id AS chunk_id, score
        LIMIT $limit
        """,
        query_string=_keyword_query(question),
        limit=candidate_k,
    )
    return {record["chunk_id"]: float(record["score"]) for record in results}


def _normalize_scores(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    maximum = max(values.values()) or 1.0
    return {key: value / maximum for key, value in values.items()}


def _candidate_scores(
    question: str,
    question_embedding: Optional[List[float]],
    session,
    candidate_k: int,
) -> Dict[str, float]:
    vector_scores = _vector_candidates(session, question_embedding, candidate_k) if question_embedding else {}
    keyword_scores = _keyword_candidates(session, question, candidate_k)
    normalized_vector = _normalize_scores(vector_scores)
    normalized_keyword = _normalize_scores(keyword_scores)

    combined_scores: Dict[str, float] = {}
    candidate_ids = set(normalized_vector) | set(normalized_keyword)
    for chunk_id in candidate_ids:
        combined_scores[chunk_id] = 0.65 * normalized_vector.get(chunk_id, 0.0) + 0.35 * normalized_keyword.get(
            chunk_id, 0.0
        )
    return combined_scores


def _candidate_rows(
    session,
    candidate_ids: List[str],
    principal: AccessPrincipal,
    filters: SearchFilters | None,
) -> List[dict]:
    if not candidate_ids:
        return []

    query = """
    UNWIND $candidate_ids AS candidate_id
    MATCH (d:Document)-[:HAS_PARENT_CHUNK]->(p:ParentChunk)-[:HAS_CHILD]->(c:Chunk {chunk_id: candidate_id})
    WHERE (
        $access_mode = 'all_access'
        OR d.access_tier = 'all'
        OR d.doc_id IN $allowed_document_ids
    )
    AND (size($organizations) = 0 OR d.organization IN $organizations)
    AND (size($years) = 0 OR substring(coalesce(d.published_date, ''), 0, 4) IN $years)
    AND (size($practices) = 0 OR any(item IN coalesce(d.practices, []) WHERE item IN $practices))
    AND (size($industries) = 0 OR any(item IN coalesce(d.industries, []) WHERE item IN $industries))
    AND (size($topics) = 0 OR any(item IN coalesce(d.topics, []) WHERE item IN $topics))
    OPTIONAL MATCH (prev_parent:ParentChunk)-[:NEXT_PARENT]->(p)
    OPTIONAL MATCH (p)-[:NEXT_PARENT]->(next_parent:ParentChunk)
    RETURN
        d.doc_id AS doc_id,
        d.title AS title,
        d.path AS path,
        d.file_name AS file_name,
        d.file_type AS file_type,
        d.organization AS organization,
        d.published_date AS published_date,
        d.access_tier AS access_tier,
        d.summary AS summary,
        d.practices AS practices,
        d.industries AS industries,
        d.topics AS topics,
        p.parent_chunk_id AS parent_chunk_id,
        p.parent_index AS parent_index,
        p.heading AS parent_heading,
        p.text AS parent_text,
        p.token_estimate AS parent_token_estimate,
        c.chunk_id AS chunk_id,
        c.chunk_index AS chunk_index,
        c.child_index AS child_index,
        c.heading AS chunk_heading,
        c.text AS chunk_text,
        c.token_estimate AS chunk_token_estimate,
        prev_parent.text AS prev_parent_text,
        next_parent.text AS next_parent_text
    """

    parameters = {
        "candidate_ids": candidate_ids,
        "access_mode": principal.mode,
        "allowed_document_ids": principal.allowed_document_ids,
        "organizations": filters.organizations if filters else [],
        "years": filters.years if filters else [],
        "practices": filters.practices if filters else [],
        "industries": filters.industries if filters else [],
        "topics": filters.topics if filters else [],
    }

    return [dict(record) for record in session.run(query, **parameters)]


def _build_excerpt(parent_text: str, prev_parent_text: Optional[str], next_parent_text: Optional[str]) -> str:
    parts: List[str] = []
    if prev_parent_text and len(parent_text) < 1100:
        parts.append(prev_parent_text.strip()[-400:])
    parts.append(parent_text.strip())
    if next_parent_text and len("\n\n".join(parts)) < 1700:
        parts.append(next_parent_text.strip()[:700])
    return "\n\n".join(part for part in parts if part).strip()[:2200]


def _aggregate_parent_hits(
    candidate_ids: List[str],
    candidate_scores: Dict[str, float],
    rows: List[dict],
) -> List[SearchHit]:
    rows_by_chunk_id = {row["chunk_id"]: row for row in rows}
    parent_states: Dict[str, dict] = {}

    for chunk_id in candidate_ids:
        row = rows_by_chunk_id.get(chunk_id)
        if not row:
            continue

        score = candidate_scores.get(chunk_id, 0.0)
        parent_chunk_id = row["parent_chunk_id"]
        state = parent_states.get(parent_chunk_id)
        if state is None:
            document = DocumentRecord(
                doc_id=row["doc_id"],
                path=row["path"],
                file_name=row["file_name"],
                file_type=row["file_type"],
                title=row["title"],
                organization=row["organization"],
                published_date=row["published_date"],
                access_tier=row["access_tier"],
                practices=row["practices"] or [],
                industries=row["industries"] or [],
                topics=row["topics"] or [],
                summary=row["summary"] or "",
            )
            parent_chunk = ParentChunkRecord(
                parent_chunk_id=row["parent_chunk_id"],
                doc_id=row["doc_id"],
                parent_index=row["parent_index"],
                heading=row["parent_heading"],
                text=row["parent_text"],
                token_estimate=row["parent_token_estimate"]
                or max(1, len((row["parent_text"] or "").split())),
            )
            child_chunk = ChunkRecord(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                parent_chunk_id=row["parent_chunk_id"],
                chunk_index=row["chunk_index"],
                child_index=row["child_index"] or 0,
                heading=row["chunk_heading"],
                text=row["chunk_text"],
                token_estimate=row["chunk_token_estimate"] or max(1, len((row["chunk_text"] or "").split())),
            )
            state = {
                "document": document,
                "parent_chunk": parent_chunk,
                "best_chunk": child_chunk,
                "best_score": score,
                "score_sum": score,
                "match_count": 1,
                "prev_parent_text": row.get("prev_parent_text"),
                "next_parent_text": row.get("next_parent_text"),
            }
            parent_states[parent_chunk_id] = state
            continue

        state["score_sum"] += score
        state["match_count"] += 1
        if score > state["best_score"]:
            state["best_score"] = score
            state["best_chunk"] = ChunkRecord(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                parent_chunk_id=row["parent_chunk_id"],
                chunk_index=row["chunk_index"],
                child_index=row["child_index"] or 0,
                heading=row["chunk_heading"],
                text=row["chunk_text"],
                token_estimate=row["chunk_token_estimate"] or max(1, len((row["chunk_text"] or "").split())),
            )

    hits: List[SearchHit] = []
    for state in parent_states.values():
        aggregate_score = state["best_score"] + min(0.18, 0.06 * (state["match_count"] - 1))
        excerpt = _build_excerpt(
            state["parent_chunk"].text,
            state["prev_parent_text"],
            state["next_parent_text"],
        )
        hits.append(
            SearchHit(
                score=aggregate_score,
                document=state["document"],
                chunk=state["best_chunk"],
                parent_chunk=state["parent_chunk"],
                excerpt=excerpt,
            )
        )

    hits.sort(key=lambda item: item.score, reverse=True)
    return hits


def hybrid_graph_search(
    question: str,
    settings: Settings,
    principal: AccessPrincipal,
    question_embedding: Optional[List[float]],
    top_k: int,
    filters: Optional[SearchFilters] = None,
) -> List[SearchHit]:
    candidate_k = max(top_k * 10, 30)
    with _driver(settings) as driver:
        with driver.session(database=settings.neo4j_database) as session:
            candidate_scores = _candidate_scores(question, question_embedding, session, candidate_k)
            ranked_candidate_ids = [
                chunk_id
                for chunk_id, _ in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            ]
            rows = _candidate_rows(session, ranked_candidate_ids, principal, filters)

    return _aggregate_parent_hits(ranked_candidate_ids, candidate_scores, rows)[:top_k]
