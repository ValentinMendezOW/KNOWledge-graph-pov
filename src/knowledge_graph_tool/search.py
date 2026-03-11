from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set

from knowledge_graph_tool.config import Settings
from knowledge_graph_tool.llm import OpenAIService
from knowledge_graph_tool.models import (
    AccessPrincipal,
    AnswerBundle,
    CorpusIndex,
    DocumentRecord,
    ParentChunkRecord,
    SearchFilters,
    SearchHit,
)
from knowledge_graph_tool.taxonomy import ORGANIZATION_ALIASES, normalize_for_matching


COMPARISON_HINTS = [
    " compare ",
    " compared ",
    " versus ",
    " vs ",
    " difference ",
    " differ ",
    " against ",
    " relative to ",
]
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "article",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "report",
    "the",
    "to",
    "what",
    "with",
}
MONTH_TOKENS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]


def cosine_similarity(left: List[float], right: List[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def lexical_score(query_tokens: List[str], text: str) -> float:
    text_tokens = normalize_tokens(text)
    if not text_tokens:
        return 0.0
    counts = Counter(text_tokens)
    return sum(counts[token] for token in query_tokens) / math.sqrt(len(text_tokens))


def extract_candidate_sentences(text: str) -> List[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    return [sentence for sentence in sentences if len(sentence.split()) >= 8]


def extract_mentioned_organizations(question: str, available_organizations: Set[str]) -> List[str]:
    normalized_question = f" {normalize_for_matching(question)} "
    matches: List[str] = []
    for organization in sorted(available_organizations, key=len, reverse=True):
        aliases = ORGANIZATION_ALIASES.get(organization, []) + [organization]
        if any(f" {normalize_for_matching(alias)} " in normalized_question for alias in aliases):
            matches.append(organization)
    return matches


def is_comparison_query(question: str) -> bool:
    normalized_question = f" {normalize_for_matching(question)} "
    return any(hint in normalized_question for hint in COMPARISON_HINTS)


def title_fingerprint(title: str) -> str:
    tokens = [
        token
        for token in normalize_for_matching(title).split()
        if token not in TITLE_STOPWORDS
        and token not in MONTH_TOKENS
        and not re.fullmatch(r"\d{4}", token)
    ]
    return " ".join(tokens[:10])


def query_profile(question: str, documents: Iterable[DocumentRecord]) -> tuple[List[str], bool]:
    available_organizations = {
        document.organization for document in documents if document.organization != "Unknown"
    }
    return extract_mentioned_organizations(question, available_organizations), is_comparison_query(question)


def enrich_question_with_organizations(question: str, mentioned_organizations: List[str]) -> str:
    if not mentioned_organizations:
        return question

    normalized_question = f" {normalize_for_matching(question)} "
    additions = []
    for organization in mentioned_organizations:
        normalized_org = f" {normalize_for_matching(organization)} "
        if normalized_org not in normalized_question:
            additions.append(organization)

    if not additions:
        return question
    return f"{question} ({', '.join(additions)})"


def scoped_filters(filters: Optional[SearchFilters], organizations: List[str]) -> SearchFilters:
    return SearchFilters(
        organizations=organizations,
        practices=list(filters.practices) if filters else [],
        industries=list(filters.industries) if filters else [],
        topics=list(filters.topics) if filters else [],
        years=list(filters.years) if filters else [],
    )


def constrain_filters(
    filters: Optional[SearchFilters],
    mentioned_organizations: List[str],
) -> SearchFilters:
    base_organizations = list(filters.organizations) if filters else []
    effective_organizations = base_organizations

    if mentioned_organizations:
        if base_organizations:
            overlap = [organization for organization in mentioned_organizations if organization in base_organizations]
            effective_organizations = overlap or mentioned_organizations
        else:
            effective_organizations = mentioned_organizations

    return SearchFilters(
        organizations=effective_organizations,
        practices=list(filters.practices) if filters else [],
        industries=list(filters.industries) if filters else [],
        topics=list(filters.topics) if filters else [],
        years=list(filters.years) if filters else [],
    )


def boost_hits_for_query(
    hits: List[SearchHit],
    mentioned_organizations: List[str],
    comparison_query: bool,
) -> List[SearchHit]:
    for hit in hits:
        if hit.document.organization in mentioned_organizations:
            hit.score += 0.22 if comparison_query else 0.14
    return hits


def can_access(doc_id: str, access_tier: str, principal: AccessPrincipal) -> bool:
    if principal.mode == "all_access":
        return True
    if access_tier == "all":
        return True
    return doc_id in set(principal.allowed_document_ids)


class LocalSearchEngine:
    def __init__(self, index: CorpusIndex, settings: Settings) -> None:
        self.index = index
        self.settings = settings
        self.documents_by_id = {document.doc_id: document for document in index.documents}
        self.parent_chunks_by_id = {
            parent_chunk.parent_chunk_id: parent_chunk for parent_chunk in index.parent_chunks
        }
        self.parent_chunks_by_doc: Dict[str, List[ParentChunkRecord]] = defaultdict(list)
        for parent_chunk in index.parent_chunks:
            self.parent_chunks_by_doc[parent_chunk.doc_id].append(parent_chunk)
        for parent_chunks in self.parent_chunks_by_doc.values():
            parent_chunks.sort(key=lambda item: item.parent_index)

    @staticmethod
    def matches_filters(document, filters: Optional[SearchFilters]) -> bool:
        if not filters:
            return True
        if filters.organizations and document.organization not in filters.organizations:
            return False
        if filters.practices and not set(filters.practices).intersection(document.practices):
            return False
        if filters.industries and not set(filters.industries).intersection(document.industries):
            return False
        if filters.topics and not set(filters.topics).intersection(document.topics):
            return False
        if filters.years:
            year = (document.published_date or "")[:4]
            if year not in filters.years:
                return False
        return True

    def accessible_doc_ids(self, principal: AccessPrincipal) -> Set[str]:
        return {
            document.doc_id
            for document in self.index.documents
            if can_access(document.doc_id, document.access_tier, principal)
        }

    def build_parent_excerpt(self, parent_chunk: ParentChunkRecord) -> str:
        parent_chunks = self.parent_chunks_by_doc.get(parent_chunk.doc_id, [])
        previous_text = None
        next_text = None

        for index, item in enumerate(parent_chunks):
            if item.parent_chunk_id != parent_chunk.parent_chunk_id:
                continue
            if index > 0:
                previous_text = parent_chunks[index - 1].text
            if index + 1 < len(parent_chunks):
                next_text = parent_chunks[index + 1].text
            break

        parts: List[str] = []
        if previous_text and len(parent_chunk.text) < 1100:
            parts.append(previous_text.strip()[-400:])
        parts.append(parent_chunk.text.strip())
        if next_text and len("\n\n".join(parts)) < 1700:
            parts.append(next_text.strip()[:700])
        return "\n\n".join(part for part in parts if part).strip()[:2200]

    @staticmethod
    def unique_document_hits(hits: List[SearchHit]) -> List[SearchHit]:
        best_by_fingerprint = {}
        for hit in hits:
            fingerprint = (hit.document.organization, title_fingerprint(hit.document.title))
            current = best_by_fingerprint.get(fingerprint)
            if current is None or hit.score > current.score:
                best_by_fingerprint[fingerprint] = hit
        return sorted(best_by_fingerprint.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def select_diverse_hits(
        hits: List[SearchHit],
        limit: int,
        mentioned_organizations: List[str],
        comparison_query: bool,
    ) -> List[SearchHit]:
        if len(hits) <= limit:
            return hits

        selected: List[SearchHit] = []
        org_counts = defaultdict(int)
        selected_keys = set()

        def add_hit(hit: SearchHit) -> None:
            key = (hit.document.organization, title_fingerprint(hit.document.title))
            if key in selected_keys:
                return
            selected_keys.add(key)
            selected.append(hit)
            org_counts[hit.document.organization] += 1

        if mentioned_organizations:
            per_org_target = 2 if comparison_query and len(mentioned_organizations) > 1 else 1
            grouped_hits = {
                organization: [hit for hit in hits if hit.document.organization == organization]
                for organization in mentioned_organizations
            }
            for round_index in range(per_org_target):
                for organization in mentioned_organizations:
                    organization_hits = grouped_hits.get(organization, [])
                    if len(organization_hits) > round_index and len(selected) < limit:
                        add_hit(organization_hits[round_index])

        for hit in hits:
            if len(selected) >= limit:
                break
            if comparison_query and mentioned_organizations and hit.document.organization in mentioned_organizations:
                if org_counts[hit.document.organization] >= 2:
                    continue
            add_hit(hit)

        return selected[:limit]

    def search(
        self,
        question: str,
        principal: AccessPrincipal,
        question_embedding: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchHit]:
        limit = top_k or self.settings.top_k
        accessible_ids = self.accessible_doc_ids(principal)
        query_tokens = normalize_tokens(question)
        mentioned_organizations, comparison_query = query_profile(question, self.index.documents)
        candidate_states: Dict[str, dict] = {}

        for chunk in self.index.chunks:
            if chunk.doc_id not in accessible_ids:
                continue

            document = self.documents_by_id[chunk.doc_id]
            if not self.matches_filters(document, filters):
                continue
            score = lexical_score(query_tokens, f"{document.title} {chunk.text}")
            if question_embedding and chunk.embedding:
                score = max(score, cosine_similarity(question_embedding, chunk.embedding))

            if score <= 0:
                continue

            parent_chunk = self.parent_chunks_by_id.get(chunk.parent_chunk_id)
            parent_key = parent_chunk.parent_chunk_id if parent_chunk else chunk.chunk_id
            state = candidate_states.get(parent_key)
            if state is None:
                candidate_states[parent_key] = {
                    "document": document,
                    "parent_chunk": parent_chunk,
                    "best_chunk": chunk,
                    "best_score": score,
                    "match_count": 1,
                }
                continue

            state["match_count"] += 1
            if score > state["best_score"]:
                state["best_score"] = score
                state["best_chunk"] = chunk

        hits: List[SearchHit] = []
        for state in candidate_states.values():
            parent_chunk = state["parent_chunk"]
            excerpt = parent_chunk.text[:1800].strip() if parent_chunk else state["best_chunk"].text[:800].strip()
            if parent_chunk:
                excerpt = self.build_parent_excerpt(parent_chunk)
            hits.append(
                SearchHit(
                    score=state["best_score"] + min(0.18, 0.06 * (state["match_count"] - 1)),
                    document=state["document"],
                    chunk=state["best_chunk"],
                    excerpt=excerpt,
                    parent_chunk=parent_chunk,
                )
            )

        scored_hits = boost_hits_for_query(hits, mentioned_organizations, comparison_query)
        scored_hits.sort(key=lambda item: item.score, reverse=True)
        deduped_hits = self.unique_document_hits(scored_hits)
        return self.select_diverse_hits(deduped_hits, limit, mentioned_organizations, comparison_query)


def format_fallback_answer(question: str, hits: List[SearchHit]) -> str:
    if not hits:
        return f"No relevant sources were found for: {question}"

    query_tokens = set(normalize_tokens(question))
    selected_sentences: List[str] = []
    seen_sentences: Set[str] = set()

    for index, hit in enumerate(hits, start=1):
        best_sentence = ""
        best_score = -1.0
        for sentence in extract_candidate_sentences(hit.excerpt):
            normalized = normalize_for_matching(sentence)
            if normalized in seen_sentences:
                continue
            overlap = len(query_tokens.intersection(normalize_tokens(sentence)))
            score = overlap + min(len(sentence) / 240, 1.0)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if best_sentence:
            seen_sentences.add(normalize_for_matching(best_sentence))
            selected_sentences.append(f"- {best_sentence} [{index}]")

    if not selected_sentences:
        selected_sentences = [
            f"- {hits[0].excerpt[:260].strip()} [1]"
        ]

    intro = "Grounded take from the matched sources:"
    return "\n".join([intro, *selected_sentences[:5]])


def answer_question(
    question: str,
    principal: AccessPrincipal,
    index: Optional[CorpusIndex],
    settings: Settings,
    filters: Optional[SearchFilters] = None,
    use_llm: bool = True,
    documents: Optional[List[DocumentRecord]] = None,
) -> AnswerBundle:
    timings: Dict[str, float] = {}
    llm = OpenAIService(settings)
    profile_documents = documents or (index.documents if index else [])
    mentioned_organizations, comparison_query = query_profile(question, profile_documents)
    retrieval_question = enrich_question_with_organizations(question, mentioned_organizations)
    question_embedding = None
    if llm.enabled:
        try:
            start = time.perf_counter()
            question_embedding = llm.embed_texts([retrieval_question])[0]
            timings["embedding_seconds"] = round(time.perf_counter() - start, 3)
        except Exception:
            question_embedding = None
    effective_filters = constrain_filters(filters, mentioned_organizations)
    hits: List[SearchHit] = []
    graph_error: Optional[Exception] = None

    try:
        from knowledge_graph_tool.graph import hybrid_graph_search

        start = time.perf_counter()
        hits = hybrid_graph_search(
            question=retrieval_question,
            settings=settings,
            principal=principal,
            question_embedding=question_embedding,
            top_k=settings.top_k,
            filters=effective_filters,
        )
        if comparison_query and mentioned_organizations:
            covered_organizations = {hit.document.organization for hit in hits}
            for organization in mentioned_organizations:
                if organization in covered_organizations:
                    continue
                organization_hits = hybrid_graph_search(
                    question=retrieval_question,
                    settings=settings,
                    principal=principal,
                    question_embedding=question_embedding,
                    top_k=max(2, settings.top_k // 2),
                    filters=scoped_filters(effective_filters, [organization]),
                )
                if organization_hits:
                    hits.extend(organization_hits)
                    covered_organizations.update(hit.document.organization for hit in organization_hits)
        timings["retrieval_seconds"] = round(time.perf_counter() - start, 3)
        hits = boost_hits_for_query(hits, mentioned_organizations, comparison_query)
        hits.sort(key=lambda item: item.score, reverse=True)
        hits = LocalSearchEngine.unique_document_hits(hits)
        hits = LocalSearchEngine.select_diverse_hits(
            hits,
            settings.top_k,
            mentioned_organizations,
            comparison_query,
        )
    except Exception as error:
        graph_error = error
        if settings.require_graph and not index:
            return AnswerBundle(
                answer=f"Graph retrieval is unavailable: {error}",
                hits=[],
                used_llm=False,
                timings=timings,
            )
        if not index:
            return AnswerBundle(
                answer="Graph retrieval is unavailable and no local fallback index is present.",
                hits=[],
                used_llm=False,
                timings=timings,
            )
        start = time.perf_counter()
        engine = LocalSearchEngine(index, settings)
        hits = engine.search(
            retrieval_question,
            principal,
            question_embedding=question_embedding,
            filters=effective_filters,
        )
        timings["retrieval_seconds"] = round(time.perf_counter() - start, 3)

    if not hits:
        if graph_error and settings.require_graph:
            return AnswerBundle(
                answer=f"No matching sources were found. Graph retrieval error: {graph_error}",
                hits=[],
                used_llm=False,
                timings=timings,
            )
        return AnswerBundle(answer="No matching sources were found.", hits=[], used_llm=False, timings=timings)

    if llm.enabled and use_llm:
        try:
            start = time.perf_counter()
            answer = llm.synthesize_answer(question, hits)
            timings["synthesis_seconds"] = round(time.perf_counter() - start, 3)
            timings["total_seconds"] = round(sum(timings.values()), 3)
            return AnswerBundle(answer=answer, hits=hits, used_llm=True, timings=timings)
        except Exception:
            pass

    timings["total_seconds"] = round(sum(timings.values()), 3)
    return AnswerBundle(
        answer=format_fallback_answer(question, hits),
        hits=hits,
        used_llm=False,
        timings=timings,
    )
