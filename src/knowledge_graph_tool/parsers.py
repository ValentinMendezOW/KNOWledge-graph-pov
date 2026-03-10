from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from pypdf import PdfReader
from knowledge_graph_tool.taxonomy import (
    ORGANIZATION_ALIASES,
    canonicalize_organization,
    normalize_for_matching,
)


DATE_PATTERNS = [
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})"),
    re.compile(r"(?P<date>\d{4}-\d{2})"),
    re.compile(r"(?P<date>\d{4})"),
]


def file_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def parse_filename_metadata(path: Path) -> Dict[str, Optional[str]]:
    stem = path.stem
    parts = stem.split("_")
    organization = canonicalize_organization(parts[0].replace("-", " ").strip()) if parts else None
    published_date = None
    date_index = None

    for index, part in enumerate(parts[1:], start=1):
        for pattern in DATE_PATTERNS:
            match = pattern.fullmatch(part)
            if match:
                published_date = match.group("date")
                date_index = index
                break
        if published_date:
            break

    title_parts = parts[(date_index + 1) if date_index is not None else 1 :]
    if not title_parts:
        title_parts = [stem]

    title = " ".join(part.replace("-", " ").strip() for part in title_parts).strip()
    return {
        "organization": organization,
        "published_date": published_date,
        "title": title or stem,
    }


def infer_organization(text: str, path: Path, *candidates: Optional[str]) -> str:
    for candidate in candidates:
        canonical = canonicalize_organization(candidate)
        if canonical:
            return canonical

    search_text = f" {normalize_for_matching(path.stem)} {normalize_for_matching(text[:50000])} "
    for canonical, patterns in ORGANIZATION_ALIASES.items():
        if any(f" {normalize_for_matching(pattern)} " in search_text for pattern in patterns):
            return canonical

    return "Unknown"


def guess_pdf_title(full_text: str, organization: str) -> Optional[str]:
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    if not lines:
        return None

    guessed_lines: List[str] = []
    for line in lines[:12]:
        normalized = normalize_for_matching(line)
        if not normalized:
            continue
        if organization != "Unknown" and canonicalize_organization(line) == organization:
            continue
        if normalized in {"authors", "contents"}:
            break
        if re.search(r"\b(19|20)\d{2}\b", line) and guessed_lines:
            break
        if normalized.endswith("explainers"):
            continue
        if line.lower().startswith("by "):
            break
        guessed_lines.append(line)
        if len(" ".join(guessed_lines)) > 140:
            break

    if not guessed_lines:
        return None
    return " ".join(guessed_lines)[:180].strip()


def clean_markdown(text: str) -> str:
    text = re.sub(r"`{3}.*?`{3}", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^[#>\-\*\s]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_markdown(path: Path) -> Tuple[str, Dict[str, str]]:
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    metadata = parse_filename_metadata(path)
    organization_match = re.search(
        r"^\*\*Organization:\*\*\s*(?P<organization>.+?)\s*$",
        raw_text,
        flags=re.MULTILINE,
    )
    title = metadata["title"] or path.stem
    first_heading = next(
        (line.lstrip("# ").strip() for line in raw_text.splitlines() if line.startswith("#")),
        title,
    )
    metadata["title"] = first_heading or title
    cleaned = clean_markdown(raw_text)
    metadata["organization"] = infer_organization(
        cleaned,
        path,
        organization_match.group("organization") if organization_match else metadata.get("organization"),
        metadata.get("title"),
    )
    return cleaned, metadata


def extract_html(path: Path) -> Tuple[str, Dict[str, str]]:
    raw_html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(" ", strip=True)
    metadata = parse_filename_metadata(path)
    title = (
        soup.find("meta", attrs={"name": "pagetitle"})
        or soup.find("meta", attrs={"property": "og:title"})
        or soup.title
    )
    if title:
        metadata["title"] = title.get("content") if hasattr(title, "get") else title.get_text(strip=True)

    date_meta = soup.find("meta", attrs={"name": "dateuser"}) or soup.find(
        "meta", attrs={"name": "firstpublishdate"}
    )
    if date_meta and date_meta.get("content"):
        metadata["published_date"] = date_meta["content"][:10]

    author_meta = soup.find("meta", attrs={"name": "author"}) or soup.find(
        "meta", attrs={"name": "author_boost"}
    )
    metadata["organization"] = infer_organization(
        text,
        path,
        author_meta.get("content") if author_meta and author_meta.get("content") else metadata.get("organization"),
        metadata.get("title"),
    )
    return text, metadata


def extract_pdf(path: Path) -> Tuple[str, Dict[str, str]]:
    reader = PdfReader(str(path))
    text_parts: List[str] = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    metadata = parse_filename_metadata(path)
    metadata_has_title = bool(reader.metadata and reader.metadata.title)
    if metadata_has_title:
        metadata["title"] = reader.metadata.title
    full_text = " ".join(text_parts).strip()
    metadata["organization"] = infer_organization(
        full_text,
        path,
        metadata.get("organization"),
        metadata.get("title"),
        reader.metadata.author if reader.metadata and reader.metadata.author else None,
    )
    if not metadata_has_title:
        guessed_title = guess_pdf_title(full_text, metadata["organization"])
        if guessed_title:
            metadata["title"] = guessed_title
    return full_text, metadata


def extract_text(path: Path) -> Tuple[str, Dict[str, str]]:
    extension = path.suffix.lower()
    if extension == ".md":
        return extract_markdown(path)
    if extension == ".html":
        return extract_html(path)
    if extension == ".pdf":
        return extract_pdf(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def build_document_id(path: Path) -> str:
    return file_hash(str(path.resolve()))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _split_words(text: str, max_chars: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            current = word
            continue
        current = candidate

    if current:
        chunks.append(current.strip())
    return chunks


def _split_sentences(text: str, max_chars: int) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    if len(sentences) == 1:
        return _split_words(normalized, max_chars)

    units: List[str] = []
    for sentence in sentences:
        sentence = _normalize_text(sentence)
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        units.extend(_split_words(sentence, max_chars))
    return units


def _paragraph_units(text: str, max_chars: int) -> List[str]:
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    units: List[str] = []
    for paragraph in paragraphs:
        paragraph = _normalize_text(paragraph)
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        units.extend(_split_sentences(paragraph, max_chars))

    return units


def _assemble_chunks(units: List[str], max_chars: int, overlap_chars: int = 0) -> List[str]:
    if not units:
        return []

    chunks: List[str] = []
    current_units: List[str] = []
    current_length = 0
    index = 0

    while index < len(units):
        unit = units[index]
        separator_length = 2 if current_units else 0
        candidate_length = current_length + separator_length + len(unit)
        if current_units and candidate_length > max_chars:
            chunk_text = "\n\n".join(current_units).strip()
            chunks.append(chunk_text)

            if overlap_chars > 0:
                overlap_units: List[str] = []
                overlap_length = 0
                for overlap_unit in reversed(current_units):
                    additional = len(overlap_unit) + (2 if overlap_units else 0)
                    if overlap_units and overlap_length + additional > overlap_chars:
                        break
                    overlap_units.insert(0, overlap_unit)
                    overlap_length += additional
                current_units = overlap_units
                current_length = len("\n\n".join(current_units)) if current_units else 0
                while current_units and current_length + 2 + len(unit) > max_chars:
                    current_units = current_units[1:]
                    current_length = len("\n\n".join(current_units)) if current_units else 0
            else:
                current_units = []
                current_length = 0
            continue

        current_units.append(unit)
        current_length = candidate_length
        index += 1

    if current_units:
        chunks.append("\n\n".join(current_units).strip())

    return chunks


def infer_section_heading(text: str, default_heading: str) -> str:
    first_unit = next((part.strip() for part in text.splitlines() if part.strip()), "")
    if not first_unit:
        return default_heading
    if len(first_unit) > 120 or len(first_unit.split()) > 16:
        return default_heading
    normalized = normalize_for_matching(first_unit)
    if normalized in {"authors", "contents", "introduction"}:
        return default_heading
    return first_unit.rstrip(":")


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    return _assemble_chunks(_paragraph_units(text, max_chars), max_chars)


def split_into_parent_child_chunks(
    text: str,
    default_heading: str,
    parent_max_chars: int,
    child_max_chars: int,
    child_overlap_chars: int,
) -> List[Dict[str, object]]:
    parent_units = _paragraph_units(text, parent_max_chars)
    parent_texts = _assemble_chunks(parent_units, parent_max_chars)
    parent_chunks: List[Dict[str, object]] = []

    for parent_text in parent_texts:
        child_units = _split_sentences(parent_text, child_max_chars)
        child_chunks = _assemble_chunks(child_units, child_max_chars, overlap_chars=child_overlap_chars)
        parent_chunks.append(
            {
                "heading": infer_section_heading(parent_text, default_heading),
                "text": parent_text,
                "children": child_chunks or [parent_text],
            }
        )

    return parent_chunks


def summarize_text(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentence.strip() for sentence in sentences[:max_sentences]).strip()
