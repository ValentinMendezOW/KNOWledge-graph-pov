from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    root_dir: Path
    corpus_dir: Path
    data_dir: Path
    local_index_path: Path
    restricted_manifest_path: Path
    openai_api_key: str
    openai_chat_model: str
    openai_embedding_model: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    top_k: int
    parent_chunk_size: int
    child_chunk_size: int
    child_chunk_overlap: int
    require_graph: bool
    enable_admin_tools: bool
    openai_timeout_seconds: int


def _streamlit_secret(name: str):
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


def _setting(name: str, default):
    secret_value = _streamlit_secret(name)
    if secret_value not in (None, ""):
        return secret_value
    return os.getenv(name, default)


def _bool_setting(name: str, default: bool) -> bool:
    value = str(_setting(name, str(default).lower())).strip().lower()
    return value in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    load_dotenv()
    root_dir = repository_root()
    corpus_dir = Path(_setting("KG_CORPUS_DIR", root_dir / "papers")).expanduser()
    data_dir = root_dir / "data"
    local_index_path = Path(_setting("KG_LOCAL_INDEX_PATH", data_dir / "local_index.json")).expanduser()
    restricted_manifest_path = Path(_setting("KG_RESTRICTED_MANIFEST", root_dir / "config" / "restricted_documents.yaml")).expanduser()

    return Settings(
        root_dir=root_dir,
        corpus_dir=corpus_dir,
        data_dir=data_dir,
        local_index_path=local_index_path,
        restricted_manifest_path=restricted_manifest_path,
        openai_api_key=str(_setting("OPENAI_API_KEY", "")),
        openai_chat_model=str(_setting("OPENAI_CHAT_MODEL", "gpt-4.1-mini")),
        openai_embedding_model=str(_setting("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")),
        neo4j_uri=str(_setting("NEO4J_URI", "bolt://localhost:7687")),
        neo4j_username=str(_setting("NEO4J_USERNAME", "neo4j")),
        neo4j_password=str(_setting("NEO4J_PASSWORD", "")),
        neo4j_database=str(_setting("NEO4J_DATABASE", "neo4j")),
        top_k=int(_setting("KG_TOP_K", "6")),
        parent_chunk_size=int(_setting("KG_PARENT_CHUNK_SIZE", _setting("KG_CHUNK_SIZE", "1800"))),
        child_chunk_size=int(_setting("KG_CHILD_CHUNK_SIZE", "650")),
        child_chunk_overlap=int(_setting("KG_CHILD_CHUNK_OVERLAP", "140")),
        require_graph=_bool_setting("KG_REQUIRE_GRAPH", True),
        enable_admin_tools=_bool_setting("KG_ENABLE_ADMIN_TOOLS", False),
        openai_timeout_seconds=int(_setting("OPENAI_TIMEOUT_SECONDS", "30")),
    )
