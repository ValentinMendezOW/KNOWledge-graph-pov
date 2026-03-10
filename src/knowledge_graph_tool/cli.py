from __future__ import annotations

from typing import Optional

import typer

from knowledge_graph_tool.config import load_settings
from knowledge_graph_tool.graph import load_index_to_neo4j
from knowledge_graph_tool.ingest import build_or_load_index
from knowledge_graph_tool.models import AccessPrincipal, SearchFilters
from knowledge_graph_tool.search import answer_question


app = typer.Typer(help="Knowledge graph pilot utilities.")


@app.command("build-index")
def build_index_command(rebuild: bool = typer.Option(True, help="Rebuild the local index.")) -> None:
    settings = load_settings()
    index = build_or_load_index(settings, rebuild=rebuild)
    typer.echo(
        "Indexed "
        f"{len(index.documents)} documents, "
        f"{len(index.parent_chunks)} parent chunks, "
        f"and {len(index.chunks)} child chunks at {settings.local_index_path}"
    )


@app.command("ask")
def ask_command(
    question: str,
    access_mode: str = typer.Option("all_access", help="all_access or limited_access"),
    allowed_document_ids: Optional[str] = typer.Option(
        "", help="Comma-separated list of allowed restricted document ids."
    ),
    organizations: Optional[str] = typer.Option("", help="Comma-separated organization filters."),
) -> None:
    settings = load_settings()
    index = build_or_load_index(settings, rebuild=False)
    principal = AccessPrincipal(
        mode=access_mode,
        allowed_document_ids=[value for value in allowed_document_ids.split(",") if value],
    )
    filters = SearchFilters(organizations=[value for value in organizations.split(",") if value])
    bundle = answer_question(question, principal, index, settings, filters=filters)
    typer.echo(bundle.answer)


@app.command("load-neo4j")
def load_neo4j_command(rebuild: bool = typer.Option(False, help="Rebuild the local index first.")) -> None:
    settings = load_settings()
    index = build_or_load_index(settings, rebuild=rebuild)
    load_index_to_neo4j(index, settings)
    typer.echo(
        "Loaded "
        f"{len(index.documents)} documents, "
        f"{len(index.parent_chunks)} parent chunks, "
        f"and {len(index.chunks)} child chunks into {settings.neo4j_uri}"
    )
