from pathlib import Path

from knowledge_graph_tool.parsers import infer_organization, split_into_chunks, split_into_parent_child_chunks


def test_split_into_chunks_caps_long_paragraphs():
    text = " ".join(["alpha"] * 400)
    chunks = split_into_chunks(text, max_chars=120)

    assert chunks
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_infer_organization_does_not_match_ey_inside_words():
    text = "They expect McKinsey research to shape the market."
    organization = infer_organization(text, Path("seizing-the-agentic-ai-advantage.pdf"))

    assert organization == "McKinsey"


def test_split_into_parent_child_chunks_creates_hierarchy():
    text = "\n\n".join(
        [
            "Heading Alpha",
            " ".join(["alpha"] * 120),
            "Heading Beta",
            " ".join(["beta"] * 120),
        ]
    )

    hierarchy = split_into_parent_child_chunks(
        text=text,
        default_heading="Document Title",
        parent_max_chars=500,
        child_max_chars=180,
        child_overlap_chars=40,
    )

    assert hierarchy
    assert all(item["children"] for item in hierarchy)
    assert all(len(child) <= 180 for item in hierarchy for child in item["children"])
