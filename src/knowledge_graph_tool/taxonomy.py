from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


ORGANIZATION_ALIASES: Dict[str, List[str]] = {
    "Accenture": ["accenture"],
    "BCG": ["bcg", "boston consulting group"],
    "Bain": ["bain", "bain & company"],
    "Capgemini": ["capgemini"],
    "Deloitte": ["deloitte"],
    "EY": ["ey", "ernst & young"],
    "KPMG": ["kpmg"],
    "McKinsey": ["mckinsey", "mckinsey & company", "mckinsey global institute", "quantumblack", "mck"],
    "Oliver Wyman": ["oliver wyman", "oliverwyman", "ow"],
    "PwC": ["pwc", "pricewaterhousecoopers", "price waterhouse coopers"],
}


PRACTICE_KEYWORDS: Dict[str, List[str]] = {
    "BFS": [
        "bank",
        "banking",
        "financial services",
        "financial institution",
        "credit",
        "lending",
        "compliance",
        "risk assessment",
    ],
    "PT": [
        "transformation",
        "operating model",
        "productivity",
        "efficiency",
        "performance",
        "cost",
        "workforce",
        "process redesign",
    ],
    "Quotient": [
        "artificial intelligence",
        "ai",
        "genai",
        "generative ai",
        "agentic",
        "llm",
        "machine learning",
        "automation",
    ],
}

INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "Banking and Financial Services": [
        "bank",
        "banking",
        "financial services",
        "financial institution",
        "credit",
        "lending",
        "payments",
        "insurance",
    ],
    "Retail": ["retail", "consumer", "stores", "shopping"],
    "Public Sector": ["public sector", "government", "federal", "regulation"],
    "Healthcare and Life Sciences": [
        "healthcare",
        "pharmaceutical",
        "pharma",
        "clinical",
        "life sciences",
    ],
    "Technology": ["software", "developer", "digital", "technology", "it services"],
}

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Generative AI": ["generative ai", "genai", "foundation model", "llm"],
    "Agentic AI": ["agentic", "digital agent", "autonomous agent"],
    "Governance": ["governance", "responsible ai", "risk", "compliance", "guardrail"],
    "Workforce Transformation": ["workforce", "reskilling", "skills", "change management"],
    "Productivity": ["productivity", "efficiency", "automation", "cost"],
}


def extract_matches(text: str, mapping: Dict[str, Iterable[str]]) -> List[str]:
    lower_text = text.lower()
    matches = []
    for label, keywords in mapping.items():
        if any(keyword in lower_text for keyword in keywords):
            matches.append(label)
    return matches


def normalize_for_matching(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def canonicalize_organization(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    normalized = normalize_for_matching(value)
    for canonical, patterns in ORGANIZATION_ALIASES.items():
        normalized_patterns = [normalize_for_matching(pattern) for pattern in patterns] + [
            normalize_for_matching(canonical)
        ]
        if normalized in normalized_patterns:
            return canonical
    return None
