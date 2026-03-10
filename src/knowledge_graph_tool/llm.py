from __future__ import annotations

import json
from typing import Iterable, List

from openai import OpenAI

from knowledge_graph_tool.config import Settings
from knowledge_graph_tool.models import SearchHit
from knowledge_graph_tool.taxonomy import ORGANIZATION_ALIASES, canonicalize_organization


class OpenAIService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = (
            OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds)
            if settings.openai_api_key
            else None
        )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=[self._truncate_for_embedding(text) for text in texts],
        )
        return [item.embedding for item in response.data]

    def _create_chat_completion(
        self,
        *,
        messages: List[dict],
        temperature: float | None = None,
        max_output_tokens: int = 500,
        response_format: dict | None = None,
    ):
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        parameters = {
            "model": self.settings.openai_chat_model,
            "messages": messages,
            "max_tokens": max_output_tokens,
        }
        if temperature is not None:
            parameters["temperature"] = temperature
        if response_format is not None:
            parameters["response_format"] = response_format

        for _ in range(3):
            try:
                return self.client.chat.completions.create(**parameters)
            except Exception as error:
                message = str(error)
                updated = False
                if "max_tokens" in message and "unsupported" in message and "max_tokens" in parameters:
                    parameters["max_completion_tokens"] = parameters.pop("max_tokens")
                    updated = True
                if "temperature" in message and "unsupported" in message and "temperature" in parameters:
                    parameters.pop("temperature", None)
                    updated = True
                if not updated:
                    raise

    def _create_text_response(
        self,
        *,
        messages: List[dict],
        max_output_tokens: int = 900,
    ) -> str:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        if self.settings.openai_chat_model.startswith("gpt-5"):
            response = self.client.responses.create(
                model=self.settings.openai_chat_model,
                input=messages,
                max_output_tokens=max_output_tokens,
            )
            return response.output_text or ""

        response = self._create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_output_tokens=max_output_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _truncate_for_embedding(text: str, max_chars: int = 6000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def synthesize_answer(self, question: str, hits: List[SearchHit]) -> str:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        organizations = sorted({hit.document.organization for hit in hits})
        context_blocks = []
        for index, hit in enumerate(hits, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[{index}] Title: {hit.document.title}",
                        f"[{index}] Organization: {hit.document.organization}",
                        f"[{index}] Date: {hit.document.published_date or 'Unknown'}",
                        f"[{index}] Section: {(hit.parent_chunk.heading if hit.parent_chunk else hit.chunk.heading)}",
                        f"[{index}] Source file: {hit.document.file_name}",
                        f"[{index}] Excerpt: {hit.excerpt}",
                    ]
                )
            )

        answer = self._create_text_response(
            max_output_tokens=900,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the provided sources. "
                        "Cite every substantive claim with bracketed references like [1]. "
                        "If the sources are insufficient, say so clearly. "
                        "Before answering, inspect the Organization field in each source block. "
                        "Do not say an organization is missing if one or more sources list that organization. "
                        "Keep the answer concise and decision-useful."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Organizations present in the provided sources: {', '.join(organizations)}\n\n"
                        f"Sources:\n\n{chr(10).join(context_blocks)}"
                    ),
                },
            ],
        )
        if not answer.strip():
            raise RuntimeError("Model returned an empty synthesized answer.")
        return answer

    def infer_pdf_metadata(
        self,
        file_name: str,
        sample_text: str,
        current_title: str,
        current_organization: str,
        current_published_date: str | None,
    ) -> dict:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        organization_list = ", ".join(sorted(ORGANIZATION_ALIASES))
        response = self._create_chat_completion(
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract clean metadata from a consulting-industry PDF excerpt. "
                        "Return strict JSON with keys: title, organization, published_date. "
                        "Rules: title should be the clean human-readable report/article title only, with no authors, no dates, "
                        "no captions, no 'Getty Images', and no surrounding boilerplate. "
                        "organization must be one of: "
                        f"{organization_list}, or Unknown. "
                        "published_date should be ISO-like when visible: YYYY-MM-DD, YYYY-MM, YYYY, or null."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"File name: {file_name}\n"
                        f"Current title guess: {current_title}\n"
                        f"Current organization guess: {current_organization}\n"
                        f"Current published date guess: {current_published_date}\n\n"
                        f"Excerpt:\n{sample_text[:12000]}"
                    ),
                },
            ],
        )

        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        return {
            "title": payload.get("title") or current_title,
            "organization": canonicalize_organization(payload.get("organization")) or current_organization,
            "published_date": payload.get("published_date") or current_published_date,
        }
