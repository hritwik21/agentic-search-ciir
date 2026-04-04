from __future__ import annotations

import asyncio
import json
from typing import TypeVar

from groq import Groq
from pydantic import BaseModel, ValidationError

from app.config import settings

T = TypeVar("T", bound=BaseModel)


class GroqClient:
    """
    Small wrapper around Groq chat completions.

    Responsibilities:
    - make LLM calls
    - enforce JSON-only responses
    - normalize common schema drift
    - validate final output with Pydantic
    """

    def __init__(self) -> None:
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
        self.sem = asyncio.Semaphore(settings.LLM_CONCURRENCY)

    def _normalize_extraction_batch(self, data: dict) -> dict:
        """
        Repairs common LLM mistakes for ExtractionBatch.

        Examples fixed:
        - missing entity_type
        - flattened attributes instead of nested cells
        - null cells instead of proper cell objects
        """
        entities = data.get("entities")
        if not isinstance(entities, list):
            return {"entities": []}

        normalized_entities = []

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            entity_name = str(entity.get("entity", "")).strip()
            if not entity_name:
                continue

            entity_type = str(entity.get("entity_type", "unknown")).strip() or "unknown"

            if "cells" in entity and isinstance(entity["cells"], dict):
                normalized_cells = {}
                for key, value in entity["cells"].items():
                    if value is None:
                        normalized_cells[key] = {
                            "value": None,
                            "confidence": 0.0,
                            "sources": [],
                        }
                    elif isinstance(value, dict):
                        normalized_cells[key] = {
                            "value": value.get("value"),
                            "confidence": float(value.get("confidence", 0.0) or 0.0),
                            "sources": value.get("sources", []) or [],
                        }
                normalized_entities.append(
                    {
                        "entity": entity_name,
                        "entity_type": entity_type,
                        "cells": normalized_cells,
                    }
                )
                continue

            reserved = {"entity", "entity_type", "cells"}
            cells = {}
            for key, value in entity.items():
                if key in reserved:
                    continue
                cells[key] = {
                    "value": None if value is None else str(value),
                    "confidence": 0.5 if value not in (None, "") else 0.0,
                    "sources": [],
                }

            normalized_entities.append(
                {
                    "entity": entity_name,
                    "entity_type": entity_type,
                    "cells": cells,
                }
            )

        return {"entities": normalized_entities}

    def _normalize_deeper_query_plan(self, data: dict) -> dict:
        """
        Repairs common LLM mistakes for DeeperQueryPlan.

        Accepts either:
        - {"search_queries": [...]}
        - ["query1", "query2"]
        """
        if isinstance(data, list):
            return {"search_queries": [str(x) for x in data if str(x).strip()]}

        if isinstance(data, dict):
            queries = data.get("search_queries", [])
            if isinstance(queries, list):
                return {"search_queries": [str(x) for x in queries if str(x).strip()]}

        return {"search_queries": []}

    def _preview(self, value: object, limit: int = 600) -> str:
        text = json.dumps(value, ensure_ascii=True, default=str)
        return text if len(text) <= limit else f"{text[:limit]}..."

    def _summarize_validated(self, result: BaseModel) -> str:
        schema_name = result.__class__.__name__

        if schema_name == "TopicPlan":
            return self._preview(
                {
                    "entity_type": getattr(result, "entity_type", ""),
                    "columns": [column.name for column in getattr(result, "columns", [])],
                    "search_queries": [query.query for query in getattr(result, "search_queries", [])],
                }
            )

        if schema_name == "DeeperQueryPlan":
            return self._preview({"search_queries": getattr(result, "search_queries", [])})

        if schema_name == "ExtractionBatch":
            entities = []
            for entity in getattr(result, "entities", [])[:5]:
                entities.append(
                    {
                        "entity": entity.entity,
                        "entity_type": entity.entity_type,
                        "cells": sorted(entity.cells.keys()),
                    }
                )
            return self._preview({"count": len(getattr(result, "entities", [])), "entities": entities})

        return self._preview(result.model_dump())

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        """
        Main entry point for all LLM JSON calls.

        Steps:
        1. limit concurrent LLM calls
        2. request JSON output
        3. parse returned JSON
        4. normalize common schema drift
        5. validate with Pydantic
        """
        async with self.sem:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        content = response.choices[0].message.content or "{}"

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned invalid JSON") from exc

        if schema.__name__ == "ExtractionBatch":
            data = self._normalize_extraction_batch(data)
        elif schema.__name__ == "DeeperQueryPlan":
            data = self._normalize_deeper_query_plan(data)

        try:
            validated = schema.model_validate(data)
        except ValidationError:
            raise
        return validated
