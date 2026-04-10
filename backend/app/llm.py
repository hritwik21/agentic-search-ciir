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

    def _normalize_query_plan(self, value: object, default_tier: str = "variant") -> dict | None:
        if isinstance(value, str):
            query = value.strip()
            if not query:
                return None
            return {
                "query": query,
                "reason": "",
                "tier": default_tier,
                "covered_constraints": [],
            }

        if not isinstance(value, dict):
            return None

        query = str(value.get("query") or "").strip()
        if not query:
            return None

        covered_constraints = value.get("covered_constraints", [])
        if not isinstance(covered_constraints, list):
            covered_constraints = []

        tier = str(value.get("tier") or default_tier).strip().lower()
        if tier not in {"anchor", "variant", "recall"}:
            tier = default_tier

        return {
            "query": query,
            "reason": str(value.get("reason") or "").strip(),
            "tier": tier,
            "covered_constraints": [
                str(item).strip()
                for item in covered_constraints
                if str(item).strip()
            ],
        }

    def _normalize_constraint(self, value: object, priority: str) -> dict | None:
        if not isinstance(value, dict):
            return None

        field = str(value.get("field") or value.get("name") or "").strip()
        raw_value = value.get("value")
        constraint_value = "" if raw_value is None else str(raw_value).strip()
        if not field or not constraint_value:
            return None

        aliases = value.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        operator = str(value.get("operator") or "eq").strip().lower()
        if operator not in {"eq", "neq", "gt", "gte", "lt", "lte", "contains", "in"}:
            operator = "eq"

        out_priority = str(value.get("priority") or priority).strip().lower()
        if out_priority not in {"hard", "soft"}:
            out_priority = priority

        return {
            "field": field,
            "operator": operator,
            "value": constraint_value,
            "normalized_value": str(value.get("normalized_value") or "").strip(),
            "priority": out_priority,
            "confidence": float(value.get("confidence", 1.0) or 0.0),
            "aliases": [str(alias).strip() for alias in aliases if str(alias).strip()],
        }

    def _normalize_topic_plan(self, data: dict) -> dict:
        """
        Repairs common LLM mistakes for TopicPlan.

        Accepts partial planner outputs and normalizes query/constraint objects.
        """
        if not isinstance(data, dict):
            return {
                "normalized_topic": "",
                "entity_type": "",
                "columns": [],
                "hard_constraints": [],
                "soft_constraints": [],
                "search_queries": [],
            }

        queries = data.get("search_queries") or data.get("queries") or []
        if not isinstance(queries, list):
            queries = []

        hard_constraints = data.get("hard_constraints") or []
        soft_constraints = data.get("soft_constraints") or []
        if not isinstance(hard_constraints, list):
            hard_constraints = []
        if not isinstance(soft_constraints, list):
            soft_constraints = []

        if not hard_constraints and not soft_constraints:
            combined_constraints = data.get("constraints", [])
            if isinstance(combined_constraints, list):
                for item in combined_constraints:
                    if not isinstance(item, dict):
                        continue
                    priority = str(item.get("priority") or "hard").strip().lower()
                    if priority == "soft":
                        soft_constraints.append(item)
                    else:
                        hard_constraints.append(item)

        normalized_queries = [
            normalized
            for item in queries
            if (normalized := self._normalize_query_plan(item, default_tier="variant")) is not None
        ]
        normalized_hard_constraints = [
            normalized
            for item in hard_constraints
            if (normalized := self._normalize_constraint(item, priority="hard")) is not None
        ]
        normalized_soft_constraints = [
            normalized
            for item in soft_constraints
            if (normalized := self._normalize_constraint(item, priority="soft")) is not None
        ]

        return {
            "normalized_topic": str(data.get("normalized_topic") or data.get("topic") or "").strip(),
            "entity_type": str(data.get("entity_type") or "").strip(),
            "columns": data.get("columns", []) or [],
            "hard_constraints": normalized_hard_constraints,
            "soft_constraints": normalized_soft_constraints,
            "search_queries": normalized_queries,
        }

    def _normalize_deeper_query_plan(self, data: dict) -> dict:
        """
        Repairs common LLM mistakes for DeeperQueryPlan.

        Accepts either:
        - {"search_queries": [...]}
        - ["query1", "query2"]
        """
        if isinstance(data, list):
            return {
                "search_queries": [
                    normalized
                    for item in data
                    if (normalized := self._normalize_query_plan(item, default_tier="anchor")) is not None
                ]
            }

        if isinstance(data, dict):
            queries = data.get("search_queries", [])
            if isinstance(queries, list):
                return {
                    "search_queries": [
                        normalized
                        for item in queries
                        if (normalized := self._normalize_query_plan(item, default_tier="anchor")) is not None
                    ]
                }

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
                    "hard_constraints": [
                        f"{constraint.field}:{constraint.operator}:{constraint.value}"
                        for constraint in getattr(result, "hard_constraints", [])
                    ],
                    "soft_constraints": [
                        f"{constraint.field}:{constraint.operator}:{constraint.value}"
                        for constraint in getattr(result, "soft_constraints", [])
                    ],
                    "search_queries": [
                        {
                            "query": query.query,
                            "tier": query.tier,
                            "covered_constraints": query.covered_constraints,
                        }
                        for query in getattr(result, "search_queries", [])
                    ],
                }
            )

        if schema_name == "DeeperQueryPlan":
            return self._preview(
                {
                    "search_queries": [
                        {
                            "query": query.query,
                            "tier": query.tier,
                            "covered_constraints": query.covered_constraints,
                        }
                        for query in getattr(result, "search_queries", [])
                    ]
                }
            )

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

    async def _create_json_completion(self, system_prompt: str, user_prompt: str) -> dict:
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
        return json.loads(content)

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        """
        Main entry point for all LLM JSON calls.

        Steps:
        1. limit concurrent LLM calls
        2. request JSON output
        3. parse returned JSON
        4. normalize common schema drift
        5. validate with Pydantic
        6. retry once with stricter instructions if JSON/schema fails
        """
        last_error: Exception | None = None

        for attempt in range(2):
            try:
                data = await self._create_json_completion(system_prompt, user_prompt)
            except json.JSONDecodeError as exc:
                last_error = ValueError("LLM returned invalid JSON")
                if attempt == 0:
                    user_prompt = (
                        user_prompt
                        + "\n\nIMPORTANT: Return ONLY valid JSON that matches the required schema exactly. "
                        "Do not include markdown, comments, or trailing text."
                    )
                    continue
                raise ValueError("LLM returned invalid JSON") from exc

            if schema.__name__ == "TopicPlan":
                data = self._normalize_topic_plan(data)
            elif schema.__name__ == "ExtractionBatch":
                data = self._normalize_extraction_batch(data)
            elif schema.__name__ == "DeeperQueryPlan":
                data = self._normalize_deeper_query_plan(data)

            try:
                validated = schema.model_validate(data)
                return validated
            except ValidationError as exc:
                last_error = exc
                if attempt == 0:
                    user_prompt = (
                        user_prompt
                        + "\n\nIMPORTANT: Return ONLY valid JSON that matches the required schema exactly. "
                        "Do not omit required keys. Do not flatten nested objects."
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise ValueError("LLM completion failed unexpectedly")