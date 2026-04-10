from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class ColumnSpec(BaseModel):
    name: str
    description: str
    importance: Literal["high", "medium", "low"] = "medium"
    locked: bool = False

    @model_validator(mode="before")
    @classmethod
    def from_string(cls, value: object) -> object:
        if not isinstance(value, str):
            return value

        name = value.strip()
        if not name:
            return value

        if name == "entity":
            return {
                "name": "entity",
                "description": "Canonical entity name",
                "importance": "high",
                "locked": True,
            }

        return {
            "name": name,
            "description": f"{name.replace('_', ' ').replace('-', ' ')} for the entity",
            "importance": "medium",
        }


class ConstraintSpec(BaseModel):
    field: str
    operator: Literal["eq", "neq", "gt", "gte", "lt", "lte", "contains", "in"] = "eq"
    value: str
    normalized_value: str = ""
    priority: Literal["hard", "soft"] = "hard"
    confidence: float = 1.0
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_constraint(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        field = str(value.get("field") or value.get("name") or "").strip()
        raw_value = value.get("value")
        constraint_value = "" if raw_value is None else str(raw_value).strip()

        if not field or not constraint_value:
            return value

        operator = str(value.get("operator") or "eq").strip().lower()
        if operator not in {"eq", "neq", "gt", "gte", "lt", "lte", "contains", "in"}:
            operator = "eq"

        priority = str(value.get("priority") or "hard").strip().lower()
        if priority not in {"hard", "soft"}:
            priority = "hard"

        return {
            "field": field,
            "operator": operator,
            "value": constraint_value,
            "normalized_value": str(value.get("normalized_value") or "").strip(),
            "priority": priority,
            "confidence": float(value.get("confidence", 1.0) or 0.0),
            "aliases": [str(alias).strip() for alias in value.get("aliases", []) or [] if str(alias).strip()],
        }


class QueryPlan(BaseModel):
    query: str
    reason: str = ""
    tier: Literal["anchor", "variant", "recall"] = "variant"
    covered_constraints: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def from_string(cls, value: object) -> object:
        if isinstance(value, str):
            return {
                "query": value.strip(),
                "reason": "",
                "tier": "variant",
                "covered_constraints": [],
            }
        if isinstance(value, dict):
            tier = str(value.get("tier") or "variant").strip().lower()
            if tier not in {"anchor", "variant", "recall"}:
                tier = "variant"
            covered_constraints = value.get("covered_constraints", [])
            if not isinstance(covered_constraints, list):
                covered_constraints = []
            return {
                "query": str(value.get("query") or "").strip(),
                "reason": str(value.get("reason") or "").strip(),
                "tier": tier,
                "covered_constraints": [str(item).strip() for item in covered_constraints if str(item).strip()],
            }
        return value


class TopicPlan(BaseModel):
    normalized_topic: str
    entity_type: str
    columns: list[ColumnSpec]
    hard_constraints: list[ConstraintSpec] = Field(default_factory=list)
    soft_constraints: list[ConstraintSpec] = Field(default_factory=list)
    search_queries: list[QueryPlan]


class DeeperQueryPlan(BaseModel):
    search_queries: list[QueryPlan]


class SearchHit(BaseModel):
    """
    One normalized web search result before page fetch.
    """
    query: str
    title: str
    url: str
    snippet: str = ""
    rank: int = 0


class PageDocument(BaseModel):
    url: str
    title: str
    snippet: str = ""
    text: str = ""


class SourceRef(BaseModel):
    url: str
    title: str
    excerpt: str


class ExtractedCell(BaseModel):
    value: str | None = None
    confidence: float = 0.0
    sources: list[SourceRef] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    entity: str
    entity_type: str = "unknown"
    cells: dict[str, ExtractedCell] = Field(default_factory=dict)


class ExtractionBatch(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)


class FinalRow(BaseModel):
    entity_id: str
    entity: str
    cells: dict[str, ExtractedCell] = Field(default_factory=dict)


class FinalResult(BaseModel):
    topic: str
    entity_type: str
    columns: list[ColumnSpec]
    rows: list[FinalRow]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
