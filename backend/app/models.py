from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class ColumnSpec(BaseModel):
    name: str
    description: str
    importance: Literal["high", "medium", "low"] = "medium"

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
            }

        return {
            "name": name,
            "description": f"{name.replace('_', ' ').replace('-', ' ')} for the entity",
            "importance": "medium",
        }


class QueryPlan(BaseModel):
    query: str
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def from_string(cls, value: object) -> object:
        if isinstance(value, str):
            return {"query": value.strip(), "reason": ""}
        return value


class TopicPlan(BaseModel):
    normalized_topic: str
    entity_type: str
    columns: list[ColumnSpec]
    search_queries: list[QueryPlan]


class DeeperQueryPlan(BaseModel):
    search_queries: list[str]


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
