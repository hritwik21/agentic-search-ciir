from __future__ import annotations

import json

from app.llm import GroqClient
from app.models import ColumnSpec, ExtractionBatch, TopicPlan


PLANNER_SYSTEM = """You are an information retrieval planner.
Create a compact, practical plan for discovering entities on the web.
Rules:
- Keep columns few and useful, usually 5 to 8 including entity.
- Prefer attributes that are likely to be found on public web pages.
- Search queries should include a broad query and a few targeted variants.
- Return only structured output.
"""

EXTRACTOR_SYSTEM = """You extract entities and their attributes from a single web page.
Rules:
- Only extract entities relevant to the topic.
- Only fill a cell if the page supports it.
- Every non-empty cell must include at least one source excerpt from this page.
- Prefer precise values over speculative summaries.
- Return only structured output.
"""


class Planner:
    def __init__(self, llm: GroqClient) -> None:
        self.llm = llm

    def build_plan(self, topic: str) -> TopicPlan:
        prompt = f"Topic: {topic}\nBuild the entity discovery plan."
        return self.llm.complete_json(PLANNER_SYSTEM, prompt, TopicPlan)


class EntityExtractor:
    def __init__(self, llm: GroqClient) -> None:
        self.llm = llm

    def extract_from_page(self, topic: str, columns: list[ColumnSpec], page_title: str, page_url: str, page_text: str) -> ExtractionBatch:
        compact_columns = [{"name": c.name, "description": c.description, "importance": c.importance} for c in columns]
        user = (
            f"Topic: {topic}\n"
            f"Columns: {json.dumps(compact_columns)}\n"
            f"Page title: {page_title}\n"
            f"Page url: {page_url}\n"
            f"Page text:\n{page_text}"
        )
        return self.llm.complete_json(EXTRACTOR_SYSTEM, user, ExtractionBatch)
