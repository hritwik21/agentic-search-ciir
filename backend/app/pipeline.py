from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher

from app.config import settings
from app.console import demo_print
from app.fetch import PageFetcher
from app.llm import GroqClient
from app.models import (
    ColumnSpec,
    DeeperQueryPlan,
    ExtractedCell,
    ExtractionBatch,
    FinalResult,
    FinalRow,
    PageDocument,
    SearchHit,
    SourceRef,
    TopicPlan,
)
from app.prompts import (
    DEEPER_SEARCH_SYSTEM_PROMPT,
    EXTRACTOR_SYSTEM_PROMPT,
    FINAL_STANDARDIZATION_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from app.search import SearchClient

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(value: str) -> str:
    """
    Normalizes entity names for merging.
    """
    return _NON_ALNUM_RE.sub("-", value.lower()).strip("-")


def normalize_cell_value(value: str | None) -> str | None:
    """
    Cleans extracted values into a compact display form.
    """
    if value is None:
        return None

    cleaned = value.replace("\u00a0", " ").replace("\u200b", " ")
    cleaned = re.sub(r"\s*\n+\s*", "; ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,")
    if cleaned.lower() in {"", "n/a", "na", "none", "null", "unknown", "not available"}:
        return None
    return cleaned


def names_match(left: str, right: str) -> bool:
    """
    Treats small naming variations as the same entity.
    """
    if left == right:
        return True
    if left in right or right in left:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.88


def compact_columns(columns: list[ColumnSpec]) -> list[dict]:
    """
    Converts ColumnSpec objects into a simple JSON-friendly structure
    for prompting the LLM.
    """
    return [
        {
            "name": c.name,
            "description": c.description,
            "importance": c.importance,
        }
        for c in columns
    ]


def tokenize(text: str) -> set[str]:
    """
    Small lexical tokenizer used for lightweight ranking heuristics.
    """
    return set(x for x in _NON_ALNUM_RE.split(text.lower()) if x)


def summarize_hits(hits: list[SearchHit], limit: int = 5) -> list[dict[str, str | int]]:
    return [
        {
            "rank": hit.rank,
            "title": hit.title,
            "url": hit.url,
        }
        for hit in hits[:limit]
    ]


def summarize_pages(pages: list[PageDocument], limit: int = 5) -> list[dict[str, str | int]]:
    return [
        {
            "title": page.title,
            "url": page.url,
            "chars": len(page.text),
        }
        for page in pages[:limit]
    ]


def summarize_batch(batch: ExtractionBatch) -> list[dict[str, object]]:
    return [
        {
            "entity": entity.entity,
            "entity_type": entity.entity_type,
            "cells": sorted(entity.cells.keys()),
        }
        for entity in batch.entities[:5]
    ]


def summarize_rows(rows: list[FinalRow], limit: int = 5) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for row in rows[:limit]:
        filled_cells = {column_name: cell.value for column_name, cell in row.cells.items() if cell.value}
        summary.append(
            {
                "entity": row.entity,
                "filled_columns": sorted(filled_cells.keys()),
                "sample_values": dict(list(filled_cells.items())[:3]),
            }
        )
    return summary


class AgenticSearchPipeline:
    """
    End-to-end orchestration for:
    planning -> search -> fetch -> rerank -> extract -> merge -> deeper search
    """

    def __init__(self) -> None:
        self.llm = GroqClient()
        self.search_client = SearchClient()
        self.fetcher = PageFetcher()

    async def build_topic_plan(self, topic: str) -> TopicPlan:
        """
        Uses the planner LLM to decide:
        - entity type
        - schema columns
        - initial search queries
        """
        user_prompt = f"""
Topic: {topic}

Return JSON with:
- normalized_topic
- entity_type
- columns
- search_queries

Constraints:
- include "entity" in columns
- produce 4 to 6 search queries
- keep columns practical for public web extraction
"""
        plan = await self.llm.complete_json(PLANNER_SYSTEM_PROMPT, user_prompt, TopicPlan)

        if not any(c.name == "entity" for c in plan.columns):
            plan.columns = [ColumnSpec(name="entity", description="Canonical entity name", importance="high")] + plan.columns

        plan.search_queries = plan.search_queries[: settings.MAX_BASE_QUERIES]
        demo_print(
            "Layer 1 - Topic Plan",
            {
                "topic": topic,
                "entity_type": plan.entity_type,
                "columns": [column.name for column in plan.columns],
                "queries": [query.query for query in plan.search_queries],
            },
        )
        return plan

    def rank_hits(self, topic: str, hits: list[SearchHit]) -> list[SearchHit]:
        """
        Lightweight pre-ranking before page fetch/extraction.

        This is important for latency because it helps us avoid sending
        too many weak pages into the LLM.
        """
        topic_tokens = tokenize(topic)

        def score(hit: SearchHit) -> float:
            title_tokens = tokenize(hit.title)
            snippet_tokens = tokenize(hit.snippet)
            overlap = len(topic_tokens & (title_tokens | snippet_tokens))

            domain_bonus = 0.0
            url_lower = hit.url.lower()
            if any(x in url_lower for x in ["yelp", "tripadvisor", "doordash", "ubereats", "grubhub"]):
                domain_bonus += 1.5
            if any(x in url_lower for x in ["facebook", "instagram", "pinterest"]):
                domain_bonus -= 0.5

            rank_bonus = 1.0 / math.sqrt(max(hit.rank, 1))
            return overlap + domain_bonus + rank_bonus

        return sorted(hits, key=score, reverse=True)

    async def search_and_fetch(self, queries: list[str]) -> tuple[list[SearchHit], list[PageDocument]]:
        """
        Runs all broad queries, deduplicates hits, ranks them,
        then fetches only the stronger subset.
        """
        all_hits: dict[str, SearchHit] = {}

        for query in queries:
            hits = self.search_client.search(query, settings.SEARCH_RESULTS_PER_QUERY)
            for hit in hits:
                all_hits.setdefault(hit.url, hit)

        ranked_hits = self.rank_hits(" ".join(queries), list(all_hits.values()))
        ranked_hits = ranked_hits[: settings.MAX_PAGES_FOR_EXTRACTION]
        demo_print(
            "Layer 2 - Ranked Search Hits",
            {
                "queries": queries,
                "top_hits": summarize_hits(ranked_hits),
            },
        )

        pages = await self.fetcher.fetch_many(ranked_hits)
        demo_print("Layer 3 - Fetched Pages", {"pages": summarize_pages(pages)})
        return ranked_hits, pages

    async def extract_from_page(
        self,
        topic: str,
        entity_type: str,
        columns: list[ColumnSpec],
        page: PageDocument,
    ) -> ExtractionBatch:
        """
        Runs the extractor LLM on one page.
        """
        allowed_columns = [c.name for c in columns if c.name != "entity"]

        user_prompt = f"""
Topic: {topic}
Expected entity_type: {entity_type}
Allowed output columns: {json.dumps(allowed_columns)}
Full column schema: {json.dumps(compact_columns(columns))}
Page title: {page.title}
Page url: {page.url}

Important:
- Every entity must include entity, entity_type, and cells.
- cells must be keyed only by allowed output columns.
- Do not place fields at the top level.
- Use only evidence from this page.
- If unsupported, leave the cell out.

Page text:
{page.text}
"""
        batch = await self.llm.complete_json(EXTRACTOR_SYSTEM_PROMPT, user_prompt, ExtractionBatch)

        for entity in batch.entities:
            if entity.entity_type == "unknown":
                entity.entity_type = entity_type

        demo_print(
            "Layer 4 - Page Extraction",
            {
                "page_title": page.title,
                "page_url": page.url,
                "entities": summarize_batch(batch),
            },
        )

        return batch

    def merge_cell(self, current: ExtractedCell | None, incoming: ExtractedCell) -> ExtractedCell:
        """
        Merges two candidate values for the same cell.

        Preference order:
        - non-empty over empty
        - higher confidence
        - more sources as tie-breaker
        """
        if current is not None:
            current.value = normalize_cell_value(current.value)
        incoming.value = normalize_cell_value(incoming.value)

        if current is None:
            return incoming

        if not incoming.value:
            return current
        if not current.value:
            return incoming

        all_sources = current.sources + incoming.sources
        deduped_sources: dict[tuple[str, str], SourceRef] = {}
        for src in all_sources:
            deduped_sources[(src.url, src.excerpt)] = src

        best = current
        if incoming.confidence > current.confidence:
            best = incoming
        elif incoming.confidence == current.confidence and len(incoming.sources) > len(current.sources):
            best = incoming

        best.sources = list(deduped_sources.values())[:5]
        return best

    def prune_sparse_columns(self, columns: list[ColumnSpec], rows: list[FinalRow]) -> list[ColumnSpec]:
        """
        Drops columns that remain too sparse across final rows.
        """
        if not rows:
            return columns

        kept_columns = [column for column in columns if column.name == "entity"]
        coverage_counts: list[tuple[int, ColumnSpec]] = []
        minimum_filled = max(2, math.ceil(len(rows) * 0.5))

        for column in columns:
            if column.name == "entity":
                continue

            filled = sum(1 for row in rows if row.cells.get(column.name) and row.cells[column.name].value)
            coverage_counts.append((filled, column))
            if filled >= minimum_filled:
                kept_columns.append(column)

        if len(kept_columns) > 1:
            return kept_columns

        fallback_columns = [column for filled, column in sorted(coverage_counts, reverse=True) if filled > 0][:3]
        return kept_columns + fallback_columns

    def compact_rows_for_output(self, rows: list[FinalRow], columns: list[ColumnSpec]) -> list[FinalRow]:
        """
        Removes empty cells and cells for dropped columns from the final response.
        """
        allowed_columns = {column.name for column in columns if column.name != "entity"}
        compacted_rows: list[FinalRow] = []

        for row in rows:
            compacted_cells: dict[str, ExtractedCell] = {}
            for column_name, cell in row.cells.items():
                if column_name not in allowed_columns:
                    continue

                cell.value = normalize_cell_value(cell.value)
                if not cell.value:
                    continue
                compacted_cells[column_name] = cell

            row.cells = compacted_cells
            compacted_rows.append(row)

        return compacted_rows

    def merge_batches(self, batches: list[ExtractionBatch]) -> dict[str, FinalRow]:
        """
        Merges all extracted entities from many pages into canonical rows.
        """
        merged: dict[str, FinalRow] = {}

        for batch in batches:
            for entity in batch.entities:
                entity_name = entity.entity.strip()
                if not entity_name:
                    continue

                normalized_entity_name = normalize_name(entity_name)
                entity_key = f"{entity.entity_type}|{normalized_entity_name}"
                if entity_key not in merged:
                    for existing_key, existing_row in merged.items():
                        existing_type, _, existing_name = existing_key.partition("|")
                        if existing_type != entity.entity_type:
                            continue
                        if names_match(normalized_entity_name, existing_name):
                            entity_key = existing_key
                            break

                if entity_key not in merged:
                    merged[entity_key] = FinalRow(
                        entity_id=entity_key,
                        entity=entity_name,
                        cells={},
                    )

                row = merged[entity_key]
                for column_name, cell in entity.cells.items():
                    row.cells[column_name] = self.merge_cell(row.cells.get(column_name), cell)

        return merged

    def merge_rows_into(self, base: dict[str, FinalRow], incoming: dict[str, FinalRow]) -> dict[str, FinalRow]:
        """
        Merges a second row set into an existing row set.
        Used after deeper search.
        """
        for key, incoming_row in incoming.items():
            if key not in base:
                base[key] = incoming_row
                continue

            base_row = base[key]
            for column_name, cell in incoming_row.cells.items():
                base_row.cells[column_name] = self.merge_cell(base_row.cells.get(column_name), cell)

        return base

    def get_missing_columns(self, rows: dict[str, FinalRow], columns: list[ColumnSpec]) -> list[dict]:
        """
        Finds which important fields are still missing for which entities.
        """
        missing = []

        important_columns = [c for c in columns if c.name != "entity" and c.importance in {"high", "medium"}]

        for row in rows.values():
            missing_columns = []
            for col in important_columns:
                cell = row.cells.get(col.name)
                if not cell or not cell.value:
                    missing_columns.append(col.name)

            if missing_columns:
                missing.append({
                    "entity": row.entity,
                    "missing_columns": missing_columns,
                })

        return missing

    async def build_deeper_queries(
        self,
        topic: str,
        rows: dict[str, FinalRow],
        columns: list[ColumnSpec],
    ) -> list[str]:
        """
        Uses the LLM to generate targeted follow-up queries for missing fields.
        """
        missing = self.get_missing_columns(rows, columns)
        if not missing:
            return []

        user_prompt = f"""
Topic: {topic}

Missing coverage summary:
{json.dumps(missing[:5], indent=2)}

Return only:
{{
  "search_queries": ["query 1", "query 2"]
}}
"""
        plan = await self.llm.complete_json(
            DEEPER_SEARCH_SYSTEM_PROMPT,
            user_prompt,
            DeeperQueryPlan,
        )

        deduped = list(dict.fromkeys(plan.search_queries))
        demo_print(
            "Layer 6 - Deeper Query Plan",
            {
                "topic": topic,
                "missing": missing[:5],
                "queries": deduped[: settings.MAX_DEEPER_QUERIES],
            },
        )
        return deduped[: settings.MAX_DEEPER_QUERIES]

    def rank_final_rows(self, rows: dict[str, FinalRow], entity_type: str, topic: str) -> list[FinalRow]:
        """
        Final ranking for output rows.

        Current ranking signals:
        - more filled cells
        - higher confidence
        - mild topical match on entity name
        """
        topic_tokens = tokenize(topic)

        def score(row: FinalRow) -> tuple[float, float, str]:
            filled = sum(1 for c in row.cells.values() if c.value)
            confidence = sum(c.confidence for c in row.cells.values())
            topical = len(topic_tokens & tokenize(row.entity))
            return (filled + topical, confidence, row.entity.lower())

        return sorted(rows.values(), key=score, reverse=True)[: settings.MAX_FINAL_ROWS]

    async def standardize_final_values(self, result: FinalResult) -> FinalResult:
        user_prompt = f"""
Standardize the following final result JSON without changing its meaning:

{json.dumps(result.model_dump(), ensure_ascii=False)}
"""
        try:
            standardized = await self.llm.complete_json(
                FINAL_STANDARDIZATION_SYSTEM_PROMPT,
                user_prompt,
                FinalResult,
            )
        except Exception:
            demo_print("Layer 9 - Final Standardization", {"status": "failed", "fallback": "original values"})
            return result

        standardized_rows = {row.entity_id: row for row in standardized.rows}
        for row in result.rows:
            updated_row = standardized_rows.get(row.entity_id)
            if not updated_row:
                continue
            for column_name, cell in row.cells.items():
                updated_cell = updated_row.cells.get(column_name)
                if updated_cell is None:
                    continue
                cell.value = normalize_cell_value(updated_cell.value)

        demo_print("Layer 9 - Final Standardization", {"status": "applied"})
        return result

    async def run(self, topic: str) -> FinalResult:
        """
        Main orchestrator called by the API.

        Flow:
        1. plan
        2. search
        3. fetch
        4. extract from initial pages
        5. merge
        6. deeper search if needed
        7. rank final rows
        8. return result
        """
        plan = await self.build_topic_plan(topic)

        initial_queries = [q.query for q in plan.search_queries]
        initial_hits, initial_pages = await self.search_and_fetch(initial_queries)
        demo_print(
            "Layer 2/3 Summary",
            {
                "initial_hits": summarize_hits(initial_hits),
                "initial_pages": summarize_pages(initial_pages),
            },
        )

        initial_batches: list[ExtractionBatch] = []
        for page in initial_pages:
            try:
                batch = await self.extract_from_page(topic, plan.entity_type, plan.columns, page)
                initial_batches.append(batch)
            except Exception:
                continue

        merged_rows = self.merge_batches(initial_batches)
        demo_print(
            "Layer 5 - Initial Merged Rows",
            {
                "count": len(merged_rows),
                "rows": summarize_rows(list(merged_rows.values())),
            },
        )

        deeper_queries_used: list[str] = []
        deeper_pages_count = 0

        for _ in range(settings.DEEPER_SEARCH_ROUNDS):
            deeper_queries = await self.build_deeper_queries(topic, merged_rows, plan.columns)
            deeper_queries = [q for q in deeper_queries if q not in deeper_queries_used]
            if not deeper_queries:
                demo_print("Layer 6 - Deeper Search", {"status": "skipped", "reason": "no_new_queries"})
                break

            deeper_queries_used.extend(deeper_queries)

            deeper_hits_map: dict[str, SearchHit] = {}
            for query in deeper_queries:
                hits = self.search_client.search(query, 3)
                for hit in hits:
                    deeper_hits_map.setdefault(hit.url, hit)

            deeper_hits = self.rank_hits(topic, list(deeper_hits_map.values()))[: settings.MAX_PAGES_FOR_EXTRACTION]
            deeper_pages = await self.fetcher.fetch_many(deeper_hits)
            deeper_pages_count += len(deeper_pages)
            demo_print(
                "Layer 7 - Deeper Retrieval",
                {
                    "queries": deeper_queries,
                    "hits": summarize_hits(deeper_hits),
                    "pages": summarize_pages(deeper_pages),
                },
            )

            deeper_batches: list[ExtractionBatch] = []
            for page in deeper_pages:
                try:
                    batch = await self.extract_from_page(topic, plan.entity_type, plan.columns, page)
                    deeper_batches.append(batch)
                except Exception:
                    continue

            merged_rows = self.merge_rows_into(merged_rows, self.merge_batches(deeper_batches))
            demo_print(
                "Layer 8 - Deeper Merged Rows",
                {
                    "count": len(merged_rows),
                    "rows": summarize_rows(list(merged_rows.values())),
                },
            )

        final_rows = self.rank_final_rows(merged_rows, plan.entity_type, topic)
        final_columns = self.prune_sparse_columns(plan.columns, final_rows)
        final_rows = self.compact_rows_for_output(final_rows, final_columns)
        final_rows = [row for row in final_rows if len(row.cells) >= 2]
        diagnostics = {
            "initial_queries": initial_queries,
            "deeper_queries": deeper_queries_used,
            "initial_hits": len(initial_hits),
            "initial_pages": len(initial_pages),
            "deeper_pages": deeper_pages_count,
            "entities_found": len(merged_rows),
        }
        final_result = await self.standardize_final_values(
            FinalResult(
                topic=topic,
                entity_type=plan.entity_type,
                columns=final_columns,
                rows=final_rows,
                diagnostics=diagnostics,
            )
        )
        demo_print(
            "Layer 10 - Final Output",
            {
                "rows": summarize_rows(final_result.rows),
                "diagnostics": diagnostics,
            },
        )

        return final_result
