from __future__ import annotations

import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher

from app.config import settings
from app.console import demo_print
from app.fetch import PageFetcher
from app.llm import GroqClient
from app.models import (
    ColumnSpec,
    ConstraintSpec,
    DeeperQueryPlan,
    ExtractedCell,
    ExtractionBatch,
    FinalResult,
    FinalRow,
    PageDocument,
    QueryPlan,
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
_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(
    r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>k|m|b|thousand|million|billion)?\+?",
    re.IGNORECASE,
)
_NEGATION_HINTS = {"not", "without", "exclude", "excluding", "except", "non"}


def normalize_name(value: str) -> str:
    return _NON_ALNUM_RE.sub("-", value.lower()).strip("-")


def normalize_identifier(value: str) -> str:
    return _NON_ALNUM_RE.sub("_", value.lower()).strip("_")


def normalize_match_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value.lower()).strip()


def normalize_cell_value(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.replace("\u00a0", " ").replace("\u200b", " ")
    cleaned = re.sub(r"\s*\n+\s*", "; ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,")
    if cleaned.lower() in {"", "n/a", "na", "none", "null", "unknown", "not available"}:
        return None
    return cleaned


def names_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if left in right or right in left:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.88


def compact_columns(columns: list[ColumnSpec]) -> list[dict]:
    return [
        {
            "name": c.name,
            "description": c.description,
            "importance": c.importance,
            "locked": c.locked,
        }
        for c in columns
    ]


def tokenize(text: str) -> list[str]:
    return [x for x in _NON_ALNUM_RE.split(text.lower()) if x]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def parse_numeric_value(value: str | None) -> float | None:
    if value is None:
        return None

    text = value.strip().lower().replace(",", "")
    if not text:
        return None

    text = text.replace("usd", "").strip()
    text = text.lstrip("$€£").strip()
    text = text.replace(">=", "").replace("<=", "").replace(">", "").replace("<", "").strip()
    text = text.rstrip("+").strip()

    multiplier = 1.0
    for suffix, scale in (
        ("billion", 1_000_000_000.0),
        ("million", 1_000_000.0),
        ("thousand", 1_000.0),
        ("b", 1_000_000_000.0),
        ("m", 1_000_000.0),
        ("k", 1_000.0),
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            multiplier = scale
            break

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group(0)) * multiplier
    except ValueError:
        return None


def extract_numeric_mentions(text: str) -> list[float]:
    mentions: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        number_text = match.group("number").replace(",", "")
        suffix = (match.group("suffix") or "").lower()
        multiplier = {
            "k": 1_000.0,
            "thousand": 1_000.0,
            "m": 1_000_000.0,
            "million": 1_000_000.0,
            "b": 1_000_000_000.0,
            "billion": 1_000_000_000.0,
        }.get(suffix, 1.0)
        try:
            mentions.append(float(number_text) * multiplier)
        except ValueError:
            continue
    return mentions


def compare_numeric_value(candidate: float, operator: str, target: float) -> bool:
    if operator == "gt":
        return candidate > target
    if operator == "gte":
        return candidate >= target
    if operator == "lt":
        return candidate < target
    if operator == "lte":
        return candidate <= target
    if operator == "neq":
        return not math.isclose(candidate, target)
    return math.isclose(candidate, target)


def constraint_value_phrases(constraint: ConstraintSpec) -> list[str]:
    candidates = [constraint.value, constraint.normalized_value, *constraint.aliases]
    phrases: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        phrase = normalize_match_text(candidate)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)

    return phrases


def constraint_context_tokens(constraint: ConstraintSpec) -> set[str]:
    context_sources = [constraint.field.replace("_", " ")]
    context_sources.extend(alias for alias in constraint.aliases if re.search(r"[a-zA-Z]", alias))
    tokens: set[str] = set()
    for source in context_sources:
        tokens.update(tokenize(source))
    return tokens


def phrase_matches(haystack: str, haystack_tokens: set[str], phrase: str) -> bool:
    normalized_phrase = normalize_match_text(phrase)
    if not normalized_phrase:
        return False

    phrase_tokens = token_set(normalized_phrase)
    if not phrase_tokens:
        return False

    if len(phrase_tokens) == 1:
        token = next(iter(phrase_tokens))
        if len(token) <= 3:
            return token in haystack_tokens

    return normalized_phrase in haystack


def score_constraint_text(constraint: ConstraintSpec, text: str) -> float:
    haystack = normalize_match_text(text)
    if not haystack:
        return 0.0

    haystack_tokens = token_set(haystack)
    value_phrases = constraint_value_phrases(constraint)
    value_tokens = token_set(" ".join(value_phrases))
    context_tokens = constraint_context_tokens(constraint)

    phrase_hits = sum(1 for phrase in value_phrases if phrase_matches(haystack, haystack_tokens, phrase))
    token_coverage = 0.0
    if value_tokens:
        token_coverage = len(haystack_tokens & value_tokens) / len(value_tokens)

    score = 0.0
    if phrase_hits:
        score += 1.25 + (0.25 * (phrase_hits - 1))
    elif token_coverage >= 0.6:
        score += 0.75

    if constraint.operator in {"gt", "gte", "lt", "lte", "eq", "neq"}:
        target_value = parse_numeric_value(constraint.normalized_value or constraint.value)
        has_numeric_target = target_value is not None
        context_present = not context_tokens or bool(haystack_tokens & context_tokens)
        numeric_mentions = extract_numeric_mentions(haystack)

        if has_numeric_target and context_present and numeric_mentions:
            if any(compare_numeric_value(candidate, constraint.operator, target_value) for candidate in numeric_mentions):
                score += 1.75
            elif constraint.operator in {"gt", "gte", "lt", "lte"}:
                score -= 0.75 if constraint.priority == "hard" else 0.25

    if constraint.operator == "neq":
        if any(phrase_matches(haystack, haystack_tokens, phrase) for phrase in value_phrases):
            score -= 1.0

    return score


def summarize_hits(hits: list[SearchHit], limit: int = 5) -> list[dict[str, str | int]]:
    return [
        {
            "rank": hit.rank,
            "title": hit.title,
            "url": hit.url,
            "query": hit.query,
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


class BM25Scorer:
    """
    Lightweight BM25 over title + snippet documents.
    Pure Python so you can copy-paste and run without extra dependencies.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_count = len(corpus_tokens)
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.doc_count, 1)

        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[Counter[str]] = []
        for doc in corpus_tokens:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        if not query_tokens or doc_index >= self.doc_count:
            return 0.0

        tf = self.term_freqs[doc_index]
        dl = self.doc_lengths[doc_index]
        score = 0.0

        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self.idf(term)
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += idf * ((freq * (self.k1 + 1)) / max(denom, 1e-9))
        return score


class AgenticSearchPipeline:
    """
    End-to-end orchestration for:
    planning -> search -> fetch -> rerank -> extract -> merge -> deeper search
    """

    def __init__(self) -> None:
        self.llm = GroqClient()
        self.search_client = SearchClient()
        self.fetcher = PageFetcher()

    def dedupe_constraints(self, constraints: list[ConstraintSpec], priority: str) -> list[ConstraintSpec]:
        deduped: dict[tuple[str, str, str], ConstraintSpec] = {}

        for constraint in constraints:
            field = normalize_identifier(constraint.field)
            value = (constraint.normalized_value or constraint.value).strip()
            if not field or not value:
                continue

            normalized_constraint = constraint.model_copy(
                update={
                    "field": field,
                    "priority": priority,
                    "normalized_value": constraint.normalized_value.strip() or value,
                    "aliases": list(dict.fromkeys(alias.strip() for alias in constraint.aliases if alias.strip())),
                }
            )
            deduped[(field, normalized_constraint.operator, normalized_constraint.normalized_value.lower())] = normalized_constraint

        return list(deduped.values())

    def ensure_constraint_columns(
        self,
        columns: list[ColumnSpec],
        hard_constraints: list[ConstraintSpec],
        soft_constraints: list[ConstraintSpec],
    ) -> list[ColumnSpec]:
        updated_columns: list[ColumnSpec] = []

        for column in columns:
            normalized_name = normalize_identifier(column.name)
            locked = column.locked or normalized_name == "entity"
            importance = column.importance

            if any(constraint.field == normalized_name for constraint in hard_constraints):
                locked = True
                importance = "high"
            elif any(constraint.field == normalized_name for constraint in soft_constraints) and importance == "low":
                importance = "medium"

            updated_columns.append(
                column.model_copy(
                    update={
                        "name": normalized_name,
                        "importance": importance,
                        "locked": locked,
                    }
                )
            )

        existing_names = {column.name for column in updated_columns}
        for constraint in hard_constraints:
            if constraint.field in existing_names:
                continue
            updated_columns.append(
                ColumnSpec(
                    name=constraint.field,
                    description=f"{constraint.field.replace('_', ' ')} used to evaluate user constraints",
                    importance="high",
                    locked=True,
                )
            )
            existing_names.add(constraint.field)

        for constraint in soft_constraints:
            if constraint.field in existing_names:
                continue
            updated_columns.append(
                ColumnSpec(
                    name=constraint.field,
                    description=f"{constraint.field.replace('_', ' ')} relevant to the user topic",
                    importance="medium",
                    locked=False,
                )
            )
            existing_names.add(constraint.field)

        if "entity" not in existing_names:
            updated_columns.insert(
                0,
                ColumnSpec(
                    name="entity",
                    description="Canonical entity name",
                    importance="high",
                    locked=True,
                ),
            )

        return updated_columns

    def infer_query_covered_constraints(
        self,
        query: str,
        constraints: list[ConstraintSpec],
    ) -> list[str]:
        covered_fields: list[str] = []
        for constraint in constraints:
            if score_constraint_text(constraint, query) >= 1.0:
                covered_fields.append(constraint.field)
        return sorted(set(covered_fields))

    def finalize_query_plans(
        self,
        topic: str,
        query_plans: list[QueryPlan],
        hard_constraints: list[ConstraintSpec],
        soft_constraints: list[ConstraintSpec],
        include_topic_anchor: bool,
        limit: int,
    ) -> list[QueryPlan]:
        topic_query = " ".join(topic.split())
        hard_fields = {constraint.field for constraint in hard_constraints}
        all_constraints = hard_constraints + soft_constraints
        tier_priority = {"anchor": 2, "variant": 1, "recall": 0}

        combined_plans: list[QueryPlan] = []
        if include_topic_anchor and topic_query:
            combined_plans.append(
                QueryPlan(
                    query=topic_query,
                    reason="Preserve the original user phrasing and explicit constraints.",
                    tier="anchor",
                    covered_constraints=sorted(hard_fields),
                )
            )
        combined_plans.extend(query_plans)

        deduped: dict[str, QueryPlan] = {}
        for plan in combined_plans:
            query = " ".join(plan.query.split())
            if not query:
                continue

            covered_constraints = set(normalize_identifier(item) for item in plan.covered_constraints if item.strip())
            covered_constraints.update(self.infer_query_covered_constraints(query, all_constraints))

            tier = plan.tier
            if query.lower() == topic_query.lower():
                tier = "anchor"
                covered_constraints.update(hard_fields)
            elif hard_fields and hard_fields.issubset(covered_constraints):
                tier = "anchor"
            elif covered_constraints and tier == "recall":
                tier = "variant"

            normalized_query = query.lower()
            candidate = QueryPlan(
                query=query,
                reason=plan.reason,
                tier=tier,
                covered_constraints=sorted(covered_constraints),
            )

            existing = deduped.get(normalized_query)
            if existing is None:
                deduped[normalized_query] = candidate
                continue

            merged_covered = sorted(set(existing.covered_constraints) | set(candidate.covered_constraints))
            stronger_tier = existing.tier
            if tier_priority[candidate.tier] > tier_priority[existing.tier]:
                stronger_tier = candidate.tier
            merged_reason = existing.reason if len(existing.reason) >= len(candidate.reason) else candidate.reason
            deduped[normalized_query] = existing.model_copy(
                update={
                    "tier": stronger_tier,
                    "covered_constraints": merged_covered,
                    "reason": merged_reason,
                }
            )

        ranked_query_plans = sorted(
            deduped.values(),
            key=lambda plan: (
                tier_priority.get(plan.tier, 0),
                len(set(plan.covered_constraints) & hard_fields),
                len(plan.covered_constraints),
                len(plan.query),
            ),
            reverse=True,
        )
        return ranked_query_plans[:limit]

    def prefer_hit(
        self,
        current: SearchHit | None,
        incoming: SearchHit,
        query_lookup: dict[str, QueryPlan],
    ) -> SearchHit:
        if current is None:
            return incoming

        tier_priority = {"anchor": 2, "variant": 1, "recall": 0}
        current_plan = query_lookup.get(current.query)
        incoming_plan = query_lookup.get(incoming.query)
        current_score = (
            tier_priority.get(current_plan.tier if current_plan else "recall", 0),
            len(current_plan.covered_constraints if current_plan else []),
            -current.rank,
        )
        incoming_score = (
            tier_priority.get(incoming_plan.tier if incoming_plan else "recall", 0),
            len(incoming_plan.covered_constraints if incoming_plan else []),
            -incoming.rank,
        )
        if incoming_score > current_score:
            return incoming
        return current

    async def build_topic_plan(self, topic: str) -> TopicPlan:
        user_prompt = f"""
Topic: {topic}

Return JSON with:
- normalized_topic
- entity_type
- columns
- hard_constraints
- soft_constraints
- search_queries

Constraints:
- include "entity" in columns
- produce 4 to 6 search queries
- use snake_case for column names and constraint fields
- preserve explicit user constraints instead of broadening them away
- keep columns practical for public web extraction
"""
        plan = await self.llm.complete_json(PLANNER_SYSTEM_PROMPT, user_prompt, TopicPlan)
        plan.normalized_topic = plan.normalized_topic or topic
        plan.hard_constraints = self.dedupe_constraints(plan.hard_constraints, priority="hard")
        plan.soft_constraints = [
            constraint
            for constraint in self.dedupe_constraints(plan.soft_constraints, priority="soft")
            if constraint.field not in {hard_constraint.field for hard_constraint in plan.hard_constraints}
        ]
        plan.columns = self.ensure_constraint_columns(plan.columns, plan.hard_constraints, plan.soft_constraints)
        plan.search_queries = self.finalize_query_plans(
            topic=topic,
            query_plans=plan.search_queries,
            hard_constraints=plan.hard_constraints,
            soft_constraints=plan.soft_constraints,
            include_topic_anchor=True,
            limit=settings.MAX_BASE_QUERIES,
        )
        demo_print(
            "Layer 1 - Topic Plan",
            {
                "topic": topic,
                "entity_type": plan.entity_type,
                "columns": [column.name for column in plan.columns],
                "hard_constraints": [constraint.model_dump() for constraint in plan.hard_constraints],
                "soft_constraints": [constraint.model_dump() for constraint in plan.soft_constraints],
                "queries": [query.model_dump() for query in plan.search_queries],
            },
        )
        return plan

    def _build_bm25_inputs(self, hits: list[SearchHit]) -> tuple[BM25Scorer, list[list[str]]]:
        corpus_tokens: list[list[str]] = []
        for hit in hits:
            text = f"{hit.title} {hit.snippet}"
            corpus_tokens.append(tokenize(text))
        return BM25Scorer(corpus_tokens), corpus_tokens

    def _query_tokens_for_hit_ranking(
        self,
        topic: str,
        hard_constraints: list[ConstraintSpec],
        soft_constraints: list[ConstraintSpec],
    ) -> list[str]:
        tokens = tokenize(topic)
        for constraint in hard_constraints + soft_constraints:
            tokens.extend(tokenize(constraint.field))
            tokens.extend(tokenize(constraint.normalized_value or constraint.value))
            for alias in constraint.aliases:
                tokens.extend(tokenize(alias))
        return tokens

    def _source_prior(self, url: str) -> float:
        url_lower = url.lower()
        bonus = 0.0

        trusted_markers = [
            "crunchbase.com",
            "linkedin.com",
            "github.com",
            "techcrunch.com",
            "forbes.com",
            "bloomberg.com",
            "wikipedia.org",
            ".gov/",
            ".edu/",
        ]
        noisy_markers = [
            "facebook.com",
            "instagram.com",
            "pinterest.com",
            "tiktok.com",
        ]

        if any(x in url_lower for x in trusted_markers):
            bonus += 0.6
        if any(x in url_lower for x in noisy_markers):
            bonus -= 0.4
        return bonus

    def rank_hits(
        self,
        topic: str,
        hits: list[SearchHit],
        query_plans: list[QueryPlan],
        hard_constraints: list[ConstraintSpec],
        soft_constraints: list[ConstraintSpec],
    ) -> list[SearchHit]:
        """
        Hybrid pre-ranking before fetch.

        Signals:
        - BM25 over title + snippet
        - constraint-aware text match
        - query tier and covered constraints
        - source prior
        - SERP rank
        """
        if not hits:
            return []

        query_lookup = {plan.query: plan for plan in query_plans}
        hard_fields = {constraint.field for constraint in hard_constraints}
        soft_fields = {constraint.field for constraint in soft_constraints}
        tier_bonus = {"anchor": 2.5, "variant": 1.25, "recall": 0.25}

        bm25, _ = self._build_bm25_inputs(hits)
        ranking_query_tokens = self._query_tokens_for_hit_ranking(topic, hard_constraints, soft_constraints)

        def score(hit_index_and_hit: tuple[int, SearchHit]) -> float:
            idx, hit = hit_index_and_hit
            text = f"{hit.title} {hit.snippet} {hit.url}"
            bm25_score = bm25.score(ranking_query_tokens, idx)

            query_plan = query_lookup.get(hit.query)
            query_bonus = 0.0
            if query_plan is not None:
                query_bonus += tier_bonus.get(query_plan.tier, 0.0)
                query_bonus += 1.75 * len(set(query_plan.covered_constraints) & hard_fields)
                query_bonus += 0.75 * len(set(query_plan.covered_constraints) & soft_fields)

            hard_constraint_bonus = sum(1.75 * score_constraint_text(constraint, text) for constraint in hard_constraints)
            soft_constraint_bonus = sum(0.8 * score_constraint_text(constraint, text) for constraint in soft_constraints)

            # Penalize missing evidence for clearly-preserved hard constraints in stronger query tiers.
            hard_miss_penalty = 0.0
            if query_plan is not None and query_plan.tier in {"anchor", "variant"}:
                for constraint in hard_constraints:
                    if constraint.field in set(query_plan.covered_constraints):
                        if score_constraint_text(constraint, text) <= 0.0:
                            hard_miss_penalty += 0.6

            rank_bonus = 1.0 / math.sqrt(max(hit.rank, 1))
            domain_bonus = self._source_prior(hit.url)

            return (
                1.0 * bm25_score
                + query_bonus
                + hard_constraint_bonus
                + soft_constraint_bonus
                + 0.75 * rank_bonus
                + domain_bonus
                - hard_miss_penalty
            )

        ranked = sorted(enumerate(hits), key=score, reverse=True)
        return [hit for _, hit in ranked]

    async def search_and_fetch(
        self,
        topic: str,
        query_plans: list[QueryPlan],
        hard_constraints: list[ConstraintSpec],
        soft_constraints: list[ConstraintSpec],
    ) -> tuple[list[SearchHit], list[PageDocument]]:
        all_hits: dict[str, SearchHit] = {}
        queries = [query_plan.query for query_plan in query_plans]
        query_lookup = {query_plan.query: query_plan for query_plan in query_plans}

        for query in queries:
            hits = self.search_client.search(query, settings.SEARCH_RESULTS_PER_QUERY)
            for hit in hits:
                all_hits[hit.url] = self.prefer_hit(all_hits.get(hit.url), hit, query_lookup)

        ranked_hits = self.rank_hits(topic, list(all_hits.values()), query_plans, hard_constraints, soft_constraints)
        ranked_hits = ranked_hits[: settings.MAX_PAGES_FOR_EXTRACTION]

        demo_print(
            "Layer 2 - Ranked Search Hits",
            {
                "queries": [query_plan.model_dump() for query_plan in query_plans],
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
        hard_constraints: list[ConstraintSpec],
        page: PageDocument,
    ) -> ExtractionBatch:
        allowed_columns = [c.name for c in columns if c.name != "entity"]

        user_prompt = f"""
Topic: {topic}
Expected entity_type: {entity_type}
Allowed output columns: {json.dumps(allowed_columns)}
Full column schema: {json.dumps(compact_columns(columns))}
Hard constraints: {json.dumps([constraint.model_dump() for constraint in hard_constraints])}
Page title: {page.title}
Page url: {page.url}

Important:
- Every entity must include entity, entity_type, and cells.
- cells must be keyed only by allowed output columns.
- Do not place fields at the top level.
- Use only evidence from this page.
- If a page clearly contradicts a hard constraint for an entity, skip that entity.
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
        if not rows:
            return columns

        kept_columns = [column for column in columns if column.name == "entity" or column.locked]
        kept_names = {column.name for column in kept_columns}
        coverage_counts: list[tuple[int, ColumnSpec]] = []
        minimum_filled = max(2, math.ceil(len(rows) * 0.5))

        for column in columns:
            if column.name == "entity":
                continue

            filled = sum(1 for row in rows if row.cells.get(column.name) and row.cells[column.name].value)
            coverage_counts.append((filled, column))
            if column.locked or filled >= minimum_filled:
                if column.name in kept_names:
                    continue
                kept_columns.append(column)
                kept_names.add(column.name)

        if len(kept_columns) > 1:
            return kept_columns

        fallback_columns = [column for filled, column in sorted(coverage_counts, reverse=True) if filled > 0][:3]
        return kept_columns + fallback_columns

    def compact_rows_for_output(self, rows: list[FinalRow], columns: list[ColumnSpec]) -> list[FinalRow]:
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
        for key, incoming_row in incoming.items():
            if key not in base:
                base[key] = incoming_row
                continue

            base_row = base[key]
            for column_name, cell in incoming_row.cells.items():
                base_row.cells[column_name] = self.merge_cell(base_row.cells.get(column_name), cell)

        return base

    def get_missing_columns(self, rows: dict[str, FinalRow], columns: list[ColumnSpec]) -> list[dict]:
        missing = []
        important_columns = [c for c in columns if c.name != "entity" and c.importance in {"high", "medium"}]

        for row in rows.values():
            missing_columns = []
            for col in important_columns:
                cell = row.cells.get(col.name)
                if not cell or not cell.value:
                    missing_columns.append(col.name)

            if missing_columns:
                missing.append(
                    {
                        "entity": row.entity,
                        "missing_columns": missing_columns,
                    }
                )

        return missing

    async def build_deeper_queries(
        self,
        topic: str,
        rows: dict[str, FinalRow],
        columns: list[ColumnSpec],
        hard_constraints: list[ConstraintSpec],
        soft_constraints: list[ConstraintSpec],
    ) -> list[QueryPlan]:
        missing = self.get_missing_columns(rows, columns)
        if not missing:
            return []

        user_prompt = f"""
Topic: {topic}
Hard constraints: {json.dumps([constraint.model_dump() for constraint in hard_constraints], indent=2)}
Soft constraints: {json.dumps([constraint.model_dump() for constraint in soft_constraints], indent=2)}

Missing coverage summary:
{json.dumps(missing[:5], indent=2)}

Return only:
{{
  "search_queries": [
    {{
      "query": "query 1",
      "reason": "why this query helps",
      "tier": "anchor",
      "covered_constraints": ["constraint_field"]
    }}
  ]
}}
"""
        plan = await self.llm.complete_json(
            DEEPER_SEARCH_SYSTEM_PROMPT,
            user_prompt,
            DeeperQueryPlan,
        )

        finalized_queries = self.finalize_query_plans(
            topic=topic,
            query_plans=plan.search_queries,
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            include_topic_anchor=False,
            limit=settings.MAX_DEEPER_QUERIES,
        )
        demo_print(
            "Layer 6 - Deeper Query Plan",
            {
                "topic": topic,
                "missing": missing[:5],
                "queries": [query.model_dump() for query in finalized_queries],
            },
        )
        return finalized_queries

    def score_row_constraints(self, row: FinalRow, constraints: list[ConstraintSpec], weight: float) -> float:
        total = 0.0
        for constraint in constraints:
            cell = row.cells.get(constraint.field)
            if not cell or not cell.value:
                continue
            total += weight * score_constraint_text(constraint, cell.value)
        return total

    def rank_final_rows(
        self,
        rows: dict[str, FinalRow],
        entity_type: str,
        topic: str,
        hard_constraints: list[ConstraintSpec],
        soft_constraints: list[ConstraintSpec],
    ) -> list[FinalRow]:
        topic_tokens = token_set(topic)

        def score(row: FinalRow) -> tuple[float, float, int, str]:
            filled = sum(1 for c in row.cells.values() if c.value)
            confidence = sum(c.confidence for c in row.cells.values())
            topical = len(topic_tokens & token_set(row.entity))
            hard_constraint_bonus = self.score_row_constraints(row, hard_constraints, weight=2.25)
            soft_constraint_bonus = self.score_row_constraints(row, soft_constraints, weight=1.0)
            locked_column_bonus = sum(
                0.5
                for field in {constraint.field for constraint in hard_constraints}
                if row.cells.get(field) and row.cells[field].value
            )
            return (
                filled + topical + hard_constraint_bonus + soft_constraint_bonus + locked_column_bonus,
                confidence,
                filled,
                row.entity.lower(),
            )

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
        plan = await self.build_topic_plan(topic)

        initial_queries = [q.query for q in plan.search_queries]
        initial_hits, initial_pages = await self.search_and_fetch(
            topic,
            plan.search_queries,
            plan.hard_constraints,
            plan.soft_constraints,
        )

        initial_batches: list[ExtractionBatch] = []
        for page in initial_pages:
            try:
                batch = await self.extract_from_page(topic, plan.entity_type, plan.columns, plan.hard_constraints, page)
                initial_batches.append(batch)
            except Exception:
                demo_print("Layer 4 - Extraction Failure", {"url": page.url})
                continue

        merged_rows = self.merge_batches(initial_batches)

        deeper_queries_used: list[str] = []
        deeper_pages_count = 0

        for _ in range(settings.DEEPER_SEARCH_ROUNDS):
            deeper_query_plans = await self.build_deeper_queries(
                topic,
                merged_rows,
                plan.columns,
                plan.hard_constraints,
                plan.soft_constraints,
            )
            deeper_query_plans = [query for query in deeper_query_plans if query.query not in deeper_queries_used]
            if not deeper_query_plans:
                break

            deeper_queries_used.extend(query.query for query in deeper_query_plans)

            deeper_hits_map: dict[str, SearchHit] = {}
            deeper_query_lookup = {query.query: query for query in deeper_query_plans}
            for query in deeper_query_plans:
                hits = self.search_client.search(query.query, 3)
                for hit in hits:
                    deeper_hits_map[hit.url] = self.prefer_hit(deeper_hits_map.get(hit.url), hit, deeper_query_lookup)

            deeper_hits = self.rank_hits(
                topic,
                list(deeper_hits_map.values()),
                deeper_query_plans,
                plan.hard_constraints,
                plan.soft_constraints,
            )[: settings.MAX_PAGES_FOR_EXTRACTION]

            deeper_pages = await self.fetcher.fetch_many(deeper_hits)
            deeper_pages_count += len(deeper_pages)

            deeper_batches: list[ExtractionBatch] = []
            for page in deeper_pages:
                try:
                    batch = await self.extract_from_page(
                        topic,
                        plan.entity_type,
                        plan.columns,
                        plan.hard_constraints,
                        page,
                    )
                    deeper_batches.append(batch)
                except Exception:
                    demo_print("Layer 7 - Deeper Extraction Failure", {"url": page.url})
                    continue

            merged_rows = self.merge_rows_into(merged_rows, self.merge_batches(deeper_batches))

        final_rows = self.rank_final_rows(
            merged_rows,
            plan.entity_type,
            topic,
            plan.hard_constraints,
            plan.soft_constraints,
        )
        final_columns = self.prune_sparse_columns(plan.columns, final_rows)
        final_rows = self.compact_rows_for_output(final_rows, final_columns)
        final_rows = [row for row in final_rows if len(row.cells) >= 2]

        diagnostics = {
            "initial_queries": [query.model_dump() for query in plan.search_queries],
            "deeper_queries": deeper_queries_used,
            "hard_constraints": [constraint.model_dump() for constraint in plan.hard_constraints],
            "soft_constraints": [constraint.model_dump() for constraint in plan.soft_constraints],
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
            "Layer 8 - Final Result",
            {
                "rows": summarize_rows(final_rows),
                "diagnostics": diagnostics,
            },
        )
        return final_result