from __future__ import annotations

import asyncio
import re
from html import unescape
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.models import PageDocument, SearchHit

_WHITESPACE_RE = re.compile(r"\s+")


def html_to_text(html: str, limit: int) -> str:
    """
    Converts raw HTML into compact plain text for LLM consumption.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(" ")
    text = unescape(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:limit]


class PageFetcher:
    """
    Concurrent page fetcher with a small in-memory cache.
    """

    def __init__(self) -> None:
        self.timeout = httpx.Timeout(settings.FETCH_TIMEOUT_SECONDS)
        self.sem = asyncio.Semaphore(settings.FETCH_CONCURRENCY)
        self.cache: dict[str, PageDocument] = {}
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AgenticSearchBot/1.0)"
        }

    async def _fetch_one(self, client: httpx.AsyncClient, hit: SearchHit) -> PageDocument | None:
        """
        Fetches one URL, cleans the HTML, and returns a PageDocument.
        Uses cache when available.
        """
        if hit.url in self.cache:
            return self.cache[hit.url]

        async with self.sem:
            try:
                response = await client.get(hit.url, headers=self.headers, follow_redirects=True)
                if response.status_code >= 400:
                    return None

                page = PageDocument(
                    url=str(response.url),
                    title=hit.title,
                    snippet=hit.snippet,
                    text=html_to_text(response.text, settings.MAX_PAGE_CHARS),
                )

                if page.text:
                    self.cache[hit.url] = page
                    return page
            except Exception:
                return None

        return None

    async def fetch_many(self, hits: Iterable[SearchHit]) -> list[PageDocument]:
        """
        Fetches many URLs concurrently.
        Filters out failures and empty pages.
        """
        hit_list = list(hits)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._fetch_one(client, hit) for hit in hit_list]
            pages = await asyncio.gather(*tasks)
        kept_pages = [p for p in pages if p and p.text]
        return kept_pages
