from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from ddgs import DDGS

from app.models import SearchHit


def normalize_url(url: str) -> str:
    """
    Removes query params and fragments to improve deduplication.
    """

    parts = urlsplit(url.strip())
    cleaned = parts._replace(query="", fragment="")
    return urlunsplit(cleaned).rstrip("/")


class SearchClient:
    """
    Thin wrapper over DDGS search.
    """

    def __init__(self, timeout: int = 10) -> None:
        self.ddgs = DDGS(timeout=timeout)

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        """
        Runs one search query and returns normalized, deduplicated hits.
        """
        raw = self.ddgs.text(
            query,
            max_results=max_results,
            region="us-en",
            safesearch="moderate",
            backend="auto",
        )

        results: list[SearchHit] = []
        seen: set[str] = set()

        for rank, item in enumerate(raw or [], start=1):
            url = item.get("href") or item.get("url") or ""
            if not url:
                continue

            url = normalize_url(url)
            if not url or url in seen:
                continue
            seen.add(url)

            results.append(
                SearchHit(
                    query=query,
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("body") or item.get("snippet") or "").strip(),
                    rank=rank,
                )
            )
        return results
