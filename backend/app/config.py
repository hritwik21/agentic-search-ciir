from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    SEARCH_RESULTS_PER_QUERY: int = 5
    FETCH_TIMEOUT_SECONDS: float = 10.0
    FETCH_CONCURRENCY: int = 8
    LLM_CONCURRENCY: int = 3
    MAX_PAGE_CHARS: int = 12000

    MAX_BASE_QUERIES: int = 4
    MAX_DEEPER_QUERIES: int = 6
    DEEPER_SEARCH_ROUNDS: int = 1

    MAX_PAGES_FOR_EXTRACTION: int = 8
    MAX_FINAL_ROWS: int = 10


settings = Settings()