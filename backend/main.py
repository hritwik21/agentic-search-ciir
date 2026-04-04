from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import AliasChoices, BaseModel, Field

from app.console import demo_print
from app.pipeline import AgenticSearchPipeline

app = FastAPI(title="Agentic Search API")
pipeline = AgenticSearchPipeline()

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

allowed_origins = [
    "http://localhost:3000",
    frontend_url,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    topic: str = Field(..., min_length=2, validation_alias=AliasChoices("topic", "query", "text", "q"))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path("frontend/index.html"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
async def search(request: SearchRequest) -> dict:
    try:
        demo_print("API Request", {"topic": request.topic})
        result = await pipeline.run(request.topic)
        demo_print(
            "API Response",
            {
                "topic": request.topic,
                "rows": len(result.rows),
                "diagnostics": result.diagnostics,
            },
        )
        return result.model_dump()
    except Exception as exc:
        demo_print("API Error", {"topic": request.topic, "error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc)) from exc
