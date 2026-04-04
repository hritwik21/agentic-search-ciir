# Backend

FastAPI backend for the agentic search demo.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
