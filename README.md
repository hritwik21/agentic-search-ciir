# Agentic Search 

Agentic Search is a full-stack demo that turns an open-ended topic into a structured comparison table. The system plans search queries, retrieves public web pages, extracts entity-level fields with evidence, merges duplicates, and returns ranked results in a simple UI. The latest version adds **constraint-aware query planning** and a **hybrid reranker** so queries with important filters such as geography, dates, numeric thresholds, or exclusions are handled more reliably.

## Demo

- **Frontend:** https://agentic-search-ciir-frontend.vercel.app/
- **Backend:** https://agentic-search-backend-hg.onrender.com/

<img width="1179" height="578" alt="Screenshot 2026-04-10 at 3 01 26 PM" src="https://github.com/user-attachments/assets/1ce86cdd-a7e7-4cc0-86c5-9cdb40559645" />

## Architecture

<img width="1439" height="1646" alt="image" src="https://github.com/user-attachments/assets/510bb013-c2f4-4228-a447-f9c5c9988a31" />

### Tech stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/Tailwind%20CSS-38B2AC?logo=tailwindcss&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Vercel-000000?logo=vercel&logoColor=white" alt="Vercel" />
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Groq-black" alt="Groq" />
  <img src="https://img.shields.io/badge/DDGS-blue" alt="DDGS" />
  <img src="https://img.shields.io/badge/httpx-green" alt="httpx" />
  <img src="https://img.shields.io/badge/BeautifulSoup-yellowgreen" alt="BeautifulSoup" />
  <img src="https://img.shields.io/badge/Pydantic-red?logo=pydantic&logoColor=white" alt="Pydantic" />
  <img src="https://img.shields.io/badge/Render-46E3B7?logo=render&logoColor=white" alt="Render" />
</p>

## Components

### 1. Planner LLM

Parses the user query into a retrieval plan with:

- normalized topic
- inferred entity type
- dynamic schema
- initial search queries
- **hard constraints** and **soft constraints**

This separates core retrieval intent from filters such as location, time, and numeric thresholds, so later stages do not treat the query as a flat bag of words.

### 2. Search Client

Executes multiple planned queries with DDGS to improve recall across different query formulations. Candidate hits are aggregated and URL-deduplicated before fetch to avoid redundant downstream processing.

### 3. Hybrid Reranker

Performs first-stage ranking over search hits before page fetch using:

- **BM25-style lexical matching** on title and snippet
- **query-plan priors** from stronger query variants
- **constraint-aware scoring** for hard and soft filters
- **source priors** for cleaner vs noisier domains
- **SERP rank prior** as a weak additional signal

This improves early precision by promoting hits that match both topical relevance and explicit query constraints.

### 4. Fetcher

Fetches the top-ranked pages concurrently, removes boilerplate, and truncates content to a bounded context window for extraction.

### 5. Extractor LLM

Maps each fetched page into structured entity-field outputs. Each extracted field is paired with supporting evidence, making the final table grounded and inspectable.

### 6. Merge and Standardize

Resolves cross-page duplicates using normalized names and fuzzy matching, then consolidates field values and evidence into a single entity-centric representation.

### 7. Deeper Query Planner

Generates targeted follow-up queries when important fields remain missing. These follow-up queries aim to improve coverage while still preserving important user constraints.

### 8. Final Result

Ranks merged entities using:

- field coverage
- extraction confidence
- topical alignment
- constraint alignment

This pushes more complete and better-matched entities higher in the final output.

## Repo Structure

```text
agentic-search-github-ready/
├── backend/                  # FastAPI API and search pipeline
│   ├── app/
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/                 # Next.js UI
│   ├── src/app/
│   ├── package.json
│   └── .env.example
├── render.yaml               # Optional Render deployment blueprint
├── .gitignore
└── README.md
```

## API Reference

### `GET /health`
Returns API health status.

**Response**
```json
{
  "status": "ok"
}
```

### `POST /search`
Runs the search pipeline and returns structured results.

**Request**
```json
{
  "topic": "AI startups in the US with funding > 10M"
}
```

**Response**
```json
{
  "topic": "AI startups in the US with funding > 10M",
  "entity_type": "startup",
  "columns": [
    {
      "name": "entity",
      "description": "Canonical entity name",
      "importance": "high"
    }
  ],
  "rows": [
    {
      "entity_id": "startup|example-ai",
      "entity": "Example AI",
      "cells": {
        "location": {
          "value": "United States",
          "confidence": 0.92,
          "sources": [
            {
              "url": "https://example.com",
              "title": "Example AI",
              "excerpt": "Example AI is a US-based startup..."
            }
          ]
        }
      }
    }
  ],
  "diagnostics": {}
}
```

## Future Work

- **Add a learned reranker after first-stage retrieval.**  
  Add a cross-encoder or learning-to-rank step on the top candidates after BM25 pre-ranking. This should improve precision by scoring full query-document relevance instead of relying mostly on lexical match and heuristic boosts.

- **Improve constraint parsing for exclusions, time filters, and numeric conditions.**  
  Better handle filters such as `not`, date ranges, before/after conditions, and numeric thresholds or intervals. This would make query planning and reranking align more closely with the actual user intent.

- **Strengthen entity resolution beyond name-based fuzzy matching.**  
  Use additional signals such as source domain, location, company attributes, and supporting evidence when merging entities. This would reduce both false merges and missed merges across aliases or naming variations.

- **Add caching and observability across the pipeline.**  
  Cache query plans, search results, fetched pages, and extraction outputs where appropriate, and track metrics for retrieval quality, latency, and extraction coverage. This would improve efficiency and make debugging easier.

## Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/agentic-search-ciir.git
cd agentic-search-ciir

# 2. Start the backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

3. Add your API key in backend/.env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
FRONTEND_URL=http://localhost:3000

# 4. Run the backend
uvicorn main:app --reload

# 5. In a new terminal, start the frontend
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

### Sample queries

- `AI startups in the US with funding > 10M`
- `open source vector databases under 2GB RAM`
- `robotics companies in Europe with series B funding` 
