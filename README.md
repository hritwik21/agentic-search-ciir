# Agentic Search

Agentic Search is a full-stack demo that turns an open-ended user topic into a structured comparison table. The system plans search queries, retrieves public web pages, extracts entity-level fields with evidence, merges duplicates, and returns ranked rows through a simple web UI.

## What this repository contains

```text
agentic-search-github-ready/
├── backend/                  # FastAPI API and agentic pipeline
│   ├── app/
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/                 # Next.js UI
│   ├── src/app/
│   ├── package.json
│   └── .env.example
├── render.yaml               # Optional Render deployment blueprint for backend
├── .gitignore
└── README.md
```

## Approach

The system follows a staged retrieval and extraction pipeline rather than relying on a single prompt.

1. **Topic planning**
   - The planner LLM converts the user topic into a normalized topic, an entity type, a dynamic schema, and a set of initial search queries.
   - This avoids hardcoding domain-specific columns.

2. **Broad retrieval**
   - The backend issues multiple web queries with DDGS.
   - Results are URL-normalized and deduplicated before fetch.

3. **Lightweight ranking before fetch**
   - Search hits are re-ranked using lexical overlap, result position, and simple domain priors.
   - This keeps the expensive extraction stage focused on stronger pages.

4. **Page fetch and cleanup**
   - Pages are fetched concurrently with `httpx`.
   - Boilerplate HTML is stripped with BeautifulSoup and the text is truncated to a safe budget.

5. **Structured extraction with provenance**
   - The extractor LLM converts each page into typed entities and cells.
   - Each cell includes confidence and source references so the final table remains grounded.

6. **Merge and standardization**
   - Entities are merged using normalized names plus fuzzy matching.
   - Cell values are cleaned and conflicting evidence is reduced.

7. **Coverage-driven deeper search**
   - If key columns remain empty, the system generates targeted follow-up queries.
   - This is meant to improve coverage without blindly repeating the initial search.

## Design decisions

### Dynamic schema instead of hardcoded columns
The core goal is to support generic research tasks, not only restaurants or one vertical. The planner therefore chooses columns at runtime.

### Separate retrieval, extraction, and merge stages
These stages are split across modules so that ranking, extraction quality, and entity resolution can be tuned independently.

### Typed validation around LLM output
Pydantic models act as a contract between the LLM and the rest of the system. This reduces brittle downstream handling when the model returns slightly malformed JSON.

### Evidence-carrying cells
Every non-empty value can point to a supporting URL. This makes the output easier to audit and demo.

### Budgeted multi-step search
The code intentionally limits query count, page count, and text size so the demo remains usable on modest hardware and cheaper API usage.

## Known limitations

- The ranking heuristics are simple and not learned.
- Entity deduplication is name-based and can still merge near-matches incorrectly or fail on aliases.
- The system depends on public web pages, so page quality and anti-bot restrictions affect results.
- Extraction quality depends on the LLM and can still produce sparse or partially incorrect fields.
- The current UI is a demo table, not a full analyst workflow with editing, export, or result inspection panes.
- CORS in the backend is configured mainly for local development. For deployment, set the exact frontend URL.

## Setup instructions

### 1) Clone the repo

```bash
git clone https://github.com/<your-username>/agentic-search.git
cd agentic-search
```

### 2) Start the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Groq key to `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
FRONTEND_URL=http://localhost:3000
```

Then run:

```bash
uvicorn main:app --reload
```

### 3) Start the frontend

In a new terminal:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## GitHub push steps

### Option A: GitHub website + terminal

1. Create a new empty public repository on GitHub.
2. Do not add a README, `.gitignore`, or license during creation if you are pushing this folder as-is.
3. In your terminal:

```bash
cd agentic-search-github-ready
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

GitHub documents both creating repositories and pushing existing local code from the command line. citeturn278717search2turn278717search17turn278717search20

### Option B: GitHub CLI

```bash
cd agentic-search-github-ready
git init
git add .
git commit -m "Initial commit"
gh repo create <repo-name> --public --source=. --remote=origin --push
```

GitHub’s CLI supports creating a repository and pushing an existing local project directly. citeturn278717search2turn278717search5

## What “URL with a live demo on a free tier cloud instance” means

It means your submission includes a public link that the reviewer can open in a browser and try without running the code locally.

Examples:
- a frontend URL like `https://your-app.vercel.app`
- a backend health endpoint like `https://your-api.onrender.com/health`

For this project, the most practical path is:
- **Frontend on Vercel** using the free hobby tier for Next.js
- **Backend on Render** using a free Python web service

Render supports free web services for Python apps and documents the standard FastAPI deployment flow with a build command and a `uvicorn` start command. citeturn278717search0turn278717search6turn278717search9turn278717search15

## How to create a live demo

### Backend on Render

1. Push this repository to GitHub.
2. Sign in to Render.
3. Click **New +** → **Web Service**.
4. Connect your GitHub repo.
5. Select the repo and set:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   - `GROQ_API_KEY`
   - `GROQ_MODEL`
   - `FRONTEND_URL` later, after your frontend URL is known
7. Deploy.
8. Test `https://<your-render-service>.onrender.com/health`

These settings match Render’s FastAPI deployment documentation. citeturn278717search0turn278717search15

### Frontend on Vercel

1. Import the same GitHub repo into Vercel.
2. Set the **Root Directory** to `frontend`.
3. Add the environment variable:
   - `NEXT_PUBLIC_API_URL=https://<your-render-service>.onrender.com`
4. Deploy.
5. Copy the Vercel URL.
6. Go back to Render and set `FRONTEND_URL=https://<your-vercel-app>.vercel.app` if you want strict CORS.

## Recommended submission checklist

- Remove all real secrets from `.env` files before pushing.
- Keep only `.env.example` files in GitHub.
- Make the repository public.
- Add a short demo section to this README once deployed:

```md
## Live demo
- Frontend: https://<your-vercel-url>
- Backend health: https://<your-render-url>/health
```

- Verify the frontend can call the deployed backend.
- Add 2 to 3 example queries in the README for reviewers.

## Security note

The original zip contained local environment files and generated artifacts. Those should stay out of source control. This cleaned version keeps only example environment files and excludes build folders, virtual environments, and local caches.
