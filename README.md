# LuminaData Enterprise v6.0
**AI-Powered Citizen Data Quality Platform**

Built with LangGraph · FastAPI MCP · pgvector RAG · Streamlit

---

## What is this?

LuminaData is a data quality monitoring platform for citizen registries. You talk to it in plain English and it:

- **Queries your database** — "Show all citizens with missing National IDs"
- **Runs compliance audits** — "Check our NDMO/PDPL compliance score"
- **Explains DQ issues** — "Why are 12 records flagged as Non-Compliant?"

Upload your own NDMO/PDPL PDF documents and the AI uses them when generating compliance reports.

---

## Requirements

| Tool | Version | Check |
|---|---|---|
| Docker Desktop | Latest | `docker --version` |
| Python | 3.11 | `python --version` |
| Git | Any | `git --version` |
| Groq API key | Free | [console.groq.com](https://console.groq.com) |

---

## Project Files

```
luminadata/
├── app.py                  Streamlit UI — chat, dashboard, PDF uploader
├── orchestration.py        LangGraph agents (Orchestrator, SQL, DQ, General)
├── mcp_server.py           FastAPI MCP server — all database tools
├── mcp_client.py           HTTP client used by agents to call MCP tools
├── rag_ingest.py           CLI script to ingest PDFs (alternative to UI uploader)
├── init_db.sql             Database schema + 25 sample citizens
├── docker-compose.yml      Runs PostgreSQL + MCP server in Docker
├── Dockerfile.mcp          Docker image for MCP server
├── Dockerfile.app          Docker image for Streamlit (optional)
├── requirements_app.txt    Python packages for local Streamlit run
├── requirements_mcp.txt    Python packages for MCP Docker container
└── .env.example            Environment variable template — copy to .env
```

---

## Setup — 5 Steps

### Step 1 — Get the code

```bash
git clone https://github.com/YOUR_USERNAME/luminadata.git
cd luminadata
```

### Step 2 — Create your `.env` file

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` in any text editor and set your Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
```

Leave everything else as the defaults.

### Step 3 — Create Python virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements_app.txt
```

> First install takes 3–5 minutes — it downloads sentence-transformers (~80MB).

### Step 4 — Start Docker services

Open Docker Desktop first, then run:

```bash
docker compose up postgres mcp_server -d --build
```

Wait ~30 seconds, then check everything is healthy:

```bash
docker compose ps
```

Both containers should show `(healthy)` in the STATUS column.

Confirm the MCP server is responding:

```bash
# Windows PowerShell
Invoke-WebRequest http://localhost:8765/health

# Mac / Linux
curl http://localhost:8765/health
```

Expected: `{"status":"ok","db":true,"version":"6.0.0"}`

### Step 5 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**Login:**

| Username | Password | Access |
|---|---|---|
| `admin` | `admin123` | Full — can upload PDFs |
| `viewer` | `viewer123` | Read-only |

---

## How to Use

### Ask questions in the chat

**Data queries:**
- "Show all citizens with expired IDs"
- "How many records are missing a phone number?"
- "List citizens from Riyadh with missing National IDs"
- "Are there any records where expiry date is before issue date?"

**Compliance audits:**
- "Run a full NDMO compliance audit"
- "What is our current DQ score?"
- "Check PDPL Article 5 compliance"

**General questions:**
- "What does the Completeness dimension mean?"
- "How do I connect DBeaver to the database?"

### Upload PDF documents (Admin only)

1. Sign in as `admin`
2. In the left sidebar, find **📚 RAG Knowledge Base**
3. Click the file picker → select your PDF
4. Click **Ingest** → wait for the progress bar

After ingesting, the AI uses your PDF content when generating compliance reports.

**Recommended PDFs:**
- NDMO National Data Governance Operational Framework
- PDPL (Personal Data Protection Law)
- NDMO Data Quality Standard

### DBeaver database connection

| Field | Value |
|---|---|
| Host | `localhost` |
| Port | `5433` |
| Database | `luminadata` |
| Username | `lumina_user` |
| Password | `lumina_pass` |

---

## Architecture

```
Browser → Streamlit UI
               │
               ▼
      LangGraph Orchestrator
      classifies intent into:
      SQL_QUERY | NDMO_COMPLIANCE_CHECK | GENERAL_INFO
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
   SQL Agent  DQ Agent  General Agent
       │       │
       └───┬───┘
           ▼
      MCP Server (Docker :8765)
      tools: execute_sql_query
             query_NDMO_knowledge_base
             get_dq_summary ...
           │
           ▼
      PostgreSQL (Docker :5433)
      ├── citizens          (registry)
      ├── audit_logs        (agent history)
      ├── decision_memory   (corrections)
      └── rag_documents     (your PDFs)
```

Agents never touch the database directly — everything goes through the MCP server.

---

## Stopping and Restarting

```bash
# Stop (keeps data)
docker compose down

# Stop + delete all data (fresh start)
docker compose down -v

# Restart
docker compose up postgres mcp_server -d
streamlit run app.py
```

---

## Environment Variables

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | — | ✅ | Get free at console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | No | Groq model name |
| `MCP_SERVER_URL` | `http://localhost:8765` | No | MCP server address |
| `MCP_SECRET` | `lumina_mcp_secret_2024` | No | Shared auth token — change in production |
| `DB_HOST` | `localhost` | No | Postgres host |
| `DB_PORT` | `5433` | No | Postgres port (external) |
| `DB_NAME` | `luminadata` | No | Database name |
| `DB_USER` | `lumina_user` | No | Database username |
| `DB_PASSWORD` | `lumina_pass` | No | Database password |

---

## Troubleshooting

**"MCP Offline" in the sidebar**
```bash
docker compose ps          # check container status
docker compose logs mcp_server --tail=30   # check for errors
docker compose down -v && docker compose up postgres mcp_server -d --build
```

**Docker build takes 5+ minutes**
Make sure `sentence-transformers` and `pymupdf` are NOT in `requirements_mcp.txt`.
They belong only in `requirements_app.txt`.

**"No readable text found in PDF"**
Your PDF is scanned (image-only). Use [smallpdf.com](https://smallpdf.com) → PDF OCR to add a text layer, then re-upload.

**Port 5433 already in use**
```bash
# Windows
net stop postgresql

# Mac / Linux
sudo lsof -i :5433
```

**Groq API error**
Check `GROQ_API_KEY` in your `.env` file. Get a new key at [console.groq.com](https://console.groq.com).

---

## Sample Data

The database is pre-loaded with **25 synthetic citizen records** (from `init_db.sql`) with intentional quality issues:

- 4 records with missing National IDs
- 11 records with expired ID documents
- 1 record where expiry date is before issue date
- Several records missing phone or email

> All names and IDs are fictional — no real personal data is included.

---

## Tech Stack

| | Technology |
|---|---|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM | Groq (llama-3.1-8b-instant) |
| MCP server | FastAPI + uvicorn |
| Database | PostgreSQL 15 + pgvector |
| PDF parsing | PyMuPDF |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Containers | Docker Compose |

---

*LuminaData v6.0 · LangGraph · MCP · pgvector RAG · Streamlit*