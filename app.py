# -*- coding: utf-8 -*-
"""
LuminaData Enterprise v6.0
AI-Powered Citizen Data Guardian — Agentic Edition

Architecture:
  • LangGraph Master Orchestrator routes to specialised agents
  • All data access via MCP Server (never direct DB calls)
  • pgvector RAG for NDMO / PDPL compliance knowledge
  • Streamlit Chat Interface with real-time Thought Trace
  • Live Dashboard: Total Records | Compliance Score | Active Alerts
"""

from __future__ import annotations

import os, hashlib, time, io, json
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Config ────────────────────────────────────────────────────────────────────
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8765")
os.environ.setdefault("MCP_SERVER_URL", MCP_SERVER_URL)

st.set_page_config(
    page_title="LuminaData v6",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Lato:wght@300;400;700&display=swap');

:root {
  --bg:       #f8fafc;
  --surface:  #ffffff;
  --border:   #e2e8f0;
  --accent:   #0284c7;
  --accent2:  #7c3aed;
  --green:    #10b981;
  --amber:    #f59e0b;
  --red:      #ef4444;
  --text:     #1e293b;
  --muted:    #64748b;
  --font-h:   'Syne', sans-serif;
  --font-b:   'Lato', sans-serif;
  --font-m:   'JetBrains Mono', monospace;
}

.stApp { background: var(--bg); color: var(--text); font-family: var(--font-b); }
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }

/* Header */
.lumina-header {
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0f4ff 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 28px 36px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}
.lumina-header::before {
  content: '';
  position: absolute; top: -40px; right: -40px;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(2,132,199,0.08) 0%, transparent 70%);
}
.lumina-title {
  font-family: var(--font-h);
  font-size: 2rem; font-weight: 800;
  color: #0f172a; letter-spacing: -0.03em;
  margin: 0 0 4px;
}
.lumina-title span { color: var(--accent); }
.lumina-sub { font-family: var(--font-b); font-size: .9rem; color: var(--muted); margin: 0; }
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(2,132,199,0.1); border: 1px solid rgba(2,132,199,0.25);
  border-radius: 20px; padding: 3px 10px;
  font-size: .72rem; color: var(--accent); font-family: var(--font-m);
  margin: 0 4px;
}
.badge-green { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.25); color: var(--green); }
.badge-purple { background: rgba(124,58,237,0.1); border-color: rgba(124,58,237,0.25); color: #a78bfa; }

/* KPI Cards */
.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
}
.kpi-card::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0;
  height: 2px;
}
.kpi-card.blue::after  { background: var(--accent); }
.kpi-card.green::after { background: var(--green); }
.kpi-card.amber::after { background: var(--amber); }
.kpi-card.red::after   { background: var(--red); }
.kpi-label { font-family: var(--font-m); font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }
.kpi-value { font-family: var(--font-h); font-size: 2.4rem; font-weight: 800; color: #1e293b; line-height: 1; }
.kpi-value.blue  { color: var(--accent); }
.kpi-value.green { color: var(--green); }
.kpi-value.amber { color: var(--amber); }
.kpi-value.red   { color: var(--red); }
.kpi-delta { font-size: .78rem; color: var(--muted); margin-top: 4px; }

/* Alerts */
.alert-bar {
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.25);
  border-left: 4px solid var(--red);
  border-radius: 8px;
  padding: 10px 16px;
  margin: 6px 0;
  font-size: .83rem;
  color: #fca5a5;
}
.alert-bar.amber {
  background: rgba(245,158,11,0.08);
  border-color: rgba(245,158,11,0.25);
  border-left-color: var(--amber);
  color: #fcd34d;
}

/* Chat */
.chat-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
}
.chat-header {
  background: linear-gradient(90deg, var(--surface), #e0f2fe);
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
}
.chat-title { font-family: var(--font-h); font-size: .9rem; font-weight: 600; color: #1e293b; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); display: inline-block; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

.msg-user {
  background: rgba(2,132,199,0.06);
  border-left: 3px solid var(--accent);
  border-radius: 4px 12px 12px 4px;
  padding: 12px 16px;
  margin: 10px 0;
  font-size: .88rem;
  color: var(--text);
}
.msg-ai {
  background: rgba(16,185,129,0.04);
  border-left: 3px solid var(--green);
  border-radius: 4px 12px 12px 4px;
  padding: 12px 16px;
  margin: 10px 0;
  font-size: .88rem;
  color: var(--text);
}
.msg-meta {
  font-family: var(--font-m);
  font-size: .68rem;
  color: var(--muted);
  margin-bottom: 6px;
}
.intent-tag {
  display: inline-block;
  font-family: var(--font-m); font-size: .65rem;
  background: #f1f5f9; border: 1px solid #cbd5e1;
  border-radius: 4px; padding: 1px 6px; color: var(--muted);
  margin-left: 6px;
}

/* Thought trace */
.thought-step {
  font-family: var(--font-m); font-size: .78rem;
  color: #475569;
  background: rgba(0,0,0,0.04);
  border-left: 3px solid var(--accent2);
  border-radius: 4px;
  padding: 5px 10px;
  margin: 3px 0;
}

/* SQL box */
.sql-box {
  background: #f1f5f9;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  font-family: var(--font-m); font-size: .83rem;
  color: #0369a1;
  white-space: pre-wrap; word-break: break-all;
  margin: 8px 0;
}

/* Login */
.login-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 44px;
  max-width: 420px;
  margin: 0 auto;
}
.login-title { font-family: var(--font-h); font-size: 1.6rem; font-weight: 800; color: #1e293b; margin-bottom: 4px; }
.login-sub   { font-size: .85rem; color: var(--muted); margin-bottom: 24px; }

/* Sidebar */
.sidebar-section { margin-bottom: 20px; }
.sidebar-label {
  font-family: var(--font-m); font-size: .68rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: .1em; margin-bottom: 8px;
}
.chip {
  display: inline-block;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 20px; padding: 5px 12px;
  font-size: .8rem; color: var(--text);
  cursor: pointer; margin: 3px 2px;
  transition: all .15s;
}
.chip:hover { border-color: var(--accent); color: var(--accent); }

/* Score gauge label */
.score-label {
  font-family: var(--font-h); font-size: 1.1rem; font-weight: 700;
  text-align: center; margin-top: -8px;
}

/* Stacked dimension bars */
.dim-bar-wrap { margin: 6px 0; }
.dim-name { font-size: .78rem; color: var(--muted); font-family: var(--font-m); margin-bottom: 3px; }
.dim-bar-bg { background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden; }
.dim-bar-fill { height: 8px; border-radius: 4px; }

/* Override Streamlit inputs for light theme */
div[data-testid="stTextInput"] input {
  background: #ffffff !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent) !important;
}
.stButton>button {
  background: var(--accent) !important;
  color: #000 !important;
  font-family: var(--font-h) !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 8px !important;
}
.stButton>button:hover { opacity: .9; }
</style>
""", unsafe_allow_html=True)

# ── Auth ──────────────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

USERS = {
    "admin":  {"password": _hash("admin123"),  "role": "Admin"},
    "viewer": {"password": _hash("viewer123"), "role": "Viewer"},
}

def init_session():
    defaults = {
        "authenticated": False,
        "username":      None,
        "role":          None,
        "chat_history":  [],   # list of {role, content, intent, thoughts, sql}
        "dq_cache":      None,
        "dq_cache_ts":   0.0,
        "mcp_ok":        None,
        "applied_fixes":  [],
        "pending_edit":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── MCP health check (cached 30s) ─────────────────────────────────────────────
def _check_mcp() -> bool:
    try:
        import httpx
        resp = httpx.get(f"{MCP_SERVER_URL}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False

def get_mcp_status() -> bool:
    if st.session_state.mcp_ok is None:
        st.session_state.mcp_ok = _check_mcp()
    return st.session_state.mcp_ok

# ── DQ summary (cached 60s) ───────────────────────────────────────────────────
def get_dq_cached() -> dict:
    now = time.time()
    if st.session_state.dq_cache is None or (now - st.session_state.dq_cache_ts) > 60:
        try:
            from mcp_client import dq_summary
            st.session_state.dq_cache    = dq_summary()
            st.session_state.dq_cache_ts = now
        except Exception:
            st.session_state.dq_cache = {
                "total_records": 0,
                "compliance_score": 0,
                "active_alerts": ["MCP server unreachable"],
                "dimensions": {},
            }
    return st.session_state.dq_cache

# ── PDF RAG ingestion ────────────────────────────────────────────────────────
def _pdf_extract_pages(data: bytes, filename: str) -> list[dict]:
    """
    Extract text from PDF bytes using PyMuPDF.
    Raises a clear, user-friendly RuntimeError for every known failure mode.
    """
    try:
        import fitz
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is not installed.\n"
            "Fix: run  pip install pymupdf  then restart Streamlit."
        )

    # ── Try to open ───────────────────────────────────────────────────────────
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise RuntimeError(
            f"Could not open '{filename}' as a PDF.\n"
            f"The file may be corrupted or not a valid PDF.\n"
            f"Technical detail: {e}"
        )

    # ── Password-protected check ──────────────────────────────────────────────
    if doc.is_encrypted:
        doc.close()
        raise RuntimeError(
            f"'{filename}' is password-protected.\n"
            "Please remove the password (in Adobe Acrobat: File → Properties → "
            "Security → No Security) and re-upload."
        )

    # ── Extract text page by page ─────────────────────────────────────────────
    pages      = []
    empty_pages = 0
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
        else:
            empty_pages += 1
    doc.close()

    total_pages = len(pages) + empty_pages

    # ── Scanned / image-only PDF ──────────────────────────────────────────────
    if not pages:
        raise RuntimeError(
            f"'{filename}' appears to be a scanned PDF — no text layer found.\n\n"
            f"All {total_pages} page(s) contain only images.\n\n"
            "Solutions:\n"
            "  1. Use Adobe Acrobat: Edit → Make Text Searchable (OCR)\n"
            "  2. Use a free online OCR tool: smallpdf.com or ilovepdf.com\n"
            "  3. Use the open-source tool: ocrmypdf (pip install ocrmypdf)"
        )

    # ── Warn if partially scanned ─────────────────────────────────────────────
    if empty_pages > 0:
        st.warning(
            f"⚠️ {empty_pages} of {total_pages} pages had no readable text "
            f"(possibly scanned images). Only the {len(pages)} text pages will be ingested."
        )

    return pages


def ingest_pdf_bytes(filename: str, data: bytes, progress_bar=None) -> dict:
    """
    Full ingestion pipeline: extract → chunk → embed → store in pgvector.

    Parameters
    ----------
    filename    : original filename (used as source key in rag_documents)
    data        : raw PDF bytes from st.file_uploader
    progress_bar: optional st.progress object for live feedback

    Returns
    -------
    {"chunks": N, "pages": N, "preview": "first 200 chars..."}

    Raises RuntimeError with a user-friendly message on any failure.
    """
    import psycopg2
    from psycopg2.extras import execute_values

    def _progress(pct: float, label: str):
        if progress_bar is not None:
            progress_bar.progress(pct, text=label)

    # ── Step 1: Extract text ──────────────────────────────────────────────────
    _progress(0.05, "📄 Reading PDF pages...")
    pages = _pdf_extract_pages(data, filename)   # raises on failure

    # ── Step 2: Load embedding model ──────────────────────────────────────────
    _progress(0.15, "🧠 Loading embedding model (first run: ~30 seconds)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed.\n"
            "Fix: run  pip install sentence-transformers  then restart Streamlit."
        )

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    _progress(0.25, f"✂️ Splitting {len(pages)} pages into chunks...")
    CHUNK_SIZE, OVERLAP = 400, 80
    all_chunks = []
    for p in pages:
        words = p["text"].split()
        start, idx = 0, 0
        while start < len(words):
            end   = min(start + CHUNK_SIZE, len(words))
            chunk = " ".join(words[start:end])
            all_chunks.append({
                "source":  filename,
                "page":    p["page"],
                "idx":     idx,
                "content": chunk,
            })
            idx  += 1
            if end == len(words):
                break
            start += CHUNK_SIZE - OVERLAP

    if not all_chunks:
        raise RuntimeError("PDF text was found but produced no chunks — the file may be nearly empty.")

    # ── Step 4: Embed ─────────────────────────────────────────────────────────
    _progress(0.40, f"⚡ Embedding {len(all_chunks)} chunks (this takes ~10–30 seconds)...")
    try:
        texts      = [c["content"] for c in all_chunks]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

    # ── Step 5: Store in PostgreSQL ───────────────────────────────────────────
    _progress(0.80, "💾 Storing in PostgreSQL pgvector...")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5433")
    DB_NAME = os.getenv("DB_NAME", "luminadata")
    DB_USER = os.getenv("DB_USER", "lumina_user")
    DB_PASS = os.getenv("DB_PASSWORD", "lumina_pass")

    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASS,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to PostgreSQL: {e}\n"
            "Make sure Docker is running and the postgres container is healthy."
        )

    try:
        cur = conn.cursor()
        # Create table if first ever ingest
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id         SERIAL PRIMARY KEY,
                source     TEXT NOT NULL,
                page_num   INTEGER,
                chunk_idx  INTEGER,
                content    TEXT NOT NULL,
                embedding  vector(384),
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_rag_embedding
                ON rag_documents USING hnsw (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL;
        """)
        # Replace existing chunks for this file
        cur.execute("DELETE FROM rag_documents WHERE source = %s", (filename,))
        # Insert
        rows = [
            (c["source"], c["page"], c["idx"], c["content"],
             "[" + ",".join(str(float(x)) for x in emb) + "]")
            for c, emb in zip(all_chunks, embeddings)
        ]
        execute_values(cur, """
            INSERT INTO rag_documents (source, page_num, chunk_idx, content, embedding)
            VALUES %s
        """, rows)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Database insert failed: {e}")
    finally:
        cur.close()
        conn.close()

    _progress(1.0, "✅ Done!")

    # First 200 chars of page 1 as preview
    preview = pages[0]["text"][:200].replace("\n", " ") + "..."

    return {"chunks": len(rows), "pages": len(pages), "preview": preview}


def get_rag_documents() -> list[dict]:
    """Return list of ingested RAG documents from PostgreSQL."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5433"),
            dbname=os.getenv("DB_NAME", "luminadata"),
            user=os.getenv("DB_USER", "lumina_user"),
            password=os.getenv("DB_PASSWORD", "lumina_pass"),
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT source, COUNT(*) as chunks, MAX(created_at)::date as uploaded
            FROM rag_documents
            GROUP BY source ORDER BY MAX(created_at) DESC
        """)
        rows = [{"source": r[0], "chunks": r[1], "uploaded": str(r[2])}
                for r in cur.fetchall()]
        cur.close(); conn.close()
        return rows
    except Exception:
        return []


def delete_rag_document(source: str):
    """Delete all chunks for a source file."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5433"),
        dbname=os.getenv("DB_NAME", "luminadata"),
        user=os.getenv("DB_USER", "lumina_user"),
        password=os.getenv("DB_PASSWORD", "lumina_pass"),
    )
    cur = conn.cursor()
    cur.execute("DELETE FROM rag_documents WHERE source = %s", (source,))
    conn.commit(); cur.close(); conn.close()


# ── Login ─────────────────────────────────────────────────────────────────────
def render_login():
    st.markdown('<div style="text-align:center; padding: 60px 0 30px;">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'Syne\',sans-serif; font-size:2.4rem; font-weight:800; color:#0f172a; letter-spacing:-0.04em; margin:0">Lumina<span style="color:#0284c7">Data</span></p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:.9rem; margin-top:6px">Enterprise v6.0 · Agentic Data Quality Platform</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
        st.markdown('<p class="login-title">Sign In</p>', unsafe_allow_html=True)
        st.markdown('<p class="login-sub">Access the AI-powered data guardian</p>', unsafe_allow_html=True)
        uname = st.text_input("Username", placeholder="admin or viewer", label_visibility="collapsed")
        pw    = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
        if st.button("Sign In →", use_container_width=True):
            user = USERS.get(uname.strip().lower())
            if user and user["password"] == _hash(pw):
                st.session_state.authenticated = True
                st.session_state.username = uname.strip().lower()
                st.session_state.role     = user["role"]
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown('<p style="text-align:center; color:#64748b; font-size:.78rem; margin-top:16px">Demo — <b>admin</b>/admin123 &nbsp;|&nbsp; <b>viewer</b>/viewer123</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if not st.session_state.authenticated:
    render_login()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'Syne\',sans-serif; font-size:1.3rem; font-weight:800; color:#0f172a; letter-spacing:-0.03em; margin:0 0 2px">Lumina<span style="color:#0284c7">Data</span></p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:.75rem; margin:0 0 20px">Enterprise v6.0</p>', unsafe_allow_html=True)
    st.divider()

    # User info
    role_color = "#10b981" if st.session_state.role == "Admin" else "#00d4ff"
    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:12px 14px; margin-bottom:16px;">
      <p style="font-family:'JetBrains Mono',monospace; font-size:.7rem; color:#64748b; margin:0 0 4px;">SIGNED IN AS</p>
      <p style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#1e293b; margin:0">{st.session_state.username}</p>
      <span style="background:rgba(16,185,129,0.1); border:1px solid {role_color}; color:{role_color}; border-radius:20px; padding:2px 8px; font-size:.7rem; font-family:'JetBrains Mono',monospace;">{st.session_state.role}</span>
    </div>
    """, unsafe_allow_html=True)

    # MCP status
    mcp_ok = get_mcp_status()
    mcp_color = "#10b981" if mcp_ok else "#ef4444"
    mcp_text  = "MCP Online" if mcp_ok else "MCP Offline"
    st.markdown(f'<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:8px 12px; margin-bottom:16px; font-family:\'JetBrains Mono\',monospace; font-size:.72rem; color:{mcp_color};">⬤ {mcp_text}</div>', unsafe_allow_html=True)

    st.divider()

    # ── PDF Knowledge Base ────────────────────────────────────────────────────
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace; font-size:.68rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px;">📚 RAG Knowledge Base</p>', unsafe_allow_html=True)

    if st.session_state.role == "Admin":
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="PDF will be chunked, embedded, and stored in PostgreSQL for RAG queries.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_name  = uploaded_file.name
            if st.button(f"Ingest '{file_name}'", use_container_width=True, type="primary"):
                progress_bar = st.progress(0, text="Starting...")
                try:
                    result = ingest_pdf_bytes(file_name, file_bytes, progress_bar)
                    progress_bar.empty()
                    st.success(
                        f"✅ Done! {result['chunks']} chunks from "
                        f"{result['pages']} pages stored in PostgreSQL."
                    )
                    # Show text preview so user can confirm it was read correctly
                    with st.expander("📖 Text preview (first page)", expanded=False):
                        st.caption(result["preview"])
                except RuntimeError as e:
                    progress_bar.empty()
                    st.error(str(e))
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Unexpected error: {e}")
    else:
        st.caption("PDF upload requires Admin role.")

    # Show ingested documents
    rag_docs = get_rag_documents()
    if rag_docs:
        st.markdown(
            f'<p style="font-size:.72rem; color:#64748b; margin:8px 0 4px;">'
            f'{len(rag_docs)} document(s) in knowledge base</p>',
            unsafe_allow_html=True,
        )
        for doc in rag_docs:
            doc_col, del_col = st.columns([3, 1])
            with doc_col:
                st.markdown(
                    f'<div style="background:#f0fdf4; border:1px solid #bbf7d0; '
                    f'border-radius:6px; padding:6px 10px; font-size:.75rem; '
                    f'color:#166534; margin:2px 0;">'
                    f'📄 {doc["source"]}<br>'
                    f'<span style="color:#64748b">{doc["chunks"]} chunks · {doc["uploaded"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with del_col:
                if st.session_state.role == "Admin":
                    if st.button("🗑", key=f"del_{doc['source']}", help=f"Delete {doc['source']}"):
                        try:
                            delete_rag_document(doc["source"])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
    else:
        st.markdown(
            '<p style="font-size:.75rem; color:#94a3b8; margin:4px 0 0;">'
            'No documents yet. Upload a PDF above to enable RAG.</p>',
            unsafe_allow_html=True,
        )
    st.divider()


    # Quick prompts
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace; font-size:.68rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px;">Quick Prompts</p>', unsafe_allow_html=True)
    quick_prompts = [
        "Run NDMO full compliance audit",
        "Show all citizens with missing National IDs",
        "How many expired ID documents are there?",
        "Show city/region inconsistencies",
        "What is the overall DQ score?",
        "List citizens missing phone or email",
        "Check for duplicate national IDs",
    ]
    for qp in quick_prompts:
        if st.button(qp, key=f"qp_{qp[:20]}", use_container_width=True):
            st.session_state["pending_prompt"] = qp

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if st.button("Sign Out", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lumina-header">
  <p class="lumina-title">Lumina<span>Data</span> Enterprise</p>
  <p class="lumina-sub">AI-Powered Citizen Data Guardian · LangGraph · MCP · pgvector RAG</p>
  <div style="margin-top:14px">
    <span class="badge">⚡ LangGraph Orchestrator</span>
    <span class="badge badge-green">🛡️ NDMO DQ Agent</span>
    <span class="badge badge-purple">🔍 SQL Agent</span>
    <span class="badge">📚 pgvector RAG</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Chat Interface ────────────────────────────────────────────────────────────
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
st.markdown("""
<div class="chat-header">
  <span class="dot"></span>
  <span class="chat-title">LuminaData AI Assistant</span>
  <span style="font-family:'JetBrains Mono',monospace; font-size:.7rem; color:#94a3b8; margin-left:auto;">LangGraph · MCP · RAG</span>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Render chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px; color: #64748b;">
          <p style="font-size:2rem; margin-bottom:8px">🛡️</p>
          <p style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#475569;">How can I help?</p>
          <p style="font-size:.85rem;">Ask me to query the citizen database, run a NDMO compliance audit, or explain any feature.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                  <p class="msg-meta">👤 {st.session_state.username} &nbsp;·&nbsp; {msg.get('ts','')} <span class="intent-tag">{msg.get('intent','')}</span></p>
                  {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # AI message with collapsible thought trace
                st.markdown(f"""
                <div class="msg-ai">
                  <p class="msg-meta">🤖 LuminaData Agent &nbsp;·&nbsp; {msg.get('ts','')}</p>
                """, unsafe_allow_html=True)

                # Thought trace
                if msg.get("thoughts"):
                    with st.expander("🔍 Thought Trace", expanded=False):
                        for t in msg["thoughts"]:
                            st.markdown(f'<div class="thought-step">{t}</div>', unsafe_allow_html=True)

                # SQL executed
                if msg.get("sql"):
                    with st.expander("📄 SQL Executed", expanded=False):
                        st.markdown(f'<div class="sql-box">{msg["sql"]}</div>', unsafe_allow_html=True)

                # Response
                st.markdown(msg["content"])
                st.markdown("</div>", unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────────────────────────────────────
# Handle pending prompt from sidebar chips
pending = st.session_state.pop("pending_prompt", None)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Message",
        value=pending or "",
        placeholder="Ask about citizen data, NDMO compliance, or anything about LuminaData...",
        label_visibility="collapsed",
    )
    submit = st.form_submit_button("Send →", use_container_width=False)

if submit and user_input.strip():
    query = user_input.strip()
    ts    = datetime.now().strftime("%H:%M")

    # Add user message to history
    st.session_state.chat_history.append({
        "role":    "user",
        "content": query,
        "ts":      ts,
        "intent":  "...",
    })

    # Run the LangGraph pipeline with live thought trace
    with st.status("🧠 Thinking...", expanded=True) as status:
        st.write("Invoking LangGraph Orchestrator...")
        try:
            from orchestration import run_agent
            result = run_agent(
                user_message=query,
                username=st.session_state.username,
                user_role=st.session_state.role,
            )

            for thought in result.get("thoughts", []):
                st.write(thought)

            status.update(label="✅ Done", state="complete", expanded=False)

            intent   = result.get("intent",       "GENERAL_INFO")
            response = result.get("response",     "No response generated.")
            thoughts = result.get("thoughts",     [])
            sql      = result.get("sql_executed", None)
            dq_rep   = result.get("dq_report",    None)

            # Update user message intent label
            st.session_state.chat_history[-1]["intent"] = intent

            rag_sources = result.get("rag_sources", None)
            # Add AI response
            st.session_state.chat_history.append({
                "role":        "assistant",
                "content":     response,
                "ts":          datetime.now().strftime("%H:%M"),
                "intent":      intent,
                "thoughts":    thoughts,
                "sql":         sql,
                "rag_sources": rag_sources,
            })

            # Refresh DQ cache if compliance audit was run
            if intent == "NDMO_COMPLIANCE_CHECK" and dq_rep:
                st.session_state.dq_cache    = dq_rep
                st.session_state.dq_cache_ts = time.time()

        except Exception as exc:
            status.update(label=f"❌ Error: {exc}", state="error")
            st.session_state.chat_history.append({
                "role":     "assistant",
                "content":  f"❌ **Agent Error:** {exc}\n\nCheck that the MCP Server is running at `{MCP_SERVER_URL}`.",
                "ts":       datetime.now().strftime("%H:%M"),
                "intent":   "ERROR",
                "thoughts": [],
                "sql":      None,
            })

    st.rerun()

# ── Agent Architecture Diagram ────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Architecture Diagram ── */
.arch-wrap {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 28px 32px 24px;
  margin-bottom: 24px;
  font-family: 'Lato', sans-serif;
}
.arch-section-title {
  font-family: 'Syne', sans-serif;
  font-size: .72rem; font-weight: 700;
  color: #94a3b8; text-transform: uppercase;
  letter-spacing: .12em; margin: 0 0 18px;
  text-align: center;
}

/* Layers */
.arch-layer {
  display: flex;
  justify-content: center;
  gap: 14px;
  margin: 0;
}
.arch-connector {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 28px;
  color: #cbd5e1;
  font-size: 1.1rem;
  letter-spacing: 2px;
  margin: 2px 0;
}
.arch-connector-wide {
  display: flex;
  justify-content: space-around;
  align-items: center;
  height: 28px;
  margin: 2px 0;
  padding: 0 12%;
  color: #cbd5e1;
  font-size: 1.1rem;
  letter-spacing: 2px;
}

/* Base card */
.arch-card {
  border-radius: 12px;
  padding: 14px 16px;
  border: 1px solid transparent;
  display: flex;
  flex-direction: column;
  gap: 5px;
  min-width: 170px;
  position: relative;
}
.arch-icon {
  font-size: 1.3rem;
  margin-bottom: 2px;
}
.arch-card-title {
  font-family: 'Syne', sans-serif;
  font-size: .82rem; font-weight: 700;
  color: #1e293b;
  line-height: 1.2;
}
.arch-card-sub {
  font-size: .72rem;
  color: #64748b;
  line-height: 1.3;
}
.arch-port-badge {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: .63rem;
  background: rgba(0,0,0,0.05);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 4px;
  padding: 1px 5px;
  color: #475569;
  margin-top: 2px;
  width: fit-content;
}

/* Intent tags on orchestrator */
.arch-intents {
  display: flex; flex-wrap: wrap; gap: 4px;
  margin-top: 4px;
}
.arch-intent {
  font-family: 'JetBrains Mono', monospace;
  font-size: .6rem;
  background: rgba(2,132,199,0.08);
  border: 1px solid rgba(2,132,199,0.2);
  border-radius: 4px;
  padding: 2px 6px;
  color: #0284c7;
}

/* Tool chips inside agent cards */
.arch-tool-list { display: flex; flex-direction: column; gap: 3px; margin-top: 4px; }
.arch-tool-chip {
  font-family: 'JetBrains Mono', monospace;
  font-size: .6rem;
  border-radius: 4px;
  padding: 2px 6px;
  width: fit-content;
}

/* MCP tool grid */
.arch-mcp-tools {
  display: flex; flex-wrap: wrap; gap: 4px;
  margin-top: 6px;
}
.arch-mcp-chip {
  font-family: 'JetBrains Mono', monospace;
  font-size: .6rem;
  background: rgba(180,83,9,0.07);
  border: 1px solid rgba(180,83,9,0.18);
  border-radius: 4px;
  padding: 2px 6px;
  color: #92400e;
}

/* Card colour themes */
.arch-c-input      { background:#f8fafc; border-color:#e2e8f0; }
.arch-c-orch       { background:#eff6ff; border-color:#bfdbfe; }
.arch-c-sql        { background:#f0fdfa; border-color:#99f6e4; }
.arch-c-NDMO      { background:#f0fdf4; border-color:#bbf7d0; }
.arch-c-general    { background:#faf5ff; border-color:#e9d5ff; }
.arch-c-mcp        { background:#fffbeb; border-color:#fde68a; }
.arch-c-pg         { background:#f0f9ff; border-color:#bae6fd; }
.arch-c-danswer    { background:#eef2ff; border-color:#c7d2fe; }
.arch-c-falkor     { background:#fff1f2; border-color:#fecdd3; }

.arch-c-sql    .arch-tool-chip { background:rgba(8,145,178,0.07); border:1px solid rgba(8,145,178,0.2); color:#0e7490; }
.arch-c-NDMO  .arch-tool-chip { background:rgba(5,150,105,0.07); border:1px solid rgba(5,150,105,0.2); color:#059669; }
.arch-c-general.arch-tool-chip { background:rgba(124,58,237,0.07); border:1px solid rgba(124,58,237,0.2); color:#7c3aed; }
.arch-c-general .arch-tool-chip{ background:rgba(124,58,237,0.07); border:1px solid rgba(124,58,237,0.2); color:#7c3aed; }
</style>

<div class="arch-wrap">
  <p class="arch-section-title">⚙️ Multi-Agent Architecture</p>
  <!-- Layer 1: User -->
  <div class="arch-layer">
    <div class="arch-card arch-c-input" style="min-width:140px; align-items:center; text-align:center;">
      <span class="arch-icon">👤</span>
      <span class="arch-card-title">User Input</span>
      <span class="arch-card-sub">Natural language query</span>
    </div>
  </div>
  <div class="arch-connector">↓</div>
  <!-- Layer 2: Orchestrator -->
  <div class="arch-layer">
    <div class="arch-card arch-c-orch" style="min-width:420px;">
      <span class="arch-icon">⚡</span>
      <span class="arch-card-title">Master Orchestrator</span>
      <span class="arch-card-sub">LangGraph · Classifies intent and routes to the correct specialist agent</span>
      <div class="arch-intents">
        <span class="arch-intent">SQL_QUERY</span>
        <span class="arch-intent">NDMO_COMPLIANCE_CHECK</span>
        <span class="arch-intent">GENERAL_INFO</span>
        <span class="arch-intent">ERROR</span>
      </div>
    </div>
  </div>
  <div class="arch-connector-wide">↙&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘</div>
  <!-- Layer 3: Agents -->
  <div class="arch-layer">
    <div class="arch-card arch-c-sql">
      <span class="arch-icon">🔍</span>
      <span class="arch-card-title">SQL Agent</span>
      <span class="arch-card-sub">Natural Language → SQL · Queries citizen registry</span>
      <div class="arch-tool-list">
        <span class="arch-tool-chip">🔧 execute_sql_query</span>
        <span class="arch-tool-chip">🔧 execute_write_sql</span>
        <span class="arch-tool-chip">🔧 get_table_schema</span>
        <span class="arch-tool-chip">🔧 list_public_tables</span>
      </div>
    </div>
    <div class="arch-card arch-c-NDMO">
      <span class="arch-icon">🛡️</span>
      <span class="arch-card-title">NDMO DQ Agent</span>
      <span class="arch-card-sub">Compliance audit &amp; RAG · NDMO / PDPL standards</span>
      <div class="arch-tool-list">
        <span class="arch-tool-chip">🔧 get_dq_summary</span>
        <span class="arch-tool-chip">🔧 query_NDMO_knowledge_base</span>
      </div>
    </div>
    <div class="arch-card arch-c-general">
      <span class="arch-icon">📚</span>
      <span class="arch-card-title">General Info Agent</span>
      <span class="arch-card-sub">Platform guidance &amp; docs · RAG-backed Q&amp;A</span>
      <div class="arch-tool-list">
        <span class="arch-tool-chip">🔧 query_NDMO_knowledge_base</span>
        <span class="arch-tool-chip">🔧 get_dq_summary</span>
      </div>
    </div>
  </div>
  <div class="arch-connector">↓</div>
  <!-- Layer 4: MCP Server -->
  <div class="arch-layer">
    <div class="arch-card arch-c-mcp" style="min-width:480px;">
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="arch-icon">🔌</span>
        <span class="arch-card-title">MCP Server</span>
        <span class="arch-port-badge">:8765</span>
      </div>
      <span class="arch-card-sub">Model Context Protocol · Unified tool gateway — agents never call DBs directly</span>
      <div class="arch-mcp-tools">
        <span class="arch-mcp-chip">execute_sql_query</span>
        <span class="arch-mcp-chip">execute_write_sql</span>
        <span class="arch-mcp-chip">get_table_schema</span>
        <span class="arch-mcp-chip">list_public_tables</span>
        <span class="arch-mcp-chip">query_NDMO_knowledge_base</span>
        <span class="arch-mcp-chip">get_dq_summary</span>
      </div>
    </div>
  </div>
  <div class="arch-connector-wide">↙&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘</div>
  <!-- Layer 5: Data Sources -->
  <div class="arch-layer">
    <div class="arch-card arch-c-pg" style="align-items:center; text-align:center;">
      <span class="arch-icon">🐘</span>
      <span class="arch-card-title">PostgreSQL</span>
      <span class="arch-card-sub">Citizen Registry DB</span>
      <span class="arch-port-badge">port 5432</span>
    </div>
    <div class="arch-card arch-c-danswer" style="align-items:center; text-align:center;">
      <span class="arch-icon">📖</span>
      <span class="arch-card-title">pgvector RAG</span>
      <span class="arch-card-sub">NDMO / PDPL Docs · all-MiniLM-L6-v2</span>
      <span class="arch-port-badge">rag_documents table</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Live Dashboard ────────────────────────────────────────────────────────────
dq = get_dq_cached()
total_records    = dq.get("total_records",    0)
compliance_score = dq.get("compliance_score", 0)
active_alerts    = dq.get("active_alerts",    [])
dims             = dq.get("dimensions",       {})

# Score colour
if compliance_score >= 85:
    score_cls = "green"
elif compliance_score >= 65:
    score_cls = "amber"
else:
    score_cls = "red"

alert_cls = "red" if len(active_alerts) >= 3 else ("amber" if active_alerts else "green")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="kpi-card blue">
      <p class="kpi-label">Total Records</p>
      <p class="kpi-value blue">{total_records:,}</p>
      <p class="kpi-delta">Citizens in registry</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="kpi-card {score_cls}">
      <p class="kpi-label">Compliance Score</p>
      <p class="kpi-value {score_cls}">{compliance_score}%</p>
      <p class="kpi-delta">NDMO DQ standard</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="kpi-card {alert_cls}">
      <p class="kpi-label">Active Alerts</p>
      <p class="kpi-value {alert_cls}">{len(active_alerts)}</p>
      <p class="kpi-delta">Requiring attention</p>
    </div>""", unsafe_allow_html=True)

# Active alert banners
if active_alerts:
    with st.expander(f"🚨 {len(active_alerts)} Active Alert(s)", expanded=False):
        for alert in active_alerts:
            cls = "amber" if "expir" in alert.lower() else ""
            st.markdown(f'<div class="alert-bar {cls}">⚠️ {alert}</div>', unsafe_allow_html=True)

# DQ dimension mini-bars
if dims:
    completeness = dims.get("completeness", {})
    avg_completeness = sum(completeness.values()) / max(len(completeness), 1) if completeness else 100
    validity    = dims.get("validity",    {})
    timeliness  = dims.get("timeliness", {})
    consistency = dims.get("consistency",{})

    total = max(total_records, 1)
    validity_issues = sum(validity.values())
    validity_score  = max(0, round(100 - (validity_issues / total) * 100))
    timeliness_score = max(0, round(100 - (timeliness.get("expired_ids", 0) / total) * 100))
    consistency_score = max(0, round(100 - (consistency.get("inconsistent_city_region", 0) / total) * 100))

    dim_data = [
        ("Completeness", avg_completeness),
        ("Validity",     validity_score),
        ("Timeliness",   timeliness_score),
        ("Consistency",  consistency_score),
    ]

    with st.expander("📊 DQ Dimensions Breakdown", expanded=False):
        bar_cols = st.columns(4)
        for i, (name, score) in enumerate(dim_data):
            clr = "#10b981" if score >= 85 else ("#f59e0b" if score >= 65 else "#ef4444")
            with bar_cols[i]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": name, "font": {"size": 11, "color": "#64748b", "family": "JetBrains Mono"}},
                    number={"suffix": "%", "font": {"size": 22, "color": clr, "family": "Syne"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#e2e8f0"},
                        "bar":  {"color": clr, "thickness": 0.3},
                        "bgcolor": "#f8fafc",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 65],  "color": "rgba(239,68,68,0.08)"},
                            {"range": [65, 85], "color": "rgba(245,158,11,0.08)"},
                            {"range": [85, 100],"color": "rgba(16,185,129,0.08)"},
                        ],
                    },
                ))
                fig.update_layout(
                    height=160, margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})