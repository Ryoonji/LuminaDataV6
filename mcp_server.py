# -*- coding: utf-8 -*-
"""
LuminaData Enterprise v6.0 — MCP Server
========================================
Centralised Model Context Protocol layer.
All agents call tools through this server — never touching the DB directly.

Tools exposed:
  • execute_sql_query          – safe read-only SQL on PostgreSQL
  • execute_write_sql          – write SQL (admin only, allowlist enforced)
  • get_table_schema           – column metadata for a given table
  • list_public_tables         – discover all tables in public schema
  • query_NDMO_knowledge_base – pgvector RAG over uploaded PDF documents
  • get_dq_summary             – compute live data-quality metrics

Transport: HTTP/JSON  (served via FastAPI on port 8765)
Run independently:  python mcp_server.py
"""

from __future__ import annotations

import os, re, logging
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sqlalchemy as sa
from sqlalchemy import pool, text
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

# ── Config ────────────────────────────────────────────────────────────────────
DB_HOST      = os.getenv("DB_HOST",      "postgres")
DB_PORT      = os.getenv("DB_PORT",      "5432")
DB_NAME      = os.getenv("DB_NAME",      "luminadata")
DB_USER      = os.getenv("DB_USER",      "lumina_user")
DB_PASSWORD  = os.getenv("DB_PASSWORD",  "lumina_pass")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
MCP_SECRET   = os.getenv("MCP_SECRET",   "lumina_mcp_secret_2024")

# SQL safety — block DDL / DML in the read-only tool
_FORBIDDEN_READ = frozenset({
    "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE",
    "GRANT", "REVOKE", "INSERT", "UPDATE", "REPLACE", "COPY",
})

# Write tool — only approved UPDATE corrections on citizens
_ALLOWED_WRITE_PATTERNS = [
    r"^\s*UPDATE\s+citizens\s+SET\s+.+WHERE\s+id\s*=\s*\d+",
]

# ── Lazy embedding model (loaded once on first RAG query) ─────────────────────
_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded: all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning("sentence-transformers unavailable: %s", exc)
    return _embed_model


# ── DB Engine ─────────────────────────────────────────────────────────────────
_engine: sa.engine.Engine | None = None

def get_engine() -> sa.engine.Engine:
    global _engine
    if _engine is None:
        _engine = sa.create_engine(
            DATABASE_URL,
            poolclass=pool.QueuePool,
            pool_size=5, max_overflow=10,
            pool_timeout=30, pool_recycle=1800,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10},
        )
    return _engine


def _run_read_sql(sql: str, params: dict | None = None) -> list[dict]:
    eng = get_engine()
    with eng.connect() as conn:
        result = conn.execute(text(sql), params or {})
        cols   = list(result.keys())
        return [dict(zip(cols, row)) for row in result.fetchall()]


def _run_write_sql(sql: str) -> dict:
    eng = get_engine()
    with eng.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    return {"status": "ok", "sql": sql}


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LuminaData MCP Server",
    description="Model Context Protocol layer for LuminaData Enterprise v6.0",
    version="6.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Auth ──────────────────────────────────────────────────────────────────────
def verify_token(x_mcp_secret: str = Header(...)):
    if x_mcp_secret != MCP_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorised MCP call")


# ── Request models ────────────────────────────────────────────────────────────
class SqlRequest(BaseModel):
    sql:    str
    params: dict[str, Any] = Field(default_factory=dict)

class WriteRequest(BaseModel):
    sql: str

class WriteWithAuditRequest(BaseModel):
    sql:            str
    username:       str = "admin"
    user_role:      str = "Admin"
    agent_name:     str = "DQ_Agent"
    action_type:    str = "AUTO_FIX_APPLIED"
    original_issue: str = ""

class SchemaRequest(BaseModel):
    table_name: str

class RagRequest(BaseModel):
    query:       str
    num_results: int = Field(default=5, ge=1, le=20)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        _run_read_sql("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok", "db": db_ok, "version": "6.0.0"}


# ── Tool: execute_sql_query ───────────────────────────────────────────────────
@app.post("/tools/execute_sql_query", dependencies=[Depends(verify_token)])
def execute_sql_query(req: SqlRequest):
    """Safe read-only SQL. Blocks DDL/DML keywords."""
    first_token = req.sql.strip().split()[0].upper() if req.sql.strip() else ""
    if first_token in _FORBIDDEN_READ:
        raise HTTPException(status_code=400,
            detail=f"SQL keyword '{first_token}' is not permitted in read queries.")
    try:
        rows = _run_read_sql(req.sql, req.params)
        return {"status": "ok", "rows": rows, "count": len(rows)}
    except Exception as exc:
        logger.error("SQL error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Tool: execute_write_sql ───────────────────────────────────────────────────
@app.post("/tools/execute_write_sql", dependencies=[Depends(verify_token)])
def execute_write_sql(req: WriteWithAuditRequest):
    """
    Write SQL for approved corrections only (UPDATE citizens WHERE id=N).
    Automatically logs the action to audit_logs table after success.
    """
    clean   = req.sql.strip()
    allowed = any(re.match(p, clean, re.I | re.S) for p in _ALLOWED_WRITE_PATTERNS)
    if not allowed:
        raise HTTPException(status_code=403,
            detail="Only approved UPDATE corrections on citizens are permitted.")
    try:
        result = _run_write_sql(clean)
        # ── Write audit log entry ──────────────────────────────────────────────
        try:
            _run_write_sql(
                f"INSERT INTO audit_logs "
                f"(user_name, user_role, agent_name, action_type, original_issue, executed_sql) "
                f"VALUES ("
                f"  '{req.username.replace(chr(39), '')}', "
                f"  '{req.user_role.replace(chr(39), '')}', "
                f"  '{req.agent_name.replace(chr(39), '')}', "
                f"  '{req.action_type.replace(chr(39), '')}', "
                f"  $${req.original_issue.replace('$$','')}$$, "
                f"  $${clean.replace('$$','')}$$"
                f")"
            )
        except Exception as audit_err:
            logger.warning("Audit log insert failed (non-fatal): %s", audit_err)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Write SQL error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Tool: insert_audit_log ────────────────────────────────────────────────────
class AuditLogRequest(BaseModel):
    username:       str
    user_role:      str = "Admin"
    agent_name:     str = "System"
    action_type:    str
    original_issue: str
    executed_sql:   str = ""
    mcp_tool_used:  str = ""

@app.post("/tools/insert_audit_log", dependencies=[Depends(verify_token)])
def insert_audit_log(req: AuditLogRequest):
    """Insert a row into audit_logs directly (for agent-sourced events)."""
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO audit_logs "
                    "(user_name, user_role, agent_name, action_type, original_issue, executed_sql, mcp_tool_used) "
                    "VALUES (:u, :r, :a, :t, :i, :s, :m)"
                ),
                {
                    "u": req.username[:50],
                    "r": req.user_role[:20],
                    "a": req.agent_name[:50],
                    "t": req.action_type[:50],
                    "i": req.original_issue,
                    "s": req.executed_sql,
                    "m": req.mcp_tool_used[:100],
                }
            )
            conn.commit()
        return {"status": "ok"}
    except Exception as exc:
        logger.error("Audit log insert error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Tool: get_table_schema ────────────────────────────────────────────────────
@app.post("/tools/get_table_schema", dependencies=[Depends(verify_token)])
def get_table_schema(req: SchemaRequest):
    """Return column metadata for a table."""
    try:
        rows = _run_read_sql(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name   = :tname
              AND table_schema = 'public'
            ORDER BY ordinal_position
            """,
            {"tname": req.table_name},
        )
        if not rows:
            raise HTTPException(status_code=404,
                detail=f"Table '{req.table_name}' not found.")
        return {"status": "ok", "table": req.table_name, "columns": rows}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Tool: list_public_tables ──────────────────────────────────────────────────
@app.get("/tools/list_public_tables", dependencies=[Depends(verify_token)])
def list_public_tables():
    """Return all base tables in the public schema."""
    try:
        rows = _run_read_sql(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type   = 'BASE TABLE'
            ORDER BY table_name
            """
        )
        return {"status": "ok", "tables": [r["table_name"] for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Tool: query_NDMO_knowledge_base ─────────────────────────────────────────
@app.post("/tools/query_NDMO_knowledge_base", dependencies=[Depends(verify_token)])
def query_NDMO_knowledge_base(req: RagRequest):
    """
    RAG search over NDMO/PDPL documents stored in PostgreSQL pgvector.

    Flow:
      1. Embed query with all-MiniLM-L6-v2 (local, no API key needed)
      2. Cosine similarity search against rag_documents table
      3. Return top-N chunks with source, page, and similarity score

    Falls back to built-in NDMO articles when:
      - No PDFs have been uploaded yet (rag_documents is empty)
      - sentence-transformers is not installed
    """
    model = _get_embed_model()

    if model is not None:
        try:
            query_embedding = model.encode(req.query).tolist()
            embedding_str   = "[" + ",".join(str(x) for x in query_embedding) + "]"

            rows = _run_read_sql(
                """
                SELECT
                    source,
                    page_num,
                    content,
                    1 - (embedding <=> :emb::vector) AS similarity
                FROM rag_documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :emb::vector
                LIMIT :k
                """,
                {"emb": embedding_str, "k": req.num_results},
            )

            if rows:
                logger.info("pgvector RAG: returned %d chunks", len(rows))
                return {
                    "status":  "ok",
                    "source":  "pgvector_rag",
                    "results": [
                        {
                            "title":   f"{r['source']} — page {r['page_num']}",
                            "content": r["content"],
                            "score":   round(float(r["similarity"]), 4),
                            "link":    "",
                        }
                        for r in rows
                    ],
                }
            else:
                logger.info("rag_documents is empty — using static fallback")

        except Exception as exc:
            logger.warning("pgvector RAG error: %s — using static fallback", exc)

    return _NDMO_static_fallback(req.query)


# ── Static fallback ───────────────────────────────────────────────────────────
def _NDMO_static_fallback(query: str) -> dict:
    """
    Built-in NDMO/PDPL articles used before any PDF is uploaded.
    Covers all 6 data quality dimensions.
    """
    NDMO_ARTICLES = [
        {
            "title":   "NDMO PDPL Article 5 — Data Accuracy",
            "content": (
                "Personal data must be accurate and, where necessary, kept up to date. "
                "Every reasonable step must be taken to ensure that inaccurate personal "
                "data is erased or rectified without delay. Expired identity documents "
                "constitute inaccurate data under this article."
            ),
            "score": 0.95,
        },
        {
            "title":   "NDMO NDMO DQ Pillar — Completeness",
            "content": (
                "All mandatory citizen registry fields (National ID, Date of Birth, "
                "Phone, Email) must be populated. NULL values in mandatory fields "
                "are a completeness violation. Target completeness: ≥ 98%."
            ),
            "score": 0.92,
        },
        {
            "title":   "NDMO PDPL Article 12 — Data Validity",
            "content": (
                "Saudi National IDs must conform to the format: 10 digits beginning "
                "with '1'. Phone numbers must match 05XXXXXXXX (10 digits, Saudi mobile). "
                "Invalid formats are a validity violation requiring correction."
            ),
            "score": 0.90,
        },
        {
            "title":   "NDMO NDMO DQ Pillar — Consistency",
            "content": (
                "City and region fields must be consistent. Riyadh city → Riyadh region; "
                "Jeddah / Mecca → Makkah region; Dammam / Khobar → Eastern region; "
                "Medina → Madinah region; Abha → Asir region."
            ),
            "score": 0.88,
        },
        {
            "title":   "NDMO PDPL Article 8 — Timeliness",
            "content": (
                "Identity documents must be current. Records with id_expiry_date < "
                "CURRENT_DATE are a timeliness violation. Documents expiring within "
                "30 days must be flagged for renewal."
            ),
            "score": 0.85,
        },
        {
            "title":   "NDMO NDMO DQ Pillar — Uniqueness",
            "content": (
                "Each citizen must have a unique national_id. Duplicate national IDs "
                "are a critical uniqueness violation that must be resolved immediately."
            ),
            "score": 0.93,
        },
    ]
    q      = query.lower()
    scored = [
        {**a, "score": a["score"] + (
            0.05 if any(w in q for w in a["title"].lower().split()) else 0
        )}
        for a in NDMO_ARTICLES
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"status": "ok", "source": "static_fallback", "results": scored[:5]}


# ── Tool: get_dq_summary ──────────────────────────────────────────────────────
@app.get("/tools/get_dq_summary", dependencies=[Depends(verify_token)])
def get_dq_summary():
    """Compute live data-quality KPIs from PostgreSQL."""
    try:
        rows = _run_read_sql(
            """
            WITH base AS (SELECT COUNT(*) AS total FROM citizens),
            completeness AS (
                SELECT
                    ROUND(COUNT(national_id)   * 100.0 / NULLIF(COUNT(*),0), 1) AS national_id_pct,
                    ROUND(COUNT(phone_number)  * 100.0 / NULLIF(COUNT(*),0), 1) AS phone_pct,
                    ROUND(COUNT(email)         * 100.0 / NULLIF(COUNT(*),0), 1) AS email_pct,
                    ROUND(COUNT(date_of_birth) * 100.0 / NULLIF(COUNT(*),0), 1) AS dob_pct
                FROM citizens
            ),
            validity AS (
                SELECT
                    COUNT(*) FILTER (WHERE national_id IS NOT NULL
                        AND national_id !~ '^1[0-9]{9}$')       AS invalid_ids,
                    COUNT(*) FILTER (WHERE phone_number IS NOT NULL
                        AND phone_number !~ '^05[0-9]{8}$')      AS invalid_phones,
                    COUNT(*) FILTER (WHERE id_expiry_date < id_issue_date) AS date_logic_errors
                FROM citizens
            ),
            timeliness AS (
                SELECT COUNT(*) FILTER (WHERE id_expiry_date < CURRENT_DATE) AS expired_ids
                FROM citizens
            ),
            uniqueness AS (
                SELECT COUNT(*) AS duplicate_ids
                FROM (
                    SELECT national_id FROM citizens
                    WHERE national_id IS NOT NULL
                    GROUP BY national_id HAVING COUNT(*) > 1
                ) dup
            ),
            consistency AS (
                SELECT COUNT(*) AS inconsistent_city_region
                FROM citizens
                WHERE (city='Riyadh' AND region!='Riyadh')
                   OR (city='Jeddah' AND region!='Makkah')
                   OR (city='Mecca'  AND region!='Makkah')
                   OR (city='Dammam' AND region!='Eastern')
                   OR (city='Khobar' AND region!='Eastern')
                   OR (city='Medina' AND region!='Madinah')
                   OR (city='Abha'   AND region!='Asir')
            )
            SELECT
                base.total,
                completeness.*,
                validity.*,
                timeliness.*,
                uniqueness.*,
                consistency.*
            FROM base, completeness, validity, timeliness, uniqueness, consistency
            """
        )
        if not rows:
            raise HTTPException(status_code=500, detail="DQ query returned no rows.")

        r   = rows[0]
        tot = max(int(r.get("total", 1)), 1)

        issues      = sum([
            int(r.get("invalid_ids",              0)),
            int(r.get("invalid_phones",           0)),
            int(r.get("date_logic_errors",        0)),
            int(r.get("expired_ids",              0)),
            int(r.get("duplicate_ids",            0)),
            int(r.get("inconsistent_city_region", 0)),
        ])
        null_ids    = round(tot * (1 - float(r.get("national_id_pct", 100)) / 100))
        null_phones = round(tot * (1 - float(r.get("phone_pct",       100)) / 100))
        null_emails = round(tot * (1 - float(r.get("email_pct",       100)) / 100))
        issues     += null_ids + null_phones + null_emails

        compliance_score = max(0, round(100 - (min(issues, tot) / tot) * 100, 1))

        active_alerts = []
        if int(r.get("expired_ids",              0)) > 0:
            active_alerts.append(f"{r['expired_ids']} expired ID documents")
        if int(r.get("invalid_ids",              0)) > 0:
            active_alerts.append(f"{r['invalid_ids']} invalid National ID formats")
        if int(r.get("duplicate_ids",            0)) > 0:
            active_alerts.append(f"{r['duplicate_ids']} duplicate National IDs")
        if int(r.get("date_logic_errors",        0)) > 0:
            active_alerts.append(f"{r['date_logic_errors']} expiry < issue date errors")
        if int(r.get("inconsistent_city_region", 0)) > 0:
            active_alerts.append(f"{r['inconsistent_city_region']} city/region mismatches")
        if null_ids > 0:
            active_alerts.append(f"{null_ids} missing National IDs")

        return {
            "status":           "ok",
            "total_records":    tot,
            "compliance_score": compliance_score,
            "active_alerts":    active_alerts,
            "dimensions": {
                "completeness": {
                    "national_id_pct": float(r.get("national_id_pct", 0)),
                    "phone_pct":       float(r.get("phone_pct",       0)),
                    "email_pct":       float(r.get("email_pct",       0)),
                    "dob_pct":         float(r.get("dob_pct",         0)),
                },
                "validity": {
                    "invalid_ids":       int(r.get("invalid_ids",       0)),
                    "invalid_phones":    int(r.get("invalid_phones",    0)),
                    "date_logic_errors": int(r.get("date_logic_errors", 0)),
                },
                "timeliness": {
                    "expired_ids": int(r.get("expired_ids", 0)),
                },
                "uniqueness": {
                    "duplicate_ids": int(r.get("duplicate_ids", 0)),
                },
                "consistency": {
                    "inconsistent_city_region": int(r.get("inconsistent_city_region", 0)),
                },
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("DQ summary error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8765, reload=False)