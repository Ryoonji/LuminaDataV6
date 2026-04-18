# -*- coding: utf-8 -*-
"""
LuminaData Enterprise v6.0 — MCP Client
========================================
Thin HTTP client that all agents use to call MCP Server tools.
Agents NEVER touch the database or external APIs directly.

Usage:
    from mcp_client import call_tool, MCPError

    result = call_tool("execute_sql_query", {"sql": "SELECT COUNT(*) FROM citizens"})
    rows   = result["rows"]
"""

from __future__ import annotations

import os, logging
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger("mcp_client")

MCP_BASE_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8765")
MCP_SECRET   = os.getenv("MCP_SECRET",     "lumina_mcp_secret_2024")
_TIMEOUT     = httpx.Timeout(30.0, connect=5.0)

_HEADERS = {
    "Content-Type":  "application/json",
    "x-mcp-secret":  MCP_SECRET,
}

# ── Route map  (tool_name → (method, path)) ──────────────────────────────────
_ROUTES: dict[str, tuple[str, str]] = {
    "execute_sql_query":          ("POST", "/tools/execute_sql_query"),
    "execute_write_sql":          ("POST", "/tools/execute_write_sql"),
    "insert_audit_log":           ("POST", "/tools/insert_audit_log"),
    "get_table_schema":           ("POST", "/tools/get_table_schema"),
    "list_public_tables":         ("GET",  "/tools/list_public_tables"),
    "query_NDMO_knowledge_base":  ("POST", "/tools/query_NDMO_knowledge_base"),
    "get_dq_summary":             ("GET",  "/tools/get_dq_summary"),
}


class MCPError(Exception):
    """Raised when an MCP tool call fails."""
    def __init__(self, tool: str, status: int, detail: str):
        self.tool   = tool
        self.status = status
        self.detail = detail
        super().__init__(f"MCP [{tool}] HTTP {status}: {detail}")


def call_tool(tool_name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Synchronous MCP tool call.

    Parameters
    ----------
    tool_name : str
        One of the keys in _ROUTES.
    payload : dict, optional
        JSON body for POST routes; ignored for GET routes.

    Returns
    -------
    dict
        Parsed JSON response from MCP server.

    Raises
    ------
    MCPError
        On HTTP errors or unknown tool names.
    """
    if tool_name not in _ROUTES:
        raise MCPError(tool_name, 400, f"Unknown tool '{tool_name}'")

    method, path = _ROUTES[tool_name]
    url = MCP_BASE_URL + path

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            if method == "GET":
                resp = client.get(url, headers=_HEADERS)
            else:
                resp = client.post(url, json=payload or {}, headers=_HEADERS)
    except httpx.ConnectError as exc:
        raise MCPError(tool_name, 503, f"Cannot reach MCP server at {MCP_BASE_URL}: {exc}")
    except httpx.TimeoutException as exc:
        raise MCPError(tool_name, 504, f"MCP request timed out: {exc}")

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise MCPError(tool_name, resp.status_code, detail)

    return resp.json()


# ── Convenience wrappers used by agents ──────────────────────────────────────

def sql_query(sql: str, params: dict | None = None) -> list[dict]:
    """Execute a read-only SQL query; return list of row dicts."""
    result = call_tool("execute_sql_query", {"sql": sql, "params": params or {}})
    return result.get("rows", [])


def write_sql(
    sql:            str,
    username:       str = "admin",
    user_role:      str = "Admin",
    agent_name:     str = "DQ_Agent",
    action_type:    str = "AUTO_FIX_APPLIED",
    original_issue: str = "",
) -> dict:
    """Execute an approved write SQL correction and log to audit_logs."""
    return call_tool("execute_write_sql", {
        "sql":            sql,
        "username":       username,
        "user_role":      user_role,
        "agent_name":     agent_name,
        "action_type":    action_type,
        "original_issue": original_issue,
    })


def insert_audit_log(
    username:       str,
    action_type:    str,
    original_issue: str,
    executed_sql:   str = "",
    user_role:      str = "Admin",
    agent_name:     str = "System",
    mcp_tool_used:  str = "",
) -> dict:
    """Insert a row into audit_logs for non-write agent events."""
    return call_tool("insert_audit_log", {
        "username":       username,
        "user_role":      user_role,
        "agent_name":     agent_name,
        "action_type":    action_type,
        "original_issue": original_issue,
        "executed_sql":   executed_sql,
        "mcp_tool_used":  mcp_tool_used,
    })


def table_schema(table_name: str) -> list[dict]:
    """Return column metadata for a table."""
    result = call_tool("get_table_schema", {"table_name": table_name})
    return result.get("columns", [])


def list_tables() -> list[str]:
    """Return all public table names."""
    result = call_tool("list_public_tables")
    return result.get("tables", [])


def NDMO_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Search the NDMO knowledge base via pgvector RAG.
    Embeds query locally, passes vector to MCP for cosine similarity search.
    Falls back to static NDMO articles if sentence-transformers unavailable.
    """
    embedding = None
    try:
        from sentence_transformers import SentenceTransformer
        model     = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(query).tolist()
    except Exception:
        pass

    result = call_tool(
        "query_NDMO_knowledge_base",
        {"query": query, "num_results": num_results, "embedding": embedding},
    )
    return result.get("results", [])


def dq_summary() -> dict:
    """Return live data-quality KPIs."""
    return call_tool("get_dq_summary")