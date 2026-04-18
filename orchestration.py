# -*- coding: utf-8 -*-
"""
LuminaData Enterprise v6.0 — LangGraph Orchestration
======================================================
Master Orchestrator + three specialised agents wired with LangGraph.

Graph topology:
    START → orchestrator_node → {sql_agent | dq_agent | general_agent} → END

Intent classification:
    SQL_QUERY            → sql_agent
    NDMO_COMPLIANCE_CHECK → dq_agent
    GENERAL_INFO         → general_agent

All agents communicate with the database/RAG exclusively through
mcp_client.call_tool().  No agent imports sqlalchemy or psycopg2.
"""

from __future__ import annotations

import os, json, logging
from typing import TypedDict, Literal, Annotated, Any

# LangGraph
from langgraph.graph import StateGraph, END

# LangChain Groq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from mcp_client import (
    sql_query, write_sql, table_schema, list_tables,
    NDMO_search, dq_summary, insert_audit_log, MCPError,
)
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger("orchestration")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── LLM instance (shared across agents) ──────────────────────────────────────
def _make_llm(temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=temperature,
        max_tokens=2048,
    )


# ── Shared agent state ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Input
    user_message: str
    username:     str
    user_role:    str

    # Routing
    intent:  Literal["SQL_QUERY", "NDMO_COMPLIANCE_CHECK", "GENERAL_INFO"] | None

    # Thought trace (list of step strings, shown in UI)
    thoughts: list[str]

    # Final response
    response:     str
    sql_executed: str | None
    dq_report:    dict[str, Any] | None
    rag_sources:  list[dict] | None   # RAG chunks returned by NDMO_search
    error:        str | None


def _add_thought(state: AgentState, thought: str) -> None:
    """Append a thought string in place (mutates the list)."""
    state["thoughts"].append(thought)
    logger.info("[THOUGHT] %s", thought)


# ── Node 1: Master Orchestrator ───────────────────────────────────────────────
def orchestrator_node(state: AgentState) -> AgentState:
    """
    Classifies user intent into SQL_QUERY | NDMO_COMPLIANCE_CHECK | GENERAL_INFO | PDF_QUERY.
    PDF_QUERY is triggered when the user asks what the ingested PDF says about a topic.
    """
    _add_thought(state, "🧠 Orchestrator received message — classifying intent...")

    # ── Fast-path: PDF self-test query ────────────────────────────────────────
    msg_lower = state["user_message"].lower()
    pdf_triggers = [
        "what does the ingested pdf say",
        "what does the pdf say",
        "what does the document say",
        "what does the policy say",
        "according to the pdf",
        "from the pdf",
        "in the pdf",
    ]
    if any(t in msg_lower for t in pdf_triggers):
        state["intent"] = "PDF_QUERY"
        _add_thought(state, "📄 PDF self-test query detected — routing directly to RAG Auditor")
        return state

    llm = _make_llm(temperature=0.0)
    prompt = SystemMessage(content="""You are the Master Orchestrator for LuminaData, an AI data-quality platform.
Classify the user's message into EXACTLY ONE of these intents:

  SQL_QUERY              — user wants to query, explore, or retrieve data from the database
  NDMO_COMPLIANCE_CHECK — user wants a compliance audit, DQ score, regulation check, or NDMO/PDPL analysis
  GENERAL_INFO           — everything else: greetings, how-to questions, feature explanations

Reply with ONLY a JSON object: {"intent": "<INTENT>", "reason": "<one sentence>"}
No other text.""")

    try:
        resp = llm.invoke([prompt, HumanMessage(content=state["user_message"])])
        raw  = resp.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        intent = data.get("intent", "GENERAL_INFO").upper()
        if intent not in {"SQL_QUERY", "NDMO_COMPLIANCE_CHECK", "GENERAL_INFO"}:
            intent = "GENERAL_INFO"
        reason = data.get("reason", "")
        _add_thought(state, f"✅ Intent classified: **{intent}** — {reason}")
        state["intent"] = intent
    except Exception as exc:
        logger.warning("Orchestrator classification failed: %s", exc)
        state["intent"] = "GENERAL_INFO"
        _add_thought(state, f"⚠️ Classification error ({exc}), defaulting to GENERAL_INFO")

    return state


# ── Node 2: SQL Agent ─────────────────────────────────────────────────────────
def sql_agent(state: AgentState) -> AgentState:
    """
    1. Fetches schema from MCP (get_table_schema + list_tables).
    2. Prompts LLM to generate safe SELECT SQL.
    3. Executes via MCP (execute_sql_query).
    4. Formats results as Markdown table.
    """
    _add_thought(state, "🔎 SQL Agent activated — fetching schema context via MCP...")

    # Step 1: Schema context
    try:
        tables = list_tables()
        schema_ctx = []
        for tbl in tables:
            cols = table_schema(tbl)
            col_str = ", ".join(f"{c['column_name']} ({c['data_type']})" for c in cols)
            schema_ctx.append(f"Table `{tbl}`: {col_str}")
        schema_text = "\n".join(schema_ctx)
        _add_thought(state, f"📋 Schema loaded for {len(tables)} table(s): {', '.join(tables)}")
    except MCPError as exc:
        state["error"]    = f"MCP schema fetch failed: {exc.detail}"
        state["response"] = f"❌ Could not reach the MCP server to fetch schema: {exc.detail}"
        _add_thought(state, f"❌ MCP error during schema fetch: {exc}")
        return state

    # Step 2: Generate SQL
    _add_thought(state, "🤖 SQL Agent calling LLM to generate query...")
    llm = _make_llm()
    sys_prompt = SystemMessage(content=f"""You are the SQL Agent for LuminaData.
You translate natural language questions into safe, read-only PostgreSQL SELECT queries.

Database schema:
{schema_text}

Rules:
- Only generate SELECT statements.
- Never use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE, GRANT, REVOKE.
- Mask PII columns: replace national_id with LEFT(national_id,3)||'XXXX'||RIGHT(national_id,2),
  phone_number with LEFT(phone_number,2)||'XXXXX'||RIGHT(phone_number,2),
  email with SPLIT_PART(email,'@',1) (first 2 chars) || '***@***.com'.
- If the question cannot be answered with the schema, say so.
- Reply with ONLY valid SQL. No explanation. No markdown fences.""")

    try:
        resp = llm.invoke([sys_prompt, HumanMessage(content=state["user_message"])])
        sql  = resp.content.strip().replace("```sql", "").replace("```", "").strip()
        _add_thought(state, f"📝 Generated SQL:\n```sql\n{sql}\n```")
    except Exception as exc:
        state["error"]    = str(exc)
        state["response"] = f"❌ LLM error during SQL generation: {exc}"
        return state

    # Step 3: Execute via MCP
    _add_thought(state, "⚡ Executing SQL via MCP tool `execute_sql_query`...")
    try:
        result = sql_query(sql)
        state["sql_executed"] = sql
        _add_thought(state, f"✅ Query returned {len(result)} row(s)")
    except MCPError as exc:
        state["error"]    = exc.detail
        state["response"] = f"❌ SQL execution failed: {exc.detail}\n\nGenerated SQL:\n```sql\n{sql}\n```"
        _add_thought(state, f"❌ MCP SQL execution error: {exc.detail}")
        return state

    # Step 4: Format response
    if not result:
        state["response"] = "✅ Query executed successfully — no rows returned."
        return state

    # Build markdown table
    cols = list(result[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in result[:100]]
    table  = "\n".join([header, sep] + rows)

    summary_prompt = f"Summarise these {len(result)} rows in 1-2 plain sentences:\n{table[:2000]}"
    try:
        summary_llm = _make_llm(temperature=0.3)
        summary = summary_llm.invoke([HumanMessage(content=summary_prompt)]).content.strip()
    except Exception:
        summary = f"Query returned {len(result)} row(s)."

    state["response"] = f"{summary}\n\n{table}"
    return state


# ── Node 3: NDMO DQ Agent ────────────────────────────────────────────────────
def dq_agent(state: AgentState) -> AgentState:
    """
    1. Fetches live DQ metrics via MCP (get_dq_summary).
    2. Samples citizen records via MCP (execute_sql_query).
    3. Retrieves NDMO regulations via MCP (query_NDMO_knowledge_base).
    4. Uses LLM to generate a structured compliance report.
    """
    _add_thought(state, "🛡️ NDMO DQ Agent activated — beginning compliance audit...")

    # Step 1: DQ metrics
    _add_thought(state, "📊 Fetching live DQ metrics via MCP `get_dq_summary`...")
    try:
        dq = dq_summary()
        score    = dq.get("compliance_score", 0)
        total    = dq.get("total_records",    0)
        alerts   = dq.get("active_alerts",    [])
        dims     = dq.get("dimensions",       {})
        state["dq_report"] = dq
        _add_thought(state, f"📈 DQ Score: **{score}%** across {total} records | Alerts: {len(alerts)}")
    except MCPError as exc:
        state["error"]    = exc.detail
        state["response"] = f"❌ Could not fetch DQ metrics: {exc.detail}"
        return state

    # Step 2: Citizen data sample (PII-masked)
    _add_thought(state, "🗃️ Sampling citizen records via MCP `execute_sql_query`...")
    try:
        sample = sql_query(
            """
            SELECT id, full_name,
                   LEFT(COALESCE(national_id,''),3)||'XXXX'||RIGHT(COALESCE(national_id,''),2) AS national_id_masked,
                   id_issue_date, id_expiry_date, city, region,
                   CASE WHEN phone_number IS NULL THEN 'MISSING' ELSE 'PRESENT' END AS phone_status,
                   CASE WHEN email       IS NULL THEN 'MISSING' ELSE 'PRESENT' END AS email_status
            FROM citizens LIMIT 25
            """
        )
        _add_thought(state, f"✅ Loaded {len(sample)} citizen records for audit")
    except MCPError as exc:
        _add_thought(state, f"⚠️ Could not load sample records: {exc.detail}")
        sample = []

    # Step 3: NDMO regulations
    _add_thought(state, "📚 Querying NDMO knowledge base via MCP `query_NDMO_knowledge_base`...")
    try:
        regulations = NDMO_search(state["user_message"], num_results=6)
        reg_text    = "\n\n".join(
            f"**{r['title']}**\n{r['content']}" for r in regulations
        )
        source_label = "pgvector RAG" if regulations and regulations[0].get("source") == "pgvector_rag" else "NDMO static fallback"
        _add_thought(state, f"📖 Retrieved {len(regulations)} regulation(s) from {source_label}")
        state["rag_sources"] = regulations
    except MCPError as exc:
        _add_thought(state, f"⚠️ RAG unavailable: {exc.detail} — using built-in rules")
        reg_text = "NDMO regulations unavailable. Rely on built-in DQ rules."
        state["rag_sources"] = []

    # Step 4: LLM compliance report
    _add_thought(state, "🤖 DQ Agent generating compliance report via LLM...")
    llm = _make_llm(temperature=0.2)

    dims_json   = json.dumps(dims,   indent=2)
    alerts_text = "\n".join(f"- {a}" for a in alerts) if alerts else "- No critical alerts"
    sample_json = json.dumps(sample[:10], indent=2, default=str) if sample else "[]"

    sys_prompt = SystemMessage(content=f"""You are the NDMO Data Quality Compliance Agent for LuminaData.
Your job is to produce a clear, structured compliance audit report.

=== LIVE DQ METRICS ===
Compliance Score: {score}%
Total Records:    {total}
Active Alerts:
{alerts_text}

Dimensions Detail:
{dims_json}

=== CITIZEN DATA SAMPLE (PII masked) ===
{sample_json}

=== NDMO / PDPL REGULATIONS ===
{reg_text}

=== INSTRUCTIONS ===
1. Write a structured compliance audit report in Markdown.
2. For EACH data quality dimension (Completeness, Validity, Accuracy, Timeliness, Uniqueness, Consistency):
   - State the compliance status (✅ PASS / ⚠️ WARNING / ❌ FAIL)
   - Cite the relevant NDMO/PDPL article
   - List specific violations found
   - Recommend corrective action
3. Conclude with an overall risk rating (LOW / MEDIUM / HIGH / CRITICAL) and priority actions.
4. Be specific — reference actual numbers from the metrics.
5. Do NOT expose raw PII.""")

    try:
        resp = llm.invoke([sys_prompt, HumanMessage(content=state["user_message"])])
        state["response"] = resp.content
        _add_thought(state, f"✅ Compliance report generated — Overall Score: {score}%")
    except Exception as exc:
        state["error"]    = str(exc)
        state["response"] = f"❌ LLM error generating report: {exc}"

    return state


# ── Node 5: PDF Query Agent ───────────────────────────────────────────────────
def pdf_query_agent(state: AgentState) -> AgentState:
    """
    Handles 'What does the ingested PDF say about X?' queries.
    Prioritises NDMO_search and prefixes the answer with
    [Verified from Policy Document] when pgvector results are found.
    """
    _add_thought(state, "📄 PDF Query Agent activated — searching ingested documents...")

    try:
        results = NDMO_search(state["user_message"], num_results=8)
        state["rag_sources"] = results
    except MCPError as exc:
        state["rag_sources"] = []
        state["response"]    = f"❌ RAG search failed: {exc.detail}"
        return state

    from_pdf = [r for r in results if r.get("source") == "pgvector_rag"]

    if not from_pdf:
        state["response"] = (
            "No relevant content found in the ingested PDF documents for that topic.\n\n"
            "Make sure you have uploaded an NDMO PDF via the sidebar, then try again."
        )
        _add_thought(state, "⚠️ No pgvector results — no PDFs ingested yet")
        return state

    _add_thought(state, f"✅ Found {len(from_pdf)} chunk(s) from uploaded PDF(s)")

    context = "\n\n".join(
        f"[{r['title']}  |  score: {r['score']:.2f}]\n{r['content']}"
        for r in from_pdf
    )

    llm = _make_llm(temperature=0.1)
    sys_prompt = SystemMessage(content=f"""You are the LuminaData Policy Document Agent.
The user wants to know what the ingested NDMO PDF says about a topic.
Answer ONLY using the retrieved document chunks below.
Start your response with exactly: [Verified from Policy Document]
Then answer concisely, quoting or paraphrasing the relevant passages.
If the chunks do not contain relevant information, say so clearly.

RETRIEVED CHUNKS:
{context}""")

    try:
        resp = llm.invoke([sys_prompt, HumanMessage(content=state["user_message"])])
        state["response"] = resp.content
        _add_thought(state, "✅ PDF query answer ready")
    except Exception as exc:
        state["error"]    = str(exc)
        state["response"] = f"❌ LLM error: {exc}"

    return state


# ── Node 6: General Info Agent ────────────────────────────────────────────────
def general_agent(state: AgentState) -> AgentState:
    """
    Handles general questions about the platform, features, and navigation.
    Uses system context but does not call any MCP tools.
    """
    _add_thought(state, "💬 General Info Agent activated...")

    llm = _make_llm(temperature=0.5)
    sys_prompt = SystemMessage(content="""You are the LuminaData Assistant — friendly, helpful, concise.
LuminaData v6.0 is an AI-powered Citizen Data Quality platform built with:
  • LangGraph multi-agent orchestration
  • MCP (Model Context Protocol) for tool access
  • pgvector RAG for NDMO/PDPL regulation search
  • PostgreSQL (pgvector) for data storage
  • Streamlit for the UI

You can help users:
  - Understand the platform's features
  - Navigate to the right tab/function
  - Understand data quality dimensions (Completeness, Validity, Accuracy, Timeliness, Uniqueness, Consistency)
  - Understand NDMO/PDPL compliance requirements

For data queries, tell users to ask a specific question about the citizen data.
For compliance audits, tell users to ask about NDMO compliance or DQ score.
Be brief and helpful. Use bullet points sparingly.""")

    try:
        resp = llm.invoke([sys_prompt, HumanMessage(content=state["user_message"])])
        state["response"] = resp.content
        _add_thought(state, "✅ General response ready")
    except Exception as exc:
        state["error"]    = str(exc)
        state["response"] = f"I'm sorry, I encountered an error: {exc}"

    return state


# ── Routing function ──────────────────────────────────────────────────────────
def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "GENERAL_INFO")
    routes = {
        "SQL_QUERY":             "sql_agent",
        "NDMO_COMPLIANCE_CHECK": "dq_agent",
        "PDF_QUERY":             "pdf_query_agent",
        "GENERAL_INFO":          "general_agent",
    }
    return routes.get(intent, "general_agent")


# ── Build LangGraph ───────────────────────────────────────────────────────────
def build_graph() -> Any:
    """Compile and return the LangGraph application."""
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("orchestrator",    orchestrator_node)
    graph.add_node("sql_agent",       sql_agent)
    graph.add_node("dq_agent",        dq_agent)
    graph.add_node("pdf_query_agent", pdf_query_agent)
    graph.add_node("general_agent",   general_agent)

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_intent,
        {
            "sql_agent":       "sql_agent",
            "dq_agent":        "dq_agent",
            "pdf_query_agent": "pdf_query_agent",
            "general_agent":   "general_agent",
        },
    )
    graph.add_edge("sql_agent",       END)
    graph.add_edge("dq_agent",        END)
    graph.add_edge("pdf_query_agent", END)
    graph.add_edge("general_agent",   END)

    return graph.compile()


# ── Public run function ───────────────────────────────────────────────────────
def run_agent(
    user_message: str,
    username:     str = "admin",
    user_role:    str = "Admin",
) -> AgentState:
    """
    Invoke the full LangGraph pipeline for a user message.

    Returns the final AgentState with:
      - state["response"]     : final markdown answer
      - state["thoughts"]     : list of reasoning steps (for UI trace)
      - state["intent"]       : classified intent
      - state["sql_executed"] : SQL that was run (if any)
      - state["dq_report"]    : DQ metrics dict (if compliance audit)
      - state["error"]        : error message (if any)
    """
    app = build_graph()

    initial: AgentState = {
        "user_message": user_message,
        "username":     username,
        "user_role":    user_role,
        "intent":       None,
        "thoughts":     [],
        "response":     "",
        "sql_executed": None,
        "dq_report":    None,
        "rag_sources":  None,
        "error":        None,
    }

    final = app.invoke(initial)
    return final