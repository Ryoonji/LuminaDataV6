# -*- coding: utf-8 -*-
"""
LuminaData Enterprise v6.0 — RAG Ingestion Script
===================================================
Run this script ONCE (or whenever you add new PDFs) to:
  1. Extract text from your PDF(s)
  2. Split into overlapping chunks
  3. Embed using sentence-transformers (local, no API key)
  4. Store in PostgreSQL pgvector table: rag_documents

Usage:
    python rag_ingest.py --pdf path/to/your.pdf
    python rag_ingest.py --pdf NDMO.pdf --pdf pdpl.pdf
    python rag_ingest.py --pdf docs/         # ingest entire folder

Requirements (add to requirements_mcp.txt):
    pymupdf>=1.24.0
    sentence-transformers>=2.7.0
"""

from __future__ import annotations

import os, sys, argparse, logging
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("rag_ingest")

# ── DB config ─────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "5433")       # external port for local run
DB_NAME     = os.getenv("DB_NAME",     "luminadata")
DB_USER     = os.getenv("DB_USER",     "lumina_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "lumina_pass")

# ── Chunking settings ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 80    # words overlap between chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def ensure_table(conn):
    """Create rag_documents table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id         SERIAL PRIMARY KEY,
                source     TEXT NOT NULL,        -- filename
                page_num   INTEGER,              -- page number in PDF
                chunk_idx  INTEGER,              -- chunk index within page
                content    TEXT NOT NULL,        -- raw chunk text
                embedding  vector(384),          -- all-MiniLM-L6-v2
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_rag_embedding
                ON rag_documents USING hnsw (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_rag_source
                ON rag_documents(source);
        """)
        conn.commit()
    logger.info("Table rag_documents ready.")


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text per page using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(1)

    pages = []
    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    logger.info("  Extracted %d pages from %s", len(pages), pdf_path.name)
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using all-MiniLM-L6-v2 (local)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    logger.info("  Loading embedding model (first run downloads ~80MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return [e.tolist() for e in embeddings]


def ingest_pdf(pdf_path: Path, conn):
    """Full pipeline for one PDF file."""
    logger.info("Processing: %s", pdf_path.name)

    # Check if already ingested
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM rag_documents WHERE source = %s", (pdf_path.name,))
        count = cur.fetchone()[0]
    if count > 0:
        logger.info("  Already ingested (%d chunks). Use --force to re-ingest.", count)
        return

    # Extract
    pages = extract_pages(pdf_path)
    if not pages:
        logger.warning("  No text found in PDF — is it a scanned image? Try OCR first.")
        return

    # Chunk
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "source":    pdf_path.name,
                "page_num":  page["page"],
                "chunk_idx": idx,
                "content":   chunk,
            })
    logger.info("  Created %d chunks across %d pages", len(all_chunks), len(pages))

    # Embed
    texts      = [c["content"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # Insert
    rows = [
        (c["source"], c["page_num"], c["chunk_idx"], c["content"], str(emb))
        for c, emb in zip(all_chunks, embeddings)
    ]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO rag_documents (source, page_num, chunk_idx, content, embedding)
            VALUES %s
        """, rows)
    conn.commit()
    logger.info("  ✅ Inserted %d chunks for %s", len(rows), pdf_path.name)


def list_ingested(conn):
    """Show what's already in the RAG table."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT source, COUNT(*) as chunks, MIN(created_at)::date as ingested_on
            FROM rag_documents
            GROUP BY source
            ORDER BY ingested_on DESC
        """)
        rows = cur.fetchall()
    if not rows:
        print("No documents ingested yet.")
    else:
        print(f"\n{'Source':<40} {'Chunks':>8} {'Ingested':>12}")
        print("-" * 62)
        for source, chunks, date in rows:
            print(f"{source:<40} {chunks:>8} {str(date):>12}")
        print()


def delete_source(source: str, conn):
    """Remove all chunks for a specific source file."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rag_documents WHERE source = %s", (source,))
        deleted = cur.rowcount
    conn.commit()
    logger.info("Deleted %d chunks for '%s'", deleted, source)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into LuminaData RAG (pgvector)"
    )
    parser.add_argument("--pdf",    action="append", metavar="PATH",
                        help="PDF file or folder to ingest (repeatable)")
    parser.add_argument("--list",   action="store_true",
                        help="List already-ingested documents")
    parser.add_argument("--delete", metavar="FILENAME",
                        help="Delete all chunks for a specific source file")
    parser.add_argument("--force",  action="store_true",
                        help="Re-ingest even if source already exists")
    args = parser.parse_args()

    conn = get_conn()
    ensure_table(conn)

    if args.list:
        list_ingested(conn)
        conn.close()
        return

    if args.delete:
        delete_source(args.delete, conn)
        conn.close()
        return

    if not args.pdf:
        parser.print_help()
        print("\nExample:\n  python rag_ingest.py --pdf NDMO_dq_standard.pdf")
        conn.close()
        return

    # Collect all PDF paths
    pdf_paths = []
    for p in args.pdf:
        path = Path(p)
        if path.is_dir():
            pdf_paths.extend(sorted(path.glob("**/*.pdf")))
        elif path.suffix.lower() == ".pdf" and path.exists():
            pdf_paths.append(path)
        else:
            logger.warning("Skipping '%s' — not a PDF or doesn't exist", p)

    if not pdf_paths:
        logger.error("No valid PDF files found.")
        conn.close()
        return

    if args.force:
        for p in pdf_paths:
            delete_source(p.name, conn)

    for pdf_path in pdf_paths:
        ingest_pdf(pdf_path, conn)

    logger.info("\nDone. Run with --list to see all ingested documents.")
    conn.close()


if __name__ == "__main__":
    main()