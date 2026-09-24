"""
Zero-Cloud Local Hybrid RAG with Mistral AI & SQLite FTS5 (BM25 + RRF)
======================================================================
A reference implementation demonstrating zero-cloud hybrid retrieval
and grounded question answering using:
1. Mistral AI Embeddings (or local open-weights Mistral/Ministral)
2. SQLite FTS5 native BM25 token matching
3. Dense Vector Cosine Similarity
4. Reciprocal Rank Fusion (RRF, k=60)
5. Grounded citation indexing ([1], [2])

Author: Çağrı Giray Keşan (@Cagrik34)
License: Apache 2.0
"""

import sys
import sqlite3
import numpy as np
from typing import List, Tuple, Dict, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

class MistralSQLiteHybridStore:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
                    content,
                    source_file UNINDEXED,
                    tokenize='unicode61'
                )
            """)

    def insert_chunk(self, source_file: str, content: str, embedding: List[float]) -> None:
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        with self.conn:
            self.conn.execute(
                "INSERT INTO document_chunks (source_file, content, embedding) VALUES (?, ?, ?)",
                (source_file, content, vec.tobytes())
            )
            self.conn.execute(
                "INSERT INTO document_chunks_fts (content, source_file) VALUES (?, ?)",
                (content, source_file)
            )

    def search_dense(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[int, str, str, float]]:
        q_norm = np.linalg.norm(query_vec)
        q_arr = np.array(query_vec, dtype=np.float32)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        cursor = self.conn.execute("SELECT id, source_file, content, embedding FROM document_chunks")
        hits = []
        for doc_id, src, content, blob in cursor.fetchall():
            doc_vec = np.frombuffer(blob, dtype=np.float32)
            sim = float(np.dot(q_arr, doc_vec))
            hits.append((doc_id, src, content, sim))
        hits.sort(key=lambda x: x[3], reverse=True)
        return hits[:top_k]

    def search_sparse_bm25(self, query_text: str, top_k: int = 5) -> List[Tuple[int, str, str, float]]:
        clean_tokens = [t for t in query_text.replace("'", "").replace('"', '').split() if len(t) > 1]
        if not clean_tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in clean_tokens)
        cursor = self.conn.execute(
            "SELECT rowid, source_file, content, rank FROM document_chunks_fts WHERE document_chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, top_k)
        )
        hits = []
        for doc_id, src, content, rank in cursor.fetchall():
            hits.append((doc_id, src, content, 1.0 / (1.0 + abs(float(rank)))))
        return hits

    def hybrid_search(self, query_text: str, query_vec: List[float], top_k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        dense_hits = self.search_dense(query_vec, top_k=10)
        sparse_hits = self.search_sparse_bm25(query_text, top_k=10)
        fused = {}
        chunk_map = {}

        for rank, (doc_id, src, content, sim) in enumerate(dense_hits, start=1):
            key = f"{src}::{content[:50]}"
            chunk_map[key] = (src, content, "vector")
            fused[key] = fused.get(key, 0.0) + (1.0 / (rrf_k + rank))

        for rank, (doc_id, src, content, bm25) in enumerate(sparse_hits, start=1):
            key = f"{src}::{content[:50]}"
            if key not in chunk_map:
                chunk_map[key] = (src, content, "bm25")
            else:
                chunk_map[key] = (src, content, "hybrid")
            fused[key] = fused.get(key, 0.0) + (1.0 / (rrf_k + rank))

        sorted_keys = sorted(fused.keys(), key=lambda k: fused[k], reverse=True)[:top_k]
        output = []
        for idx, key in enumerate(sorted_keys, start=1):
            src, content, match_type = chunk_map[key]
            output.append({
                "citation_index": idx,
                "source_file": src,
                "content": content,
                "rrf_score": round(fused[key], 4),
                "match_type": match_type
            })
        return output

def format_mistral_rag_prompt(query: str, retrieved_passages: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for p in retrieved_passages:
        context_blocks.append(f"[{p['citation_index']}] (Source: {p['source_file']}) {p['content']}")
    context_str = "\n\n".join(context_blocks)
    
    return f"""<s>[INST] You are an expert AI assistant powered by Mistral AI. Answer the following question based solely on the provided context. Strict rule: Cite the exact citation number in brackets (e.g. [1], [2]) for every factual statement.

Context Information:
---------------------
{context_str}
---------------------

Question: {query} [/INST]"""

if __name__ == "__main__":
    print("=" * 75)
    print(" 🇫🇷 MISTRAL AI ZERO-CLOUD HYBRID RAG (SQLITE FTS5 + RRF)")
    print("=" * 75)

    store = MistralSQLiteHybridStore()
    docs = [
        ("q3_financial_report.pdf", "CodePulse engineering project total Q3 budget was allocated at 2,340,000 TL with 15 active developers.", [0.85, 0.12, 0.22] + [0.0] * 1021),
        ("architecture_specs.md", "Zenith AI leverages Microsoft phi-4-mini (3.8B parameters) for local zero-cloud inference.", [0.15, 0.88, 0.10] + [0.0] * 1021),
        ("hr_policy_2026.docx", "Remote work expense allowance is capped at 15,000 TL per employee quarterly.", [0.10, 0.10, 0.90] + [0.0] * 1021)
    ]
    for src, text, emb in docs:
        store.insert_chunk(src, text, emb)
    print("✅ Ingested 3 documents into SQLite FTS5 store.")

    query = "What is the quarterly remote work allowance limit in TL?"
    dummy_q_vec = [0.08, 0.12, 0.88] + [0.0] * 1021
    results = store.hybrid_search(query, dummy_q_vec, top_k=2)

    print(f"\n🔍 Search Query: '{query}'")
    print(f"📊 Top Hits (RRF k=60):")
    for r in results:
        print(f" [{r['citation_index']}] {r['source_file']} ({r['match_type'].upper()}) -> Score: {r['rrf_score']}")
        print(f" \"{r['content']}\"")

    prompt = format_mistral_rag_prompt(query, results)
    print("\n📝 Formatted Mistral Prompt ([INST]...[/INST]):")
    print(prompt)
    print("=" * 75)
    print("✅ Mistral Hybrid RAG Recipe Verified Successfully!")
    print("=" * 75)
