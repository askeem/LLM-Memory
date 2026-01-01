"""
Memory Store (SQLite) — Durable Hybrid Retrieval + Metadata + Company-aware filtering.

Goals:
- Robust tokenization (tiktoken if available; regex fallback).
- Hybrid retrieval: lexical candidate generation + vector similarity reranking + recency.
- Company-aware filtering to prevent cross-client contamination.
- Metadata storage (JSON) for future expansions (tags, topics, effective dates).
- Scales beyond a handful of memories via an inverted index table.

Retrieval Pipeline:
1) Tokenize query -> candidate IDs via inverted index (fast lexical prefilter).
2) Load candidate rows (optionally company-filtered; backoff if too few).
3) Score:
   score = w_vec * cosine(vec(q), vec(item)) + w_lex * lexical_score + w_recency * recency
4) Select mix: 80% worked / 20% failed (when possible).
5) Return top k with compression (top raw_limit raw, rest summarized).
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------
# Time helpers
# ---------------------------
def _now_ts() -> float:
    return time.time()


# ---------------------------
# Tokenization (durable)
# ---------------------------
@lru_cache(maxsize=4)
def _get_encoder(name: str = "cl100k_base"):
    import tiktoken
    return tiktoken.get_encoding(name)


def tokenize(text: str) -> List[str]:
    """
    Durable tokenizer:
    - Prefer tiktoken token IDs -> stable across punctuation/JSON/symbols.
    - Fallback to regex tokenization if tiktoken unavailable.
    """
    text = text or ""
    try:
        enc = _get_encoder("cl100k_base")
        ids = enc.encode(text)
        # Convert to str tokens. This avoids issues with ints in SQLite indexing.
        return [f"t{tid}" for tid in ids]
    except Exception:
        import re
        return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ---------------------------
# Deterministic hashed embedding
# ---------------------------
def hashed_embedding_from_tokens(tokens: List[str], dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec
    for tok in tokens:
        h = zlib.adler32(tok.encode("utf-8"))
        idx = h % dim
        sign = -1.0 if (h & 1) else 1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def hashed_embedding(text: str, dim: int = 256) -> np.ndarray:
    return hashed_embedding_from_tokens(tokenize(text), dim=dim)


# ---------------------------
# Data model
# ---------------------------
@dataclass
class MemoryItem:
    id: int
    folder: str
    text: str
    summary: str
    status: str  # "worked" | "failed"
    score: float = 0.0
    is_compressed: bool = False
    company: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# ---------------------------
# Memory store
# ---------------------------
class MemoryStore:
    def __init__(self, db_path: str = "memory.sqlite", dim: int = 256):
        self.dim = dim
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_db()
        self._migrate_db()

    def _init_db(self) -> None:
        # folders: task type buckets (kept for your existing mental model)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                emb BLOB
            )
            """
        )

        # memory items
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_id INTEGER,
                text TEXT,
                summary TEXT,
                status TEXT,
                ts REAL,
                util REAL DEFAULT 1.0,
                emb BLOB,
                meta TEXT,
                company TEXT,
                task_id TEXT,
                task_type TEXT,
                FOREIGN KEY(folder_id) REFERENCES folders(id)
            )
            """
        )

        # inverted index for lexical candidate generation
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS token_index (
                token TEXT,
                item_id INTEGER,
                tf INTEGER DEFAULT 1,
                PRIMARY KEY(token, item_id),
                FOREIGN KEY(item_id) REFERENCES memory_items(id)
            )
            """
        )

        # indices
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_company ON memory_items(company);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_folder  ON memory_items(folder_id);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_status  ON memory_items(status);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_token_token   ON token_index(token);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_token_item    ON token_index(item_id);")

        self.conn.commit()

    def _migrate_db(self) -> None:
        """
        Allow upgrading older DBs. Adds missing columns gracefully.
        """
        cur = self.conn.execute("PRAGMA table_info(memory_items)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, sql_type: str) -> None:
            if name not in cols:
                self.conn.execute(f"ALTER TABLE memory_items ADD COLUMN {name} {sql_type}")

        add_col("meta", "TEXT")
        add_col("company", "TEXT")
        add_col("task_id", "TEXT")
        add_col("task_type", "TEXT")

        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_company ON memory_items(company);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_status  ON memory_items(status);")
        self.conn.commit()

    def _get_or_create_folder(self, folder_name: str) -> int:
        cur = self.conn.execute("SELECT id FROM folders WHERE name=?", (folder_name,))
        row = cur.fetchone()
        if row:
            return int(row[0])

        f_emb = hashed_embedding(folder_name, dim=self.dim).tobytes()
        cur = self.conn.execute("INSERT INTO folders (name, emb) VALUES (?, ?)", (folder_name, f_emb))
        return int(cur.lastrowid)

    def _index_tokens(self, item_id: int, text: str, max_tokens: int = 400) -> None:
        """
        Insert tokens into token_index. We cap tokens to avoid huge index writes
        for very long memories. Token frequency is stored.
        """
        toks = tokenize(text)[:max_tokens]
        if not toks:
            return

        # simple TF counts
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1

        rows = [(tok, item_id, count) for tok, count in tf.items()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO token_index(token, item_id, tf) VALUES (?, ?, ?)",
            rows
        )

    def add(
        self,
        folder_name: str,
        text: str,
        summary: str,
        status: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = meta or {}
        company = meta.get("company")
        task_id = meta.get("task_id")
        task_type = meta.get("task_type") or folder_name

        folder_id = self._get_or_create_folder(folder_name)

        emb = hashed_embedding(text, dim=self.dim).tobytes()

        cur = self.conn.execute(
            """
            INSERT INTO memory_items (folder_id, text, summary, status, ts, emb, meta, company, task_id, task_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (folder_id, text, summary, status, _now_ts(), emb, json.dumps(meta), company, task_id, task_type),
        )
        item_id = int(cur.lastrowid)

        # index tokens
        self._index_tokens(item_id, text)

        self.conn.commit()

    # ---------------------------
    # Retrieval internals
    # ---------------------------
    def _top_folders(self, query: str, n: int = 3) -> List[int]:
        qv = hashed_embedding(query, dim=self.dim)
        cur = self.conn.execute("SELECT id, emb FROM folders")
        scored: List[Tuple[float, int]] = []
        for fid, emb_blob in cur.fetchall():
            femb = np.frombuffer(emb_blob, dtype=np.float32)
            scored.append((float(np.dot(qv, femb)), int(fid)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fid for _, fid in scored[:n]]

    def _lexical_candidates(
        self,
        query: str,
        limit: int = 400,
        company: Optional[str] = None,
        folder_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Fast candidate generation using inverted index.
        Score by token overlap count (rough BM25-ish proxy).
        """
        toks = tokenize(query)
        if not toks:
            return []

        # Reduce query tokens to avoid huge IN (...) lists
        toks = toks[:80]
        placeholders = ",".join("?" * len(toks))

        # Filter via joins to memory_items for company/folder if needed
        where_clauses = []
        params: List[Any] = list(toks)

        join_items = False
        if company is not None:
            join_items = True
            where_clauses.append("mi.company = ?")
            params.append(company)

        if folder_ids:
            join_items = True
            ph2 = ",".join("?" * len(folder_ids))
            where_clauses.append(f"mi.folder_id IN ({ph2})")
            params.extend(folder_ids)

        where_sql = ""
        if join_items:
            where_sql = "AND " + " AND ".join(where_clauses)

        sql = f"""
            SELECT ti.item_id, SUM(ti.tf) AS overlap
            FROM token_index ti
            {"JOIN memory_items mi ON mi.id = ti.item_id" if join_items else ""}
            WHERE ti.token IN ({placeholders})
            {where_sql}
            GROUP BY ti.item_id
            ORDER BY overlap DESC
            LIMIT {int(limit)}
        """

        cur = self.conn.execute(sql, tuple(params))
        return [int(r[0]) for r in cur.fetchall()]

    def _fetch_items_by_ids(self, ids: List[int]) -> List[Tuple]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cur = self.conn.execute(
            f"""
            SELECT m.id, f.name, m.text, m.summary, m.status, m.emb, m.ts, m.meta, m.company
            FROM memory_items m
            JOIN folders f ON m.folder_id = f.id
            WHERE m.id IN ({placeholders})
            """,
            tuple(ids),
        )
        return cur.fetchall()

    def retrieve(
        self,
        query: str,
        k: int = 10,
        company: Optional[str] = None,
        folder_hint: Optional[str] = None,
        allow_company_backoff: bool = True,
    ) -> List[MemoryItem]:
        """
        Hybrid retrieval with company filtering.

        - company filter is strong by default; if too few candidates, can back off to global.
        - folder_hint can constrain retrieval to a folder (task type).
        """
        # determine folders
        folder_ids: Optional[List[int]] = None
        if folder_hint:
            cur = self.conn.execute("SELECT id FROM folders WHERE name=?", (folder_hint,))
            row = cur.fetchone()
            folder_ids = [int(row[0])] if row else []
        else:
            folder_ids = self._top_folders(query, n=3)

        # candidate generation: try company-filtered first
        cand_ids = self._lexical_candidates(query, limit=600, company=company, folder_ids=folder_ids)

        # backoff if too few
        if allow_company_backoff and company and len(cand_ids) < max(20, k * 3):
            cand_ids = self._lexical_candidates(query, limit=600, company=None, folder_ids=folder_ids)

        # fetch rows
        rows = self._fetch_items_by_ids(cand_ids[:800])  # hard cap

        if not rows:
            return []

        # scoring
        q_tokens = tokenize(query)
        q_set = set(q_tokens[:200])
        q_vec = hashed_embedding_from_tokens(q_tokens, dim=self.dim)
        now = _now_ts()

        items: List[MemoryItem] = []
        for rid, folder_name, text, summary, status, emb_blob, ts, meta_json, comp in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            vec_sim = float(np.dot(q_vec, emb))

            # lexical score: normalized overlap
            t_set = set(tokenize(text)[:300])
            overlap = len(q_set.intersection(t_set))
            lex = overlap / max(30.0, math.sqrt(len(q_set) * max(1, len(t_set))))

            # recency
            age = max(0.0, now - float(ts))
            rec = math.exp(-0.0001 * age)

            # combine
            score = 0.62 * vec_sim + 0.25 * lex + 0.13 * rec

            try:
                meta = json.loads(meta_json) if meta_json else {}
            except Exception:
                meta = {}

            items.append(
                MemoryItem(
                    id=int(rid),
                    folder=str(folder_name),
                    text=str(text),
                    summary=str(summary or ""),
                    status=str(status),
                    score=score,
                    company=str(comp) if comp else None,
                    meta=meta,
                )
            )

        items.sort(key=lambda x: x.score, reverse=True)

        # 80/20 worked/failed
        worked = [it for it in items if it.status == "worked"]
        failed = [it for it in items if it.status == "failed"]

        selected: List[MemoryItem] = []
        if worked:
            k_worked = max(1, int(k * 0.8))
            k_failed = k - k_worked

            selected.extend(worked[:k_worked])
            selected.extend(failed[:k_failed])
            selected.sort(key=lambda x: x.score, reverse=True)
            selected = selected[:k]
        else:
            selected = items[:k]

        # compression: keep top few raw
        raw_limit = 3
        for i, it in enumerate(selected):
            it.is_compressed = i >= raw_limit

        return selected

    def format_memories(self, items: List[MemoryItem]) -> str:
        if not items:
            return ""

        blocks = []
        for it in items:
            header = f"[{it.folder.upper()} | {it.status.upper()}"
            if it.company:
                header += f" | COMPANY={it.company}"
            header += "]"

            content = it.summary if it.is_compressed else it.text
            if it.is_compressed:
                blocks.append(f"{header} (Summarized): {content}")
            else:
                blocks.append(f"{header}:\n{content}")

        return "\n\n".join(blocks)
