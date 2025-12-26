"""
Persistent hierarchical memory store (SQLite) with simple vector embeddings.
FIXED: Deterministic hashing and improved retrieval scoring.
"""
from __future__ import annotations

import os
import re
import json
import time
import math
import sqlite3
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def _now_ts() -> float:
    return time.time()

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", text.lower())

def hashed_embedding(text: str, dim: int = 256) -> np.ndarray:
    """
    Deterministic, cheap embedding using Adler32 for cross-session consistency.
    """
    vec = np.zeros(dim, dtype=np.float32)
    toks = _tokenize(text)
    for tok in toks:
        # Use zlib.adler32 for a deterministic hash across Python restarts
        h = zlib.adler32(tok.encode("utf-8"))
        idx = h % dim
        sign = -1.0 if (h & 1) else 1.0
        vec[idx] += sign
    
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec

def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))

@dataclass
class MemoryItem:
    id:     int
    level:  str
    text:   str
    ts:     float
    util:   float
    meta:   Dict[str, Any]
    emb:    np.ndarray

class MemoryStore:
    def __init__(self, path: str, emb_dim: int = 256):
        self.path = path
        self.emb_dim = emb_dim
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                text TEXT NOT NULL,
                ts REAL NOT NULL,
                util REAL NOT NULL,
                meta_json TEXT NOT NULL,
                emb BLOB NOT NULL,
                len_tokens INTEGER NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_level_ts ON memory_items(level, ts);")
        self.conn.commit()

    def add(self, level: str, text: str, util: float = 0.0, meta: Optional[Dict[str, Any]] = None) -> int:
        meta = meta or {}
        emb = hashed_embedding(text, dim=self.emb_dim)
        emb_blob = emb.tobytes()
        ts = _now_ts()
        lt = approx_tokens(text)
        cur = self.conn.execute(
            "INSERT INTO memory_items(level, text, ts, util, meta_json, emb, len_tokens) VALUES(?,?,?,?,?,?,?)",
            (level, text, ts, float(util), json.dumps(meta, ensure_ascii=False), emb_blob, lt),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def retrieve(
        self,
        query: str,
        k: int = 8,
        token_budget: int = 800,
        alpha: float = 1.2, # Slightly higher weight on similarity
        beta: float = 0.1,
        gamma: float = 0.3, # Prioritize items marked as useful
        delta: float = 0.001,
        lambda_recency: float = 0.0001,
        level_filter: Optional[List[str]] = None,
    ) -> List[MemoryItem]:
        q = hashed_embedding(query, dim=self.emb_dim)
        now = _now_ts()

        if level_filter:
            qmarks = ",".join("?" for _ in level_filter)
            cur = self.conn.execute(
                f"SELECT id, level, text, ts, util, meta_json, emb, len_tokens FROM memory_items WHERE level IN ({qmarks})",
                tuple(level_filter),
            )
        else:
            cur = self.conn.execute(
                "SELECT id, level, text, ts, util, meta_json, emb, len_tokens FROM memory_items"
            )
        rows = cur.fetchall()

        scored: List[Tuple[float, MemoryItem, int]] = []
        for row in rows:
            (id_, level, text, ts, util, meta_json, emb_blob, len_tokens) = row
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            cos = float(np.dot(q, emb))
            age = max(0.0, now - float(ts))
            rec = math.exp(-lambda_recency * age)
            # Scoring includes a small penalty for token length to prefer concise memories
            score = alpha * cos + beta * rec + gamma * float(util) - delta * float(len_tokens)
            meta = json.loads(meta_json) if meta_json else {}
            item = MemoryItem(id=id_, level=level, text=text, ts=float(ts), util=float(util), meta=meta, emb=emb)
            scored.append((score, item, int(len_tokens)))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[MemoryItem] = []
        used = 0
        for _score, item, lt in scored:
            if used + lt > token_budget: continue
            out.append(item)
            used += lt
            if len(out) >= k: break
        return out

    def close(self) -> None:
        self.conn.close()