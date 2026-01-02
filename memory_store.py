"""
Clustered Memory Store (SQLite) — folders are learned clusters with fixed prototypes.

Defaults:
- Prototypes per folder (M): 4
- New folder creation threshold (THRESH): 0.30  (for hashed embeddings)
- Retrieval top folders: 4
- Keep top 3 memories raw, rest summarized

Key idea:
- Items are assigned to a "folder" (cluster) based on similarity to folder prototypes.
- Each folder maintains M prototype vectors (exemplars) chosen online.
- Retrieval: score folders by max dot(query, prototype_j), select top folders,
  then score items within them.

This removes the brittle "folder = task type" assumption and improves
consultant/client playbook retrieval across task types.

NOTE: This uses deterministic hashed embeddings. You can later swap in real embeddings.
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
import tiktoken
import numpy as np
import re


# -----------------------
# Defaults
# -----------------------
DIM = 256
M_PROTOS = 4
NEW_FOLDER_THRESH = 0.30
TOP_FOLDERS = 4
RAW_LIMIT = 3

# Prototype update behavior
# If the new item is dissimilar to all prototypes, it can replace one.
PROTO_REPLACE_SIM_CEIL = 0.75  # if max_sim < this, we consider replacing a prototype


def _now_ts() -> float:
    return time.time()


# -----------------------
# Tokenization (durable)
# -----------------------
@lru_cache(maxsize=4)
def _get_encoder(name: str = "cl100k_base"):
    return tiktoken.get_encoding(name)


def tokenize(text: str) -> List[str]:
    """
    Durable tokenizer:
    - Prefer tiktoken token IDs (stable across punctuation/JSON/symbols).
    - Fallback to regex if tiktoken isn't installed.
    Returns list[str] tokens.
    """
    text = text or ""
    try:
        enc = _get_encoder("cl100k_base")
        ids = enc.encode(text)
        return [f"t{tid}" for tid in ids]
    except Exception:
        
        return re.findall(r"[a-zA-Z0-9]+", text.lower())


# -----------------------
# Embeddings (deterministic hashed)
# -----------------------
def hashed_embedding_from_tokens(tokens: List[str], dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens:
        return vec
    for tok in tokens:
        h = zlib.adler32(tok.encode("utf-8"))
        idx = h % dim
        sign = -1.0 if (h & 1) else 1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def hashed_embedding(text: str, dim: int = DIM) -> np.ndarray:
    return hashed_embedding_from_tokens(tokenize(text), dim=dim)


# -----------------------
# Data model
# -----------------------
@dataclass
class MemoryItem:
    id: int
    folder_id: int
    folder_name: str
    text: str
    summary: str
    status: str  # "worked" | "failed"
    score: float = 0.0
    is_compressed: bool = False
    meta: Optional[Dict[str, Any]] = None
    company: Optional[str] = None


# -----------------------
# Store
# -----------------------
class MemoryStore:
    def __init__(
        self,
        db_path: str = "memory.sqlite",
        dim: int = DIM,
        m_protos: int = M_PROTOS,
        new_folder_thresh: float = NEW_FOLDER_THRESH,
        top_folders: int = TOP_FOLDERS,
        raw_limit: int = RAW_LIMIT,
        company_backoff: bool = True,
    ):
        self.dim = dim
        self.m_protos = m_protos
        self.new_folder_thresh = new_folder_thresh
        self.top_folders = top_folders
        self.raw_limit = raw_limit
        self.company_backoff = company_backoff

        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_db()
        self._migrate_db()

    def _init_db(self) -> None:
        # Cluster folders (learned)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mem_folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                meta TEXT,
                created_ts REAL
            )
            """
        )

        # Fixed prototype slots per folder
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS folder_prototypes (
                folder_id INTEGER,
                slot INTEGER,
                emb BLOB,
                item_id INTEGER,
                updated_ts REAL,
                PRIMARY KEY(folder_id, slot),
                FOREIGN KEY(folder_id) REFERENCES mem_folders(id)
            )
            """
        )

        # Memory items
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_id INTEGER,
                text TEXT,
                summary TEXT,
                status TEXT,
                ts REAL,
                emb BLOB,
                meta TEXT,
                company TEXT,
                FOREIGN KEY(folder_id) REFERENCES mem_folders(id)
            )
            """
        )

        # Helpful indices
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_folder   ON memory_items(folder_id);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_company  ON memory_items(company);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_status   ON memory_items(status);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_proto_folder   ON folder_prototypes(folder_id);")
        self.conn.commit()

    def _migrate_db(self) -> None:
        """
        Minimal migrations for older DBs: add missing cols if any.
        """
        cur = self.conn.execute("PRAGMA table_info(memory_items)")
        cols = {r[1] for r in cur.fetchall()}

        def add_col(col: str, typ: str):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE memory_items ADD COLUMN {col} {typ}")

        add_col("meta", "TEXT")
        add_col("company", "TEXT")

        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_company  ON memory_items(company);")
        self.conn.commit()

    # -----------------------
    # Folder/prototype helpers
    # -----------------------
    def _create_folder(self, name: str = "", meta: Optional[Dict[str, Any]] = None) -> int:
        meta = meta or {}
        cur = self.conn.execute(
            "INSERT INTO mem_folders (name, meta, created_ts) VALUES (?, ?, ?)",
            (name, json.dumps(meta), _now_ts()),
        )
        folder_id = int(cur.lastrowid)

        # Initialize empty prototypes slots
        for slot in range(self.m_protos):
            self.conn.execute(
                "INSERT OR REPLACE INTO folder_prototypes (folder_id, slot, emb, item_id, updated_ts) VALUES (?, ?, ?, ?, ?)",
                (folder_id, slot, None, None, _now_ts()),
            )
        self.conn.commit()
        return folder_id

    def _get_folder_protos(self, folder_id: int) -> List[Tuple[int, Optional[np.ndarray], Optional[int], float]]:
        """
        Returns list of (slot, emb_or_none, item_id_or_none, updated_ts)
        """
        cur = self.conn.execute(
            "SELECT slot, emb, item_id, updated_ts FROM folder_prototypes WHERE folder_id=? ORDER BY slot ASC",
            (folder_id,),
        )
        out = []
        for slot, emb_blob, item_id, uts in cur.fetchall():
            emb = None
            if emb_blob is not None:
                emb = np.frombuffer(emb_blob, dtype=np.float32)
            out.append((int(slot), emb, int(item_id) if item_id is not None else None, float(uts)))
        return out

    def _folder_score(self, q_vec: np.ndarray, protos: List[Tuple[int, Optional[np.ndarray], Optional[int], float]]) -> float:
        """
        Score folder by max dot(q, proto) over non-empty prototypes.
        """
        best = -1.0
        any_proto = False
        for _, emb, _, _ in protos:
            if emb is None:
                continue
            any_proto = True
            s = float(np.dot(q_vec, emb))
            if s > best:
                best = s
        if not any_proto:
            return 0.0
        return best

    def _select_folder(
        self,
        q_vec: np.ndarray,
        meta: Dict[str, Any],
        company: Optional[str],
    ) -> int:
        """
        Choose best folder (cluster) for insertion:
        - Prefer folders matching company meta if company is provided.
        - Score by prototypes max similarity.
        - If best score < threshold, create a new folder.
        """
        cur = self.conn.execute("SELECT id, name, meta FROM mem_folders")
        rows = cur.fetchall()
        if not rows:
            return self._create_folder(name=(company or ""), meta={"company": company} if company else {})

        # Build candidate folder list
        candidates: List[Tuple[float, int, str]] = []

        for fid, fname, fmeta_json in rows:
            fid = int(fid)
            try:
                fmeta = json.loads(fmeta_json) if fmeta_json else {}
            except Exception:
                fmeta = {}

            # Strong preference: same company if company exists
            if company is not None:
                f_company = fmeta.get("company")
                if f_company is not None and f_company != company:
                    # skip company-mismatched folders for insertion
                    continue

            protos = self._get_folder_protos(fid)
            score = self._folder_score(q_vec, protos)

            candidates.append((score, fid, fname or ""))

        # If company constrained and none exist, create new
        if not candidates:
            return self._create_folder(name=(company or ""), meta={"company": company} if company else {})

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_fid, _ = candidates[0]

        if best_score < self.new_folder_thresh:
            return self._create_folder(name=(company or ""), meta={"company": company} if company else {})

        return best_fid

    def _update_folder_prototypes(self, folder_id: int, item_id: int, item_vec: np.ndarray) -> None:
        """
        Maintain fixed M prototypes per folder using an online exemplar strategy:

        - If an empty slot exists => fill it.
        - Else compute similarity to existing prototypes:
            - If the item is sufficiently "novel" (max_sim < PROTO_REPLACE_SIM_CEIL),
              replace the prototype with the lowest similarity to this item (improves coverage).
            - Otherwise do nothing (item is redundant).
        """
        protos = self._get_folder_protos(folder_id)

        # 1) fill empty slot if any
        for slot, emb, _, _ in protos:
            if emb is None:
                self.conn.execute(
                    "UPDATE folder_prototypes SET emb=?, item_id=?, updated_ts=? WHERE folder_id=? AND slot=?",
                    (item_vec.tobytes(), item_id, _now_ts(), folder_id, slot),
                )
                self.conn.commit()
                return

        # 2) compute similarity to each prototype
        sims: List[Tuple[float, int]] = []
        for slot, emb, _, _ in protos:
            assert emb is not None
            sims.append((float(np.dot(item_vec, emb)), slot))

        sims.sort(reverse=True)
        max_sim = sims[0][0]

        if max_sim >= PROTO_REPLACE_SIM_CEIL:
            # too similar; keep prototypes as-is
            return

        # replace the proto most "different" from this item (min sim)
        min_sim, replace_slot = min(sims, key=lambda x: x[0])

        self.conn.execute(
            "UPDATE folder_prototypes SET emb=?, item_id=?, updated_ts=? WHERE folder_id=? AND slot=?",
            (item_vec.tobytes(), item_id, _now_ts(), folder_id, replace_slot),
        )
        self.conn.commit()

    # -----------------------
    # Public API
    # -----------------------
    def add(
        self,
        folder_name: str,     # kept for API compatibility; stored into meta as task_type
        text: str,
        summary: str,
        status: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert a new memory item:
        - Auto-assign to a learned folder based on similarity to folder prototypes.
        - Maintain folder prototypes (fixed size).
        """
        meta = meta or {}
        # preserve old usage: treat folder_name as task_type hint
        if "task_type" not in meta and folder_name:
            meta["task_type"] = folder_name

        company = meta.get("company")
        item_vec = hashed_embedding(text, dim=self.dim)

        folder_id = self._select_folder(item_vec, meta=meta, company=company)

        cur = self.conn.execute(
            """
            INSERT INTO memory_items (folder_id, text, summary, status, ts, emb, meta, company)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (folder_id, text, summary, status, _now_ts(), item_vec.tobytes(), json.dumps(meta), company),
        )
        item_id = int(cur.lastrowid)
        self.conn.commit()

        # update prototypes for the folder
        self._update_folder_prototypes(folder_id, item_id, item_vec)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        company: Optional[str] = None,
    ) -> List[MemoryItem]:
        """
        Retrieve memories:
        1) Score folders by query similarity to folder prototypes (max over slots).
        2) Select top folders (company-filtered; backoff optional).
        3) Fetch items from these folders and score by sim+recency.
        4) Select 80% worked / 20% failed.
        5) Compress lower-ranked items.
        """
        q_vec = hashed_embedding(query, dim=self.dim)

        # Step 1: score folders
        cur = self.conn.execute("SELECT id, name, meta FROM mem_folders")
        folder_rows = cur.fetchall()

        scored_folders: List[Tuple[float, int, str]] = []
        scored_folders_any: List[Tuple[float, int, str]] = []

        for fid, fname, fmeta_json in folder_rows:
            fid = int(fid)
            fname = fname or ""
            try:
                fmeta = json.loads(fmeta_json) if fmeta_json else {}
            except Exception:
                fmeta = {}

            protos = self._get_folder_protos(fid)
            score = self._folder_score(q_vec, protos)

            scored_folders_any.append((score, fid, fname))

            if company is not None:
                if fmeta.get("company") == company:
                    scored_folders.append((score, fid, fname))
            else:
                scored_folders.append((score, fid, fname))

        # company backoff if nothing matches
        if company is not None and self.company_backoff and not scored_folders:
            scored_folders = scored_folders_any

        scored_folders.sort(key=lambda x: x[0], reverse=True)
        top_folder_ids = [fid for _, fid, _ in scored_folders[: self.top_folders]]

        if not top_folder_ids:
            return []

        # Step 2: fetch items from top folders
        placeholders = ",".join("?" * len(top_folder_ids))
        cur = self.conn.execute(
            f"""
            SELECT id, folder_id, text, summary, status, ts, emb, meta, company
            FROM memory_items
            WHERE folder_id IN ({placeholders})
            """,
            tuple(top_folder_ids),
        )
        rows = cur.fetchall()
        if not rows:
            return []

        # Step 3: score items
        now = _now_ts()
        candidates: List[MemoryItem] = []

        # Folder name map
        f_map: Dict[int, str] = {fid: name for _, fid, name in scored_folders}

        for rid, fid, text, summary, status, ts, emb_blob, meta_json, comp in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            sim = float(np.dot(q_vec, emb))

            age = max(0.0, now - float(ts))
            recency = math.exp(-0.0001 * age)

            score = 0.7 * sim + 0.3 * recency

            try:
                meta = json.loads(meta_json) if meta_json else {}
            except Exception:
                meta = {}

            candidates.append(
                MemoryItem(
                    id=int(rid),
                    folder_id=int(fid),
                    folder_name=f_map.get(int(fid), f"folder_{fid}"),
                    text=str(text),
                    summary=str(summary or ""),
                    status=str(status),
                    score=score,
                    meta=meta,
                    company=str(comp) if comp else None,
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)

        # Step 4: 80/20 worked/failed
        worked = [c for c in candidates if c.status == "worked"]
        failed = [c for c in candidates if c.status == "failed"]

        if worked:
            k_worked = max(1, int(k * 0.8))
            k_failed = k - k_worked
            selected = worked[:k_worked] + failed[:k_failed]
            selected.sort(key=lambda x: x.score, reverse=True)
            selected = selected[:k]
        else:
            selected = candidates[:k]

        # Step 5: compression
        for i, item in enumerate(selected):
            item.is_compressed = i >= self.raw_limit

        return selected

    def format_memories(self, items: List[MemoryItem]) -> str:
        if not items:
            return ""

        blocks = []
        for it in items:
            header = f"[CLUSTER={it.folder_id} | {it.status.upper()}"
            if it.company:
                header += f" | COMPANY={it.company}"
            # show task_type if present (helpful to the model)
            tt = (it.meta or {}).get("task_type")
            if tt:
                header += f" | TYPE={tt}"
            header += "]"

            content = it.summary if it.is_compressed else it.text
            if it.is_compressed:
                blocks.append(f"{header} (Summarized): {content}")
            else:
                blocks.append(f"{header}:\n{content}")

        return "\n\n".join(blocks)
