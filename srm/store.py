"""
srm/store.py — SQLite-backed persistent memory store.

Design:
  - Texts are stored in SQLite; binary codes are derived in-process
    (never persisted) because IDF changes every time a memory is added.
  - The in-memory cache (IDF + codes) is invalidated on every write,
    ensuring the Hamming codes always reflect the current corpus IDF.
  - Thread safety: SQLite is opened with check_same_thread=False so the
    store can be shared across a simple web API layer if needed.

Schema:
    memories(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT UNIQUE NOT NULL)
"""

from __future__ import annotations

import sqlite3

import numpy as np

from .config import DB_PATH, PACK_BYTES
from .nlp import idf_table
from .encoding import encode_batch


class MemoryStore:
    """
    Persistent memory store backed by SQLite.

    Public API:
        add(text)      → bool   add a memory, returns False on duplicate
        delete(mem_id) → bool   remove by DB primary-key id
        clear()                 wipe all memories
        count()        → int    number of stored memories
        load_all()     → (ids, texts)
        get_idf()      → dict   corpus IDF table (cached)
        get_codes()    → ndarray  shape (N, PACK_BYTES) (cached)
        close()                 close the SQLite connection
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self.conn    = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT UNIQUE NOT NULL
            )""")
        self.conn.commit()
        self._flush_cache()

    # ── Cache management ──────────────────────────────────────────────

    def _flush_cache(self) -> None:
        """Invalidate all in-process caches."""
        self._ids:   list[int] | None        = None
        self._texts: list[str] | None        = None
        self._idf:   dict[str, float] | None = None
        self._codes: np.ndarray | None       = None

    # ── Reads ─────────────────────────────────────────────────────────

    def load_all(self) -> tuple[list[int], list[str]]:
        """Return (ids, texts) for all memories, ordered by insertion id."""
        if self._texts is None:
            rows        = self.conn.execute(
                "SELECT id, text FROM memories ORDER BY id"
            ).fetchall()
            self._ids   = [r[0] for r in rows]
            self._texts = [r[1] for r in rows]
        return self._ids, self._texts  # type: ignore[return-value]

    def get_idf(self) -> dict[str, float]:
        """Corpus IDF table, lazily computed and cached."""
        if self._idf is None:
            _, texts   = self.load_all()
            self._idf  = idf_table(texts)
        return self._idf

    def get_codes(self, meaning_db=None) -> np.ndarray:
        """
        Binary codes for all memories, shape (N, PACK_BYTES).

        Lazily derived from current texts + IDF; re-derived after any write.
        When meaning_db is provided, verb polarity masking and adjective
        weight scaling are applied. The two variants are cached separately.
        """
        cache_attr = '_codes_meaning' if meaning_db is not None else '_codes'
        if getattr(self, cache_attr, None) is None:
            idf      = self.get_idf()
            _, texts = self.load_all()
            codes = (
                encode_batch(texts, idf, meaning_db=meaning_db)
                if texts
                else np.empty((0, PACK_BYTES), dtype=np.uint8)
            )
            setattr(self, cache_attr, codes)
        return getattr(self, cache_attr)

    def count(self) -> int:
        """Number of stored memories (direct DB count, no cache)."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]

    # ── Writes ────────────────────────────────────────────────────────

    def add(self, text: str) -> bool:
        """
        Store *text* as a new memory.

        Returns True on success, False if the text already exists
        (duplicate detection via SQLite UNIQUE constraint).
        """
        text = text.strip()
        if not text:
            return False
        try:
            self.conn.execute(
                "INSERT INTO memories (text) VALUES (?)", (text,)
            )
            self.conn.commit()
            self._flush_cache()
            return True
        except sqlite3.IntegrityError:
            return False

    def delete(self, mem_id: int) -> bool:
        """
        Delete the memory with the given primary-key *mem_id*.

        Returns True if a row was removed, False if the id was not found.
        """
        cur = self.conn.execute(
            "DELETE FROM memories WHERE id=?", (mem_id,)
        )
        self.conn.commit()
        self._flush_cache()
        return cur.rowcount > 0

    def clear(self) -> None:
        """Delete all memories."""
        self.conn.execute("DELETE FROM memories")
        self.conn.commit()
        self._flush_cache()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.conn.close()

    def __repr__(self) -> str:
        return f"MemoryStore(db={self.db_path!r}, count={self.count()})"
