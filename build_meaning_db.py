#!/usr/bin/env python3
"""
scripts/build_meaning_db.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ingests a directory of verb-semantic JSON files (one per 2-letter prefix,
e.g. ag.json, ab.json …) and builds a SQLite meaning database for SRM.

verbs can be found at C:\dev\nameless_vector\verb_state

Schema overview
───────────────
  polarity_classes     12 semantic direction classes, each with a
                       pre-computed 128-bit bitmask that is nearly
                       orthogonal to all others (hamming ≈ 64).

  verbs                canonical verb entries (verb text, source prefix).

  verb_polarity        verb → polarity_class mapping, derived from goals
                       and final_object_states via keyword scoring.

  verb_goals           flat {verb_id, goal} rows.
  verb_mechanisms      flat {verb_id, mechanism} rows.
  verb_tools           flat {verb_id, tool} rows.
  verb_applicability   {verb_id, role, entity_type} — subject/object types.
  state_transforms     the full state-change record:
                       {verb_id, role, phase, dimension, state_value}
                       role  ∈ {subject, object}
                       phase ∈ {required, final}
                       dimension ∈ {physical, emotional, mental, positional}

Usage
─────
  python scripts/build_meaning_db.py --src data/verbs/ --db meaning.db
  python scripts/build_meaning_db.py --src data/verbs/ --db meaning.db --report

"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
import hashlib
import argparse
import textwrap
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

STATE_DIM_WEIGHTS: dict[str, float] = {
    "physical": 3.0,
    "emotional": 2.0,
    "mental": 1.0,
    "positional": 1.0,
}

_SKIP_STATES = frozenset({
    "non-applicable",
    "non applicable",
    "unchanged",
    "neutral",
})


def _final_state_terms(final_obj_states: dict | None) -> list[tuple[str, float]]:
    if not isinstance(final_obj_states, dict):
        return []
    out: list[tuple[str, float]] = []
    for dim, values in final_obj_states.items():
        weight = STATE_DIM_WEIGHTS.get(dim, 1.0)
        if isinstance(values, list):
            seq = values
        elif values is None:
            seq = []
        else:
            seq = [values]
        for sv in seq:
            text = str(sv).strip().lower()
            if text and text not in _SKIP_STATES:
                out.append((text, weight))
    return out


def _mask_from_outcome(outcome: dict) -> bytes:
    """SimHash final object state words directly into a 128-bit polarity mask."""
    weights = np.zeros(128, dtype=np.float64)

    for state_value, weight in _final_state_terms(outcome.get("final_object_states")):
        raw = hashlib.md5(state_value.encode("utf-8")).digest()
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
        weights += np.where(bits, weight, -weight)

    if not weights.any():
        verb = str(outcome.get("verb", "unknown")).strip().lower() or "unknown"
        raw = hashlib.md5(verb.encode("utf-8")).digest()
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
        weights += np.where(bits, 1.0, -1.0)

    return np.packbits((weights > 0).astype(np.uint8)).tobytes()


def _mask_from_states(final_obj_states: dict | None) -> bytes:
    """Backwards-compatible helper used by tests: derive mask from final_object_states."""
    return _mask_from_outcome({"verb": "_", "final_object_states": final_obj_states or {}})


def _label_from_states(final_obj_states: dict | None) -> str:
    terms = [state for state, _ in _final_state_terms(final_obj_states)]
    if not terms:
        return "state-mask"
    return " / ".join(terms[:3])


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS verbs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    verb    TEXT UNIQUE NOT NULL,
    prefix  TEXT NOT NULL,
    polarity_label TEXT NOT NULL DEFAULT 'state-mask',
    polarity_mask BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS verb_goals (
    verb_id INTEGER NOT NULL REFERENCES verbs(id),
    goal    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verb_mechanisms (
    verb_id   INTEGER NOT NULL REFERENCES verbs(id),
    mechanism TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verb_tools (
    verb_id INTEGER NOT NULL REFERENCES verbs(id),
    tool    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verb_applicability (
    verb_id     INTEGER NOT NULL REFERENCES verbs(id),
    role        TEXT NOT NULL CHECK(role IN ('subject','object')),
    entity_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state_transforms (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    verb_id     INTEGER NOT NULL REFERENCES verbs(id),
    role        TEXT NOT NULL CHECK(role IN ('subject','object')),
    phase       TEXT NOT NULL CHECK(phase IN ('required','final')),
    dimension   TEXT NOT NULL CHECK(dimension IN ('physical','emotional','mental','positional')),
    state_value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_verbs_verb            ON verbs(verb);
CREATE INDEX IF NOT EXISTS idx_state_transforms_verb ON state_transforms(verb_id);
CREATE INDEX IF NOT EXISTS idx_verb_goals_verb       ON verb_goals(verb_id);

-- Flat view for state delta inspection
CREATE VIEW IF NOT EXISTS verb_state_delta AS
SELECT  v.verb,
        st.role,
        st.phase,
        st.dimension,
        st.state_value
FROM    verbs v
JOIN    state_transforms st ON st.verb_id = v.id;
"""


# ── Loader ────────────────────────────────────────────────────────────────────

def load_json_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _iter_outcomes(node) -> list[dict]:
    """Return a flat list of outcome dicts from potentially nested JSON."""
    out: list[dict] = []
    if node is None:
        return out
    if isinstance(node, dict):
        out.append(node)
        return out
    if isinstance(node, list):
        for item in node:
            out.extend(_iter_outcomes(item))
    return out


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def ingest_file(conn: sqlite3.Connection,
                data: dict,
                stats: Counter) -> None:
    """Insert all verb data from one parsed JSON file."""
    prefix = data.get("prefix", "??")
    allowed_dims = {"physical", "emotional", "mental", "positional"}

    outcomes = _iter_outcomes(data.get("outcomes", []))
    for outcome in outcomes:
        verb = str(outcome.get("verb", "")).strip().lower()
        if not verb:
            continue

        # ── verbs table ──────────────────────────────────────────────
        try:
            polarity_label = _label_from_states(outcome.get("final_object_states", {}))
            polarity_mask = _mask_from_outcome(outcome)
            cur = conn.execute(
                "INSERT INTO verbs (verb, prefix, polarity_label, polarity_mask) VALUES (?,?,?,?)",
                (verb, prefix, polarity_label, polarity_mask),
            )
            verb_id = cur.lastrowid
            stats["inserted"] += 1
        except sqlite3.IntegrityError:
            verb_id = conn.execute(
                "SELECT id FROM verbs WHERE verb=?", (verb,)
            ).fetchone()[0]
            stats["skipped_dup"] += 1
            continue

        # ── goals / mechanisms / tools ────────────────────────────────
        for g in _as_list(outcome.get("goals", [])):
            conn.execute(
                "INSERT INTO verb_goals (verb_id, goal) VALUES (?,?)", (verb_id, g)
            )
        for m in _as_list(outcome.get("mechanisms", [])):
            conn.execute(
                "INSERT INTO verb_mechanisms (verb_id, mechanism) VALUES (?,?)",
                (verb_id, m),
            )
        for t in _as_list(outcome.get("tools", [])):
            if t:
                conn.execute(
                    "INSERT INTO verb_tools (verb_id, tool) VALUES (?,?)", (verb_id, t)
                )

        # ── applicability ─────────────────────────────────────────────
        for etype in _as_list(outcome.get("applicable_subjects", [])):
            conn.execute(
                "INSERT INTO verb_applicability (verb_id, role, entity_type) "
                "VALUES (?,?,?)",
                (verb_id, "subject", etype),
            )
        for etype in _as_list(outcome.get("applicable_objects", [])):
            conn.execute(
                "INSERT INTO verb_applicability (verb_id, role, entity_type) "
                "VALUES (?,?,?)",
                (verb_id, "object", etype),
            )

        # ── state transforms ──────────────────────────────────────────
        phase_keys = {
            "required": ("required_subject_states", "required_object_states"),
            "final":    ("final_subject_states",    "final_object_states"),
        }
        for phase, (subj_key, obj_key) in phase_keys.items():
            subj_states = outcome.get(subj_key, {})
            if isinstance(subj_states, dict):
                for dim, values in subj_states.items():
                    if dim not in allowed_dims:
                        stats["skipped_dim"] += 1
                        continue
                    for sv in _as_list(values):
                        sv = str(sv).strip()
                        if sv and sv.lower() != "non-applicable":
                            conn.execute(
                                "INSERT INTO state_transforms "
                                "(verb_id,role,phase,dimension,state_value) "
                                "VALUES (?,?,?,?,?)",
                                (verb_id, "subject", phase, dim, sv),
                            )

            obj_states = outcome.get(obj_key, {})
            if isinstance(obj_states, dict):
                for dim, values in obj_states.items():
                    if dim not in allowed_dims:
                        stats["skipped_dim"] += 1
                        continue
                    for sv in _as_list(values):
                        sv = str(sv).strip()
                        if sv and sv.lower() != "non-applicable":
                            conn.execute(
                                "INSERT INTO state_transforms "
                                "(verb_id,role,phase,dimension,state_value) "
                                "VALUES (?,?,?,?,?)",
                                (verb_id, "object", phase, dim, sv),
                            )


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(conn: sqlite3.Connection) -> None:
    print("\n  ── Meaning DB report ──────────────────────────────────────")
    n_verbs = conn.execute("SELECT COUNT(*) FROM verbs").fetchone()[0]
    n_states = conn.execute("SELECT COUNT(*) FROM state_transforms").fetchone()[0]
    print(f"  verbs loaded     : {n_verbs}")
    print(f"  state transforms : {n_states}")

    print("\n  Sample state labels:")
    rows = conn.execute("""
        SELECT polarity_label, COUNT(*) as n
        FROM   verbs
        GROUP  BY polarity_label
        ORDER  BY n DESC, polarity_label ASC
        LIMIT 20
    """).fetchall()
    for name, n in rows:
        print(f"    {name:<28}  {n}")

    print("\n  Sample verb meanings:")
    sample = conn.execute("""
        SELECT verb, polarity_label
        FROM   verbs
        ORDER  BY verb
        LIMIT  20
    """).fetchall()
    for verb, cls in sample:
        print(f"    {verb:<18}  →  {cls}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def build(src_dir: str, db_path: str, report: bool = False) -> None:
    src = Path(src_dir)
    json_files = sorted(src.glob("*.json"))
    if not json_files:
        print(f"  No .json files found in {src_dir!r}")
        sys.exit(1)

    print(f"  Found {len(json_files)} JSON file(s) in {src_dir!r}")

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(verbs)").fetchall()}
    if "polarity_label" not in cols:
        conn.execute("ALTER TABLE verbs ADD COLUMN polarity_label TEXT NOT NULL DEFAULT 'state-mask'")
    if "polarity_mask" not in cols:
        conn.execute("ALTER TABLE verbs ADD COLUMN polarity_mask BLOB")
    conn.commit()

    # ── ingest each file ───────────────────────────────────────────────
    stats: Counter = Counter()
    for jf in json_files:
        try:
            data = load_json_file(jf)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  \033[33m  skipping {jf.name}: {e}\033[0m")
            stats["file_errors"] += 1
            continue

        ingest_file(conn, data, stats)
        conn.commit()
        outcome_count = len(_iter_outcomes(data.get("outcomes", [])))
        print(f"  \033[32m✓\033[0m  {jf.name:<16}  ({outcome_count} outcomes)")

    print(f"\n  Inserted {stats['inserted']} verbs  "
          f"({stats['skipped_dup']} duplicates skipped)")
    print(f"  Written to: {db_path!r}")

    if report:
        print_report(conn)

    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build SRM meaning DB from verb JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python scripts/build_meaning_db.py --src data/verbs/ --db meaning.db
              python scripts/build_meaning_db.py --src data/verbs/ --db meaning.db --report
        """),
    )
    ap.add_argument("--src",    default="data/verbs/",  help="Directory of verb JSON files")
    ap.add_argument("--db",     default="meaning.db",   help="Output SQLite path")
    ap.add_argument("--report", action="store_true",    help="Print analysis after build")
    ap.add_argument("--fresh",  action="store_true",    help="Drop existing tables/views before rebuild")
    args = ap.parse_args()

    if args.fresh and os.path.exists(args.db):
        conn = sqlite3.connect(args.db)
        conn.executescript("""
            DROP VIEW IF EXISTS verb_state_delta;
            DROP VIEW IF EXISTS verb_meaning;
            DROP TABLE IF EXISTS state_transforms;
            DROP TABLE IF EXISTS verb_applicability;
            DROP TABLE IF EXISTS verb_tools;
            DROP TABLE IF EXISTS verb_mechanisms;
            DROP TABLE IF EXISTS verb_goals;
            DROP TABLE IF EXISTS verb_polarity;
            DROP TABLE IF EXISTS verbs;
            DROP TABLE IF EXISTS polarity_classes;
        """)
        conn.commit()
        conn.close()

    build(args.src, args.db, report=args.report)


if __name__ == "__main__":
    main()
