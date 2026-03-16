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

# ── Polarity class catalogue ──────────────────────────────────────────────────
#
# 12 semantic direction classes.  Each gets a deterministic 128-bit bitmask
# generated from a fixed seed so the masks are stable across rebuilds.
# Adjacent classes in the list are kept semantically related so the Hamming
# geometry is meaningful:
#   destruct ↔ preserve should be far apart (~64 bits)
#   intensify ↔ diminish should be far apart
#   aggregate ↔ disrupt should be far apart
#
# We generate masks via seeded numpy random; expected hamming between any
# two random 128-bit codes is 64.  We post-check all pairs exceed 48 bits.

POLARITY_CLASSES: list[dict] = [
    # class_id (0-indexed), name, representative goals/states
    {"name": "destruct",   "keywords": {"harm","worsen","damage","destroy","injustice",
                                         "degrade","corrupt","weaken_obj","break","ruin",
                                         "oppressed","deprived","worsened","negative"}},
    {"name": "preserve",   "keywords": {"protect","maintain","sustain","keep","conservation",
                                         "stable","unchanged","retention","guard","saved",
                                         "preserved","intact"}},
    {"name": "transfer",   "keywords": {"transfer","movement","transport","convey","relay",
                                         "send","pass","dispatch","deliver","shift","moved"}},
    {"name": "enable",     "keywords": {"create","produce","generate","build","formation",
                                         "empower","allow","facilitate","construct","initiate",
                                         "created","formed","generated"}},
    {"name": "block",      "keywords": {"prevent","stop","negate","prohibit","obstruct",
                                         "suppress","inhibit","halt","restrict","deny",
                                         "blocked","prevented","stopped"}},
    {"name": "intensify",  "keywords": {"intensify","amplify","increase","strengthen",
                                         "stimulation","escalate","heighten","magnify",
                                         "escalated","triggered","intensified","amplified"}},
    {"name": "diminish",   "keywords": {"reduce","weaken","decrease","lessen","shrink",
                                         "minimise","dampen","attenuate","lowered","reduced",
                                         "weakened","decreased"}},
    {"name": "aggregate",  "keywords": {"collection","consolidation","clustering","grouping",
                                         "merging","combine","coalesce","accumulate",
                                         "clustered","combined","grouped","compacted",
                                         "unified"}},
    {"name": "align",      "keywords": {"harmony","alignment","consent","agreement",
                                         "coordination","synchronise","reconcile","unify",
                                         "accepted","approved","affirmed","aligned",
                                         "shared belief"}},
    {"name": "disrupt",    "keywords": {"disruption","disturb","agitate","unsettle","break",
                                         "destabilise","perturb","interrupt","interfere",
                                         "disturbed","unsettled","in motion","shifted"}},
    {"name": "transform",  "keywords": {"maturation","evolution","change","conversion",
                                         "alter","mutate","develop","metamorphose",
                                         "changed","matured","evolved","aged","weathered"}},
    {"name": "process",    "keywords": {"resolution","coping","understanding","analysis",
                                         "resolve","work through","deliberate","reflect",
                                         "processed","settled","decided","experienced"}},
]

NUM_CLASSES = len(POLARITY_CLASSES)


def _generate_bitmasks(seed: int = 42) -> list[bytes]:
    """
    Generate NUM_CLASSES orthogonal-ish 128-bit bitmasks.

    Strategy: draw random uint8 arrays with a fixed seed until every
    pair of masks has hamming distance ≥ 48.  With 128 bits and 12 classes
    this typically succeeds on the first draw.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(1000):
        masks = rng.integers(0, 256, size=(NUM_CLASSES, 16), dtype=np.uint8)
        pop = np.array([bin(b).count("1") for b in range(256)], dtype=np.int32)
        ok = True
        for i in range(NUM_CLASSES):
            for j in range(i + 1, NUM_CLASSES):
                d = int(pop[masks[i] ^ masks[j]].sum())
                if d < 48:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return [m.tobytes() for m in masks]
    raise RuntimeError("Could not generate orthogonal bitmasks after 1000 attempts")


# ── Keyword → polarity class scorer ──────────────────────────────────────────

def _score_polarity(verb_data: dict) -> str:
    """
    Derive the best-fit polarity class for a verb entry.

    Scores each class by counting keyword matches in:
      goals (weight 3), mechanisms (weight 2),
      final_object_states all dimensions (weight 2),
      final_subject_states all dimensions (weight 1)

    Falls back to "transform" if no match exceeds 0.
    """
    scores: Counter = Counter()

    def _hit(text: str, weight: int) -> None:
        low = text.lower().replace("-", " ").replace("_", " ")
        for ci, cls in enumerate(POLARITY_CLASSES):
            for kw in cls["keywords"]:
                if kw in low:
                    scores[ci] += weight

    for g in verb_data.get("goals", []):
        _hit(g, 3)
    for m in verb_data.get("mechanisms", []):
        _hit(m, 2)

    for role_key in ("final_object_states", "final_subject_states"):
        weight = 2 if role_key == "final_object_states" else 1
        for dim_vals in verb_data.get(role_key, {}).values():
            for v in dim_vals:
                _hit(v, weight)

    if not scores:
        return "transform"
    best = scores.most_common(1)[0][0]
    return POLARITY_CLASSES[best]["name"]


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS polarity_classes (
    id      INTEGER PRIMARY KEY,
    name    TEXT UNIQUE NOT NULL,
    bitmask BLOB NOT NULL          -- 16 bytes = 128 bits
);

CREATE TABLE IF NOT EXISTS verbs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    verb    TEXT UNIQUE NOT NULL,
    prefix  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verb_polarity (
    verb_id  INTEGER NOT NULL REFERENCES verbs(id),
    class_id INTEGER NOT NULL REFERENCES polarity_classes(id),
    score    REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (verb_id, class_id)
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

CREATE INDEX IF NOT EXISTS idx_verb_polarity_verb   ON verb_polarity(verb_id);
CREATE INDEX IF NOT EXISTS idx_state_transforms_verb ON state_transforms(verb_id);
CREATE INDEX IF NOT EXISTS idx_verb_goals_verb       ON verb_goals(verb_id);

-- Flat lookup view: verb text → polarity class name + bitmask
CREATE VIEW IF NOT EXISTS verb_meaning AS
SELECT  v.verb,
        pc.name    AS polarity_class,
        pc.bitmask AS polarity_mask,
        vp.score
FROM    verbs v
JOIN    verb_polarity  vp ON vp.verb_id  = v.id
JOIN    polarity_classes pc ON pc.id     = vp.class_id;

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
    with open(path) as fh:
        return json.load(fh)


def ingest_file(conn: sqlite3.Connection,
                data: dict,
                class_name_to_id: dict[str, int],
                stats: Counter) -> None:
    """Insert all verb data from one parsed JSON file."""
    prefix = data.get("prefix", "??")

    for outcome in data.get("outcomes", []):
        verb = outcome.get("verb", "").strip().lower()
        if not verb:
            continue

        # ── verbs table ──────────────────────────────────────────────
        try:
            cur = conn.execute(
                "INSERT INTO verbs (verb, prefix) VALUES (?,?)", (verb, prefix)
            )
            verb_id = cur.lastrowid
            stats["inserted"] += 1
        except sqlite3.IntegrityError:
            verb_id = conn.execute(
                "SELECT id FROM verbs WHERE verb=?", (verb,)
            ).fetchone()[0]
            stats["skipped_dup"] += 1
            continue

        # ── polarity ─────────────────────────────────────────────────
        polarity_name = _score_polarity(outcome)
        class_id = class_name_to_id[polarity_name]
        conn.execute(
            "INSERT INTO verb_polarity (verb_id, class_id, score) VALUES (?,?,?)",
            (verb_id, class_id, 1.0),
        )

        # ── goals / mechanisms / tools ────────────────────────────────
        for g in outcome.get("goals", []):
            conn.execute(
                "INSERT INTO verb_goals (verb_id, goal) VALUES (?,?)", (verb_id, g)
            )
        for m in outcome.get("mechanisms", []):
            conn.execute(
                "INSERT INTO verb_mechanisms (verb_id, mechanism) VALUES (?,?)",
                (verb_id, m),
            )
        for t in outcome.get("tools", []):
            if t:
                conn.execute(
                    "INSERT INTO verb_tools (verb_id, tool) VALUES (?,?)", (verb_id, t)
                )

        # ── applicability ─────────────────────────────────────────────
        for etype in outcome.get("applicable_subjects", []):
            conn.execute(
                "INSERT INTO verb_applicability (verb_id, role, entity_type) "
                "VALUES (?,?,?)",
                (verb_id, "subject", etype),
            )
        for etype in outcome.get("applicable_objects", []):
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
            for dim, values in outcome.get(subj_key, {}).items():
                for sv in values:
                    if sv and sv.lower() != "non-applicable":
                        conn.execute(
                            "INSERT INTO state_transforms "
                            "(verb_id,role,phase,dimension,state_value) "
                            "VALUES (?,?,?,?,?)",
                            (verb_id, "subject", phase, dim, sv),
                        )
            for dim, values in outcome.get(obj_key, {}).items():
                for sv in values:
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

    print("\n  Polarity class distribution:")
    rows = conn.execute("""
        SELECT pc.name, COUNT(vp.verb_id) as n
        FROM   polarity_classes pc
        LEFT JOIN verb_polarity vp ON vp.class_id = pc.id
        GROUP  BY pc.id
        ORDER  BY n DESC
    """).fetchall()
    for name, n in rows:
        bar = "█" * n + "░" * max(0, 12 - n)
        print(f"    {name:<12}  {bar}  {n}")

    print("\n  Sample verb meanings:")
    sample = conn.execute("""
        SELECT v.verb, pc.name
        FROM   verbs v
        JOIN   verb_polarity vp ON vp.verb_id = v.id
        JOIN   polarity_classes pc ON pc.id = vp.class_id
        ORDER  BY v.verb
        LIMIT  20
    """).fetchall()
    for verb, cls in sample:
        print(f"    {verb:<18}  →  {cls}")

    print("\n  Bitmask Hamming distance matrix (min should be ≥ 48):")
    masks = conn.execute(
        "SELECT name, bitmask FROM polarity_classes ORDER BY id"
    ).fetchall()
    pop = [bin(b).count("1") for b in range(256)]
    names = [r[0][:6] for r in masks]
    arrs  = [np.frombuffer(r[1], dtype=np.uint8) for r in masks]
    header = "         " + "".join(f"{n:>7}" for n in names)
    print(f"    {header}")
    for i, (ni, ai) in enumerate(zip(names, arrs)):
        row = f"    {ni:<8}"
        for j, aj in enumerate(arrs):
            d = int(sum(pop[a ^ b] for a, b in zip(ai, aj)))
            row += f"  {d:>5}"
        print(row)
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

    # ── seed polarity classes ──────────────────────────────────────────
    bitmasks = _generate_bitmasks()
    class_name_to_id: dict[str, int] = {}
    for i, (cls, mask) in enumerate(zip(POLARITY_CLASSES, bitmasks)):
        try:
            conn.execute(
                "INSERT INTO polarity_classes (id, name, bitmask) VALUES (?,?,?)",
                (i, cls["name"], mask),
            )
        except sqlite3.IntegrityError:
            pass  # already seeded on a previous run
        class_name_to_id[cls["name"]] = i
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

        ingest_file(conn, data, class_name_to_id, stats)
        conn.commit()
        print(f"  \033[32m✓\033[0m  {jf.name:<16}  "
              f"({len(data.get('outcomes', []))} outcomes)")

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
    args = ap.parse_args()

    build(args.src, args.db, report=args.report)


if __name__ == "__main__":
    main()
