"""
srm/cli.py — command-line interface and interactive REPL.

Fixes vs. original monolith:
  • _box() now uses paragraph-aware wrapping: the body is split on
    literal \n first, then each paragraph is wrapped independently.
    This ensures the verbose attractor badge and memory text appear on
    separate lines instead of being merged by textwrap.fill.
  • Added /delete <id> REPL command.
  • Added --stats flag for non-interactive inspection.
"""

from __future__ import annotations

import sys
import textwrap
import argparse

from .config import (
    DB_PATH, NUM_CASTS, NOISE, TOP_K, SAMPLE_KB,
    MIN_COS, VOTE_FLOOR,
    W_VOTE, W_COS,
    SIM_THRESH, MAX_WORDS,
    CHAT_KB, JS_KB,
)
from .store import MemoryStore
from .pipeline import srm_query, srm_query_cast_reconstruct, srm_query_auto
from .learning import should_auto_learn
from .nlp import set_auto_expansions
from .expansions import build_expansions_from_kb, merge_expansions
from .context import ConversationContext, augment_query_with_context


# ── Box renderer ──────────────────────────────────────────────────────────────

W = 66  # Inner content width (characters)


def _box(rows: list[tuple[str, str]], title: str = "") -> str:
    """
    Render *rows* inside a Unicode box.

    Each row is (label, body).  Special label "__sep__" draws a
    horizontal separator.  Body strings containing literal \n are
    split into paragraphs; each paragraph is word-wrapped independently.

    Fix: the original used textwrap.fill(body) directly, which merged
    embedded newlines and ran the badge + memory text together.
    """
    tl, tr, bl, br = "┌", "┐", "└", "┘"
    vl, hl         = "│", "─"
    sep_l, sep_r   = "├", "┤"

    lines: list[str] = []
    top_label = f"─ {title} " if title else ""
    lines.append(f"{tl}{top_label}{hl * (W - len(top_label))}{tr}")

    for label, body in rows:
        if label == "__sep__":
            lines.append(f"{sep_l}{hl * W}{sep_r}")
            continue
        if label:
            lines.append(f"{vl}  \033[2m{label.upper()}\033[0m")

        # ── Paragraph-aware wrapping (bug fix) ────────────────────────
        for para in body.split("\n"):
            para = para.strip()
            if not para:
                lines.append(f"{vl}")
                continue
            for line in textwrap.wrap(para, width=W - 4, subsequent_indent="  ") or [""]:
                lines.append(f"{vl}  {line}")

    lines.append(f"{bl}{hl * W}{br}")
    return "\n".join(lines)


def _colour(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m"


# ── Result printer ────────────────────────────────────────────────────────────

def print_result(result: dict, verbose: bool = False) -> None:
    """Pretty-print an srm_query result dict."""
    if "error" in result:
        print(f"  \033[31m✗ {result['error']}\033[0m")
        return

    rows: list[tuple[str, str]] = [("query", result["query"])]
    if result.get("expanded_query"):
        rows.append(("expanded", result["expanded_query"]))

    if verbose:
        parts = [f"{result.get('num_memories', '?')} memories"]
        parts.append(f"{result.get('num_casts', '?')} casts")
        if result.get("num_traversal_candidates") is not None:
            parts.append(f"traverse={result['num_traversal_candidates']}")
        if result.get("num_rerank_candidates") is not None:
            parts.append(f"rerank={result['num_rerank_candidates']}")
        if result.get("confidence") is not None:
            parts.append(f"confidence={result['confidence']:.4f}")
        if result.get("meaning_enabled"):
            parts.append("meaning=on")
        if result.get("auto_selected_mode"):
            scores = result.get("auto_scores", {})
            synth_score = float(scores.get("synth", 0.0))
            recon_score = float(scores.get("reconstruct", 0.0))
            parts.append(
                f"mode={result['auto_selected_mode']}"
                f" (synth={synth_score:.4f}"
                f"  recon={recon_score:.4f})"
            )
        rows.append(("debug", " | ".join(parts)))

    rows.append(("__sep__", ""))
    rows.append(("response", result["response"]))

    if verbose and result.get("attractor_details"):
        rows.append(("__sep__", ""))
        for rank, a in enumerate(result["attractor_details"][:5]):
            bar_w = round(a["votes"] / max(result["num_casts"], 1) * 20)
            bar   = "█" * bar_w + "░" * (20 - bar_w)
            badge = (
                f"[{a['votes']:>2}/{result['num_casts']} votes  "
                f"{bar}  cos={a['cosine']:.3f}  "
                f"hybrid={a.get('hybrid_score', 0):.3f}  d={a['hamming_dist']}]"
            )
            rows.append((
                f"attractor {rank + 1}",
                f"{badge}\n{a['text']}",
            ))

    print(_box(rows, title="SRM"))


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="srm",
        description="SRM — Stochastic Resonance Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python main.py --seed                        load sample KB
              python main.py --seed -q "DNA replication"  single query
              python main.py --seed --verbose              REPL + debug info
              python main.py --load facts.txt              custom KB
              python main.py --stats                       show store stats
        """),
    )
    ap.add_argument("--db",    default=DB_PATH, metavar="PATH",
                    help="SQLite DB path (default: srm_memory.db)")
    ap.add_argument("--seed",  action="store_true",
                    help="Load built-in 26-entry sample KB")
    ap.add_argument("--seed-chat", action="store_true",
                    help="Load built-in chat seed KB")
    ap.add_argument("--seed-js", action="store_true",
                    help="Load built-in JavaScript coding KB")
    ap.add_argument("--clear", action="store_true",
                    help="Wipe all memories before starting")
    ap.add_argument("--load",  metavar="FILE",
                    help="Load memories from a plain-text file (one per line)")
    ap.add_argument("--list",  action="store_true",
                    help="List all stored memories and exit")
    ap.add_argument("--stats", action="store_true",
                    help="Print store statistics and exit")
    ap.add_argument("-q", "--query", metavar="TEXT",
                    help="Run a single query and exit")
    ap.add_argument("--casts",  type=int,   default=NUM_CASTS, metavar="N",
                    help=f"Number of stochastic probes (default: {NUM_CASTS})")
    ap.add_argument("--noise",  type=float, default=NOISE,     metavar="F",
                    help=f"Bit-flip probability per cast (default: {NOISE})")
    ap.add_argument("--adaptive-noise", action="store_true",
                    help="Enable adaptive noise that adjusts based on KB size")
    ap.add_argument("--top-k", type=int,   default=TOP_K,     metavar="K",
                    dest="top_k",
                    help=f"Maximum attractors to retrieve (default: {TOP_K})")

    ap.add_argument("--min-cos", type=float, default=MIN_COS, metavar="F",
                    dest="min_cos",
                    help=f"Minimum TF-IDF cosine to admit an attractor (default: {MIN_COS})")
    ap.add_argument("--vote-floor", type=int, default=VOTE_FLOOR, metavar="N",
                    dest="vote_floor",
                    help=f"Minimum votes to admit an attractor (default: {VOTE_FLOOR})")
    ap.add_argument("--w-vote", type=float, default=W_VOTE, metavar="F",
                    dest="w_vote",
                    help=f"Hybrid score weight for vote share (default: {W_VOTE})")
    ap.add_argument("--w-cos", type=float, default=W_COS, metavar="F",
                    dest="w_cos",
                    help=f"Hybrid score weight for TF-IDF cosine (default: {W_COS})")
    ap.add_argument("--sim-thresh", type=float, default=SIM_THRESH, metavar="F",
                    dest="sim_thresh",
                    help=f"MMR de-dup threshold in synthesis (default: {SIM_THRESH})")
    ap.add_argument("--max-words", type=int, default=MAX_WORDS, metavar="N",
                    dest="max_words",
                    help=f"Max words in assembled response (default: {MAX_WORDS})")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show attractor details and scores")

    ap.add_argument(
        "--mode",
        choices=["synth", "reconstruct", "auto"],
        default="synth",
        help="Query mode: 'synth' (default), 'reconstruct' (fragments), or 'auto' (pick best)",
    )
    ap.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Deterministic RNG seed for stochastic traversal (debug/testing)",
    )

    ap.add_argument("--meaning-db", metavar="PATH", default=None,
                    help="Optional meaning DB for verb-polarity-aware encoding")
    ap.add_argument("--auto-learn", action="store_true",
                    help="Auto-store safe conversational user messages in REPL")
    ap.add_argument("--auto-expand", action="store_true",
                    help="Generate query expansions from KB using WordNet synonyms")

    return ap


# ── REPL banner ───────────────────────────────────────────────────────────────

BANNER = """\033[36m
  ╭────────────────────────────────────────────────────────╮
  │   SRM — Stochastic Resonance Memory  v1.1              │
  │   Pure-algorithmic · no LLM · stdlib + numpy           │
  ╰────────────────────────────────────────────────────────╯\033[0m
  Commands:
    /add <text>      store a new memory
    /delete <id>     remove memory by DB id
    /list            list all memories with their ids
    /stats           show configuration & counts
    /clear           wipe all memories
    /quit            exit
"""


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    args  = build_parser().parse_args()
    store = MemoryStore(args.db)

    meaning_db = None
    if args.meaning_db:
        try:
            from .meaning import MeaningDB
            meaning_db = MeaningDB(args.meaning_db)
        except Exception as e:
            print(f"  \033[31mCould not open meaning DB: {e}\033[0m")
            sys.exit(1)

    # Build auto-expansions from KB if requested
    if args.auto_expand:
        try:
            _, texts = store.load_all()
            if texts:
                from .config import EXPANSIONS
                auto_exp = build_expansions_from_kb(texts, min_document_freq=2, max_terms=100)
                merged = merge_expansions(EXPANSIONS, auto_exp)
                set_auto_expansions(merged)
                print(f"  Generated {len(auto_exp)} auto-expansions from {len(texts)} memories")
            else:
                print("  No memories in store to generate expansions from")
        except Exception as e:
            print(f"  \033[33mCould not generate auto-expansions: {e}\033[0m")

    # ── Pre-query mutations ───────────────────────────────────────────

    if args.clear:
        store.clear()
        print("  Memory store cleared.")

    if args.seed:
        n = sum(1 for t in SAMPLE_KB if store.add(t))
        print(f"  Seeded {n} new memories  (total: {store.count()})")

    if args.seed_chat:
        n = sum(1 for t in CHAT_KB if store.add(t))
        print(f"  Seeded {n} new chat memories  (total: {store.count()})")

    if args.seed_js:
        n = sum(1 for t in JS_KB if store.add(t))
        print(f"  Seeded {n} new JavaScript memories  (total: {store.count()})")

    if args.load:
        try:
            BATCH_SIZE = 500
            added = 0
            batch = []
            with open(args.load, encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        batch.append(line)
                        if len(batch) >= BATCH_SIZE:
                            for text in batch:
                                if store.add(text):
                                    added += 1
                            batch = []
                # Process remaining lines
                for text in batch:
                    if store.add(text):
                        added += 1
            print(f"  Loaded {added} memories from {args.load!r}  "
                  f"(total: {store.count()})")
        except OSError as e:
            print(f"  \033[31mCould not read {args.load!r}: {e}\033[0m")
            sys.exit(1)

    # ── Non-interactive modes ─────────────────────────────────────────

    if args.stats:
        ids, texts = store.load_all()
        print(f"  memories : {len(texts)}")
        print(f"  db path  : {args.db}")
        print(f"  casts    : {args.casts}")
        print(f"  noise    : {args.noise}")
        print(f"  top-k    : {args.top_k}")
        print(f"  mode     : {args.mode}")
        print(f"  rng-seed : {args.rng_seed}")
        print(f"  meaning  : {bool(meaning_db)}")
        print(f"  auto-learn : {args.auto_learn}")
        store.close()
        if meaning_db:
            meaning_db.close()
        return

    if args.list:
        ids, texts = store.load_all()
        if not texts:
            print("  (empty store)")
        for mid, t in zip(ids, texts):
            print(f"  #{mid:<4}  {t}")
        store.close()
        if meaning_db:
            meaning_db.close()
        return

    def _run(q: str) -> None:
        if args.mode == "auto":
            result = srm_query_auto(
                q,
                store,
                top_k=args.top_k,
                num_casts=args.casts,
                noise=args.noise,
                min_cos=args.min_cos,
                vote_floor=args.vote_floor,
                w_vote=args.w_vote,
                w_cos=args.w_cos,
                sim_thresh=args.sim_thresh,
                max_words=args.max_words,
                meaning_db=meaning_db,
                seed=args.rng_seed,
                use_adaptive_noise=args.adaptive_noise,
            )
        elif args.mode == "reconstruct":
            result = srm_query_cast_reconstruct(
                q,
                store,
                num_casts=args.casts,
                noise=args.noise,
                sim_thresh=args.sim_thresh,
                max_words=args.max_words,
                meaning_db=meaning_db,
                seed=args.rng_seed,
                use_adaptive_noise=args.adaptive_noise,
            )
        else:
            result = srm_query(
                q,
                store,
                top_k=args.top_k,
                num_casts=args.casts,
                noise=args.noise,
                min_cos=args.min_cos,
                vote_floor=args.vote_floor,
                w_vote=args.w_vote,
                w_cos=args.w_cos,
                sim_thresh=args.sim_thresh,
                max_words=args.max_words,
                meaning_db=meaning_db,
                seed=args.rng_seed,
                use_adaptive_noise=args.adaptive_noise,
            )
        print_result(result, verbose=args.verbose)

    if args.query:
        if store.count() == 0:
            print("  Empty store — use --seed or --load first.")
        else:
            _run(args.query)
        store.close()
        if meaning_db:
            meaning_db.close()
        return

    # ── Interactive REPL ──────────────────────────────────────────────

    print(BANNER)
    n = store.count()
    if n == 0:
        print("  \033[33m⚠  Memory store is empty — use /add <text> "
              "or restart with --seed\033[0m\n")
    else:
        print(f"  {n} memories loaded · {args.casts} casts · "
              f"noise={args.noise}\n")

    # Initialize context buffer for multi-turn conversations
    context = ConversationContext(max_turns=3)

    while True:
        try:
            raw = input(_colour("srm> ", 36)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        cmd = raw.lower()

        if cmd in ("quit", "exit", "/quit", "/exit"):
            break

        elif raw.startswith("/add "):
            text = raw[5:].strip()
            if not text:
                print("  Usage: /add <text>\n")
            else:
                ok     = store.add(text)
                status = ("\033[32mencoded + stored\033[0m" if ok
                          else "\033[33mduplicate — skipped\033[0m")
                print(f"  {status}  [{store.count()} total]\n")

        elif raw.startswith("/delete "):
            arg = raw[8:].strip()
            if not arg.isdigit():
                print("  Usage: /delete <id>  (use /list to see ids)\n")
            else:
                ok = store.delete(int(arg))
                if ok:
                    print(f"  \033[32mDeleted #{arg}\033[0m  "
                          f"[{store.count()} remaining]\n")
                else:
                    print(f"  \033[31mNo memory with id #{arg}\033[0m\n")

        elif cmd == "/list":
            ids, texts = store.load_all()
            if not texts:
                print("  (empty store)\n")
            else:
                for mid, t in zip(ids, texts):
                    print(f"  #{mid:<4}  {t}")
                print()

        elif cmd == "/stats":
            ids, texts = store.load_all()
            print(f"  memories : {len(texts)}")
            print(f"  db path  : {args.db}")
            print(f"  casts    : {args.casts}")
            print(f"  noise    : {args.noise}")
            print(f"  top-k    : {args.top_k}")
            print(f"  meaning  : {bool(meaning_db)}\n")

        elif raw.startswith("/meaning "):
            text = raw[len("/meaning "):].strip()
            if not text:
                print("  Usage: /meaning <text>\n")
            else:
                try:
                    from .meaning import extract_meaning
                    m = extract_meaning(text, db=meaning_db)
                    print("  alerts:")
                    if m.get("alerts"):
                        for a in m["alerts"]:
                            print(f"    - {a}")
                    else:
                        print("    (none)")
                    print("  frames:")
                    if m.get("frames"):
                        for f in m["frames"]:
                            subj = " ".join(f.get("subject") or []) or "—"
                            obj  = " ".join(f.get("object") or []) or "—"
                            v    = f.get("verb") or "—"
                            vc   = f.get("verb_canonical") or "—"
                            pc   = f.get("polarity_class") or "—"
                            print(f"    - subj: {subj} | verb: {v} ({vc}) | class: {pc} | obj: {obj}")
                    else:
                        print("    (none)")
                    print()
                except Exception as e:
                    print(f"  \033[31mCould not extract meaning: {e}\033[0m\n")

        elif raw.startswith("/verb "):
            v = raw[len("/verb "):].strip()
            if not v:
                print("  Usage: /verb <verb>\n")
            elif not meaning_db:
                print("  (no meaning DB loaded — pass --meaning-db meaning.db)\n")
            else:
                from .meaning import describe_verb
                print(f"  {describe_verb(v, meaning_db)}\n")

        elif cmd == "/clear":
            store.clear()
            print("  \033[31mMemory store cleared.\033[0m\n")

        else:
            if store.count() == 0:
                print("  \033[33mNo memories stored. "
                      "Use /add <text> first.\033[0m\n")
                continue
            learned_msg = None
            if args.auto_learn:
                decision = should_auto_learn(raw)
                if decision.accepted and decision.normalized_text:
                    ok = store.add(decision.normalized_text)
                    status = "learned" if ok else "known"
                    learned_msg = f"  \033[36m[auto-learn: {status} ({decision.reason})]\033[0m"
                elif args.verbose:
                    learned_msg = f"  \033[2m[auto-learn skip: {decision.explanation}]\033[0m"
            
            # Augment query with context if available
            query_to_run = augment_query_with_context(raw, context, max_context_turns=2)
            
            result = None
            if args.mode == "auto":
                result = srm_query_auto(
                    query_to_run,
                    store,
                    top_k=args.top_k,
                    num_casts=args.casts,
                    noise=args.noise,
                    min_cos=args.min_cos,
                    vote_floor=args.vote_floor,
                    w_vote=args.w_vote,
                    w_cos=args.w_cos,
                    sim_thresh=args.sim_thresh,
                    max_words=args.max_words,
                    meaning_db=meaning_db,
                    seed=args.rng_seed,
                    use_adaptive_noise=args.adaptive_noise,
                )
            elif args.mode == "reconstruct":
                result = srm_query_cast_reconstruct(
                    query_to_run,
                    store,
                    num_casts=args.casts,
                    noise=args.noise,
                    sim_thresh=args.sim_thresh,
                    max_words=args.max_words,
                    meaning_db=meaning_db,
                    seed=args.rng_seed,
                    use_adaptive_noise=args.adaptive_noise,
                )
            else:
                result = srm_query(
                    query_to_run,
                    store,
                    top_k=args.top_k,
                    num_casts=args.casts,
                    noise=args.noise,
                    min_cos=args.min_cos,
                    vote_floor=args.vote_floor,
                    w_vote=args.w_vote,
                    w_cos=args.w_cos,
                    sim_thresh=args.sim_thresh,
                    max_words=args.max_words,
                    meaning_db=meaning_db,
                    seed=args.rng_seed,
                    use_adaptive_noise=args.adaptive_noise,
                )
            
            print_result(result, verbose=args.verbose)
            
            # Add turn to context if successful response
            if result and "response" in result and result["response"]:
                context.add_turn(raw, result["response"])
            
            if learned_msg:
                print(learned_msg)
            print()

    store.close()
    if meaning_db:
        meaning_db.close()
