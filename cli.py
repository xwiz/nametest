"""
srm/cli.py — command-line interface and interactive REPL.

Fixes vs. original monolith:
  • _box() now uses paragraph-aware wrapping: the body is split on
    literal \\n first, then each paragraph is wrapped independently.
    This ensures the verbose attractor badge and memory text appear on
    separate lines instead of being merged by textwrap.fill.
  • Added /delete <id> REPL command.
  • Added --stats flag for non-interactive inspection.
"""

from __future__ import annotations

import sys
import textwrap
import argparse

from .config import DB_PATH, NUM_CASTS, NOISE, TOP_K, SAMPLE_KB
from .store import MemoryStore
from .pipeline import srm_query


# ── Box renderer ──────────────────────────────────────────────────────────────

W = 66  # Inner content width (characters)


def _box(rows: list[tuple[str, str]], title: str = "") -> str:
    """
    Render *rows* inside a Unicode box.

    Each row is (label, body).  Special label "__sep__" draws a
    horizontal separator.  Body strings containing literal \\n are
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
    rows.append(("__sep__", ""))
    rows.append(("response", result["response"]))

    if verbose and result.get("attractor_details"):
        rows.append(("__sep__", ""))
        for rank, a in enumerate(result["attractor_details"][:3]):
            bar_w = round(a["votes"] / result["num_casts"] * 20)
            bar   = "█" * bar_w + "░" * (20 - bar_w)
            badge = (
                f"[{a['votes']:>2}/{result['num_casts']} votes  "
                f"{bar}  cos={a['cosine']:.3f}  d={a['hamming_dist']}]"
            )
            # Newline separates badge from memory text — preserved by _box
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
                    help="Load built-in 25-entry sample KB")
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
    ap.add_argument("--top-k", type=int,   default=TOP_K,     metavar="K",
                    dest="top_k",
                    help=f"Maximum attractors to retrieve (default: {TOP_K})")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show attractor details and scores")
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

    # ── Pre-query mutations ───────────────────────────────────────────

    if args.clear:
        store.clear()
        print("  Memory store cleared.")

    if args.seed:
        n = sum(1 for t in SAMPLE_KB if store.add(t))
        print(f"  Seeded {n} new memories  (total: {store.count()})")

    if args.load:
        try:
            with open(args.load) as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            n = sum(1 for ln in lines if store.add(ln))
            print(f"  Loaded {n} memories from {args.load!r}  "
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
        store.close()
        return

    if args.list:
        ids, texts = store.load_all()
        if not texts:
            print("  (empty store)")
        for mid, t in zip(ids, texts):
            print(f"  #{mid:<4}  {t}")
        store.close()
        return

    def _run(q: str) -> None:
        result = srm_query(
            q, store,
            top_k=args.top_k,
            num_casts=args.casts,
            noise=args.noise,
        )
        print_result(result, verbose=args.verbose)

    if args.query:
        if store.count() == 0:
            print("  Empty store — use --seed or --load first.")
        else:
            _run(args.query)
        store.close()
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
            print(f"  top-k    : {args.top_k}\n")

        elif cmd == "/clear":
            store.clear()
            print("  \033[31mMemory store cleared.\033[0m\n")

        else:
            if store.count() == 0:
                print("  \033[33mNo memories stored. "
                      "Use /add <text> first.\033[0m\n")
                continue
            _run(raw)
            print()

    store.close()
