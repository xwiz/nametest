from __future__ import annotations

import argparse

from srm.crawler import collect_new_lines
from srm.store import MemoryStore


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch conversational text and add filtered lines to an SRM DB")
    ap.add_argument("--db", default="srm_memory.db", help="SQLite DB path")
    ap.add_argument("--url", action="append", dest="urls", default=[], help="Source URL to crawl")
    args = ap.parse_args()

    if not args.urls:
        raise SystemExit("Provide at least one --url")

    store = MemoryStore(args.db)
    try:
        _, texts = store.load_all()
        known = {t.lower().strip() for t in texts}
        candidates = collect_new_lines(args.urls, known_lines=known)
        added = 0
        for item in candidates:
            if store.add(item.text):
                added += 1
        print(f"Fetched {len(candidates)} candidate lines; added {added} new memories.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
