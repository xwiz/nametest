from __future__ import annotations

import json
from pathlib import Path

from srm import __version__
from srm.config import CHAT_KB, EXPANSIONS, SAMPLE_KB


DEMO_PROMPTS = [
    "hello",
    "I am bored",
    "What is the weather today",
    "How do you feel",
    "Who are you",
    "Who am I",
    "I have headache",
    "My stomach aches",
    "How does DNA replication work?",
]


def main() -> None:
    out = Path("docs") / "demo-data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": __version__,
        "sampleKb": SAMPLE_KB,
        "chatKb": CHAT_KB,
        "expansions": EXPANSIONS,
        "prompts": DEMO_PROMPTS,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
