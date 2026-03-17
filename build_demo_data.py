from __future__ import annotations

import json
from pathlib import Path

from srm import __version__
from srm.config import CHAT_KB, DEMO_CONFIG, EXPANSIONS, JS_KB, SAMPLE_KB


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
    "write javascript code to fetch json from an api",
    "write js code to filter active users",
    "write javascript code to update an object without mutation",
]


def main() -> None:
    out = Path("docs") / "demo-data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": __version__,
        "sampleKb": SAMPLE_KB,
        "chatKb": CHAT_KB,
        "jsKb": JS_KB,
        "demoConfig": DEMO_CONFIG,
        "expansions": EXPANSIONS,
        "prompts": DEMO_PROMPTS,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
