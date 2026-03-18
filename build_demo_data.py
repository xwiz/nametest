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
    "How are you",
    "Who are you",
    "Who am I",
    "I have headache",
    "My stomach aches",
    "How does DNA replication work?",
    "write javascript code to fetch json from an api",
    "write js code to filter active users",
    "write javascript code to update an object without mutation",
]


def _cluster(label: str, summary: str, kind: str, entries: list[str]) -> dict[str, object]:
    return {
        "label": label,
        "summary": summary,
        "kind": kind,
        "entries": entries,
    }


def build_clustered_demo_kb() -> dict[str, list[dict[str, object]]]:
    return {
        "chat": [
            _cluster(
                "greetings-and-checkins",
                "Clustered sentences for greetings, short check-ins, and conversational opening turns.",
                "sentence",
                CHAT_KB[0:5],
            ),
            _cluster(
                "identity-and-self-description",
                "Clustered sentences about who the system is, how it works, and what it knows about the user.",
                "sentence",
                CHAT_KB[19:25],
            ),
            _cluster(
                "boredom-and-planning",
                "Clustered sentences about boredom, quick tasks, relaxing tasks, and planning next steps.",
                "sentence",
                CHAT_KB[13:19] + CHAT_KB[28:31],
            ),
            _cluster(
                "symptoms-and-safety",
                "Clustered sentences for headache, stomach pain, and urgent-care guidance.",
                "sentence",
                CHAT_KB[31:39],
            ),
            _cluster(
                "helpful-phrases",
                "Clustered phrases for requests, food-related intent, and describing needs clearly.",
                "phrase",
                CHAT_KB[6:13],
            ),
        ],
        "sample": [
            _cluster(
                "cell-and-genetics",
                "Clustered science sentences about cells, DNA, RNA, and gene editing.",
                "sentence",
                SAMPLE_KB[0:8],
            ),
            _cluster(
                "neuroscience-and-learning",
                "Clustered science sentences about neurons, synapses, and learning.",
                "sentence",
                SAMPLE_KB[10:12] + SAMPLE_KB[19:21],
            ),
            _cluster(
                "physics-and-cosmology",
                "Clustered science sentences about gravity, relativity, quantum effects, and light.",
                "sentence",
                SAMPLE_KB[12:19],
            ),
            _cluster(
                "earth-and-environment",
                "Clustered science sentences about tectonics, oceans, and the carbon cycle.",
                "sentence",
                SAMPLE_KB[23:26],
            ),
        ],
        "js": [
            _cluster(
                "function-shape-and-guards",
                "Clustered JavaScript phrases for small function shape, input checks, and explicit returns.",
                "phrase",
                JS_KB[0:5],
            ),
            _cluster(
                "array-transform-patterns",
                "Clustered JavaScript phrases for map, filter, reduce, and loop-based traversal.",
                "phrase",
                JS_KB[5:13],
            ),
            _cluster(
                "object-and-access-patterns",
                "Clustered JavaScript phrases for object spread, destructuring, and safe property access.",
                "phrase",
                JS_KB[13:19],
            ),
            _cluster(
                "async-fetch-and-errors",
                "Clustered JavaScript phrases for fetch, response checks, JSON parsing, and try/catch handling.",
                "phrase",
                JS_KB[19:25],
            ),
            _cluster(
                "dom-and-helper-patterns",
                "Clustered JavaScript phrases for DOM events, form extraction, and explicit helper-oriented code.",
                "phrase",
                JS_KB[25:],
            ),
        ],
    }


def main() -> None:
    out = Path("docs") / "demo-data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    clustered_kb = build_clustered_demo_kb()
    payload = {
        "version": __version__,
        "sampleKb": SAMPLE_KB,
        "chatKb": CHAT_KB,
        "jsKb": JS_KB,
        "clusteredKb": clustered_kb,
        "demoConfig": DEMO_CONFIG,
        "expansions": EXPANSIONS,
        "prompts": DEMO_PROMPTS,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
