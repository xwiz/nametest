# SRM — Stochastic Resonance Memory

A tiny < 100KB memory-and-response system built with **Python + SQLite + NumPy**.

The goal of this MVP is simple:

- **show that a specialized LLM-style system doesnt require a billion dataset or even explicit training**
- **handle only narrow tasks by searching explicit memories**
- **use stochastic search plus meaning inference to separate relevant memories**
- **reconstruct responses from fragments instead of free-form token generation**

SRM is not trying to compete with a frontier transformer.
It is trying to demonstrate that for specific tasks, you can get useful behaviour from:

- **explicit knowledge rows instead of giant hidden weights**
- **stochastic search in Hamming space instead of full neural generation**
- **meaning inference from verbs, nouns, and polarity masks**
- **response reconstruction from retrieved fragments**

That makes the system:

- **inspectable**
- **editable**
- **cheap to run**
- **easy to debug when behaviour changes**

## MVP shape

- **Memory**
  - Knowledge is stored as explicit text memories in SQLite.
  - You can inspect, add, delete, or replace memories directly.

- **Search**
  - Queries are encoded into 128-bit SimHash-style codes.
  - Retrieval uses noisy stochastic traversal plus TF-IDF re-ranking.

- **Meaning inference**
  - Optional `meaning.db` applies verb-driven polarity masking.
  - This helps separate actions with different meanings that share similar nouns.

- **Response generation**
  - `synth` assembles sentence-style responses from retrieved memories.
  - `reconstruct` works better for fragment-style KBs.
  - `auto` tries both and keeps the stronger output.

## Browser demo

- **[Open browser demo](https://htmlpreview.github.io/?https://raw.githubusercontent.com/xwiz/nametest/main/docs/index.html)**

The browser demo uses the current built-in chat, JavaScript, and sample KBs plus exported query expansions.

## What the current system does

- **Retrieval**
  - Encodes text into 128-bit codes
  - Uses noisy multi-cast traversal to find stable attractors
  - Re-ranks with TF-IDF cosine

- **Response modes**
  - `synth`: sentence assembly from top memories
  - `reconstruct`: fragment-oriented cast reconstruction
  - `auto`: runs both and keeps the better result

- **Meaning-aware mode**
  - Optional `meaning.db`
  - Applies verb-driven polarity masks to object tokens
  - Keeps compatibility with legacy meaning DB layouts while preferring the new direct-mask schema

- **Safe continuous learning**
  - Optional REPL auto-learning with conservative rules
  - Rejects secrets, PII, URLs, code-like text, emergencies, and broad free-form questions

- **Opt-in crawler ingestion**
  - Pulls filtered conversational lines from URLs
  - De-duplicates against existing memory
  - Intended for curated idle-learning workflows, not blind scraping

## Install

```powershell
python -m pip install -r requirements.txt
```

## Fastest demo runs

### 1) General science sample KB

```powershell
python main.py --seed --mode auto
```

Try prompts like:

```text
How does DNA replication work?
What is a black hole?
How do cells produce energy?
```

### 2) Conversational chat KB

```powershell
python main.py --seed-chat --mode auto --auto-learn
```

Try prompts like:

```text
I am bored
What is the weather today
How do you feel
Who are you
Who am I
I have headache
My stomach aches
```

### 3) Meaning-aware run

First build the meaning DB:

```powershell
python build_meaning_db.py --src C:\dev\nameless_vector\verb_state --db meaning.db --fresh --report
```

Then run with it:

```powershell
python main.py --seed-chat --mode auto --meaning-db meaning.db
```

## Useful CLI commands

- **Seed sample KB**
  - `python main.py --seed`

- **Seed chat KB**
  - `python main.py --seed-chat`

- **Single query**
  - `python main.py --seed-chat --mode auto -q "Who are you"`

- **Verbose query debugging**
  - `python main.py --seed --mode auto --verbose -q "How does DNA replication work?"`

- **Inspect store**
  - `python main.py --db srm_memory.db --stats`
  - `python main.py --db srm_memory.db --list`

- **Load a custom KB file**
  - `python main.py --db my_kb.db --load data\health_kb.txt --mode auto`

## REPL commands

- **`/add <text>`**
  - Store a memory manually

- **`/delete <id>`**
  - Delete one memory by DB id

- **`/list`**
  - Show stored memories

- **`/stats`**
  - Show current DB/config

- **`/clear`**
  - Wipe the active memory DB

- **`/meaning <text>`**
  - Show extracted meaning / frames

- **`/verb <verb>`**
  - Show a verb summary from `meaning.db`

## Best way to train / improve the system

### 1) Keep data sources separated

Use different stores for different roles:

- **Base chat memory**
  - Start from `--seed-chat`
  - Add generic conversational scaffolding

- **Domain memory**
  - Use separate DBs or separate text files for health, science, etc.
  - Load them intentionally with `--load`

- **User-learned memory**
  - Let `--auto-learn` capture only safe conversational statements
  - Keep this conservative

This avoids poisoning one global DB with mixed-quality data.

### 2) Prefer high-quality plain text lines

The best training input format today is still:

- one sentence or short fragment per line
- factual or clearly useful conversational scaffolding
- low redundancy
- no secrets, IDs, addresses, tokens, passwords, or URLs unless the dataset explicitly requires them

### 3) Use `--mode auto` by default

`auto` is the safest default because:

- sentence KBs usually work better with synthesis
- fragment KBs usually work better with reconstruction
- the code already scores both and keeps the better output

### 4) Improve retrieval by editing `EXPANSIONS`

If the system understands a topic but misses natural wording, update `srm/config.py`:

- add surface query terms people actually type
- map them to terms that exist in your KB

This is one of the highest-leverage improvements in the current system.

### 5) Improve chat quality by expanding `CHAT_KB`

Add more generic conversation patterns, for example:

- clarification phrasing
- identity / capability explanations
- common symptom guidance
- planning / next-step scaffolding

Avoid adding exact canned responses for one prompt only.
Favor reusable lines that let synthesis construct answers dynamically.

### 6) Use safe auto-learning, not unrestricted chat dumping

Recommended:

```powershell
python main.py --seed-chat --mode auto --auto-learn
```

Current auto-learning is intentionally conservative.
That is a feature, not a limitation.

### 7) Use the crawler only as an opt-in data source

Example:

```powershell
python crawl_chat_kb.py --db srm_memory.db --url https://en.wikipedia.org/wiki/Conversation
```

Use curated pages and review the resulting DB quality.
Do not treat crawler output as automatically trustworthy.

## Health KB workflow

### Download public datasets

```powershell
python download_kbs.py --data-dir data --build-kb-out data\health_kb.txt
```

### Build a health KB manually

```powershell
python build_health_kb.py --medquad-dir data\MedQuAD-master --fdc data\FoodData_Central_foundation_food_csv --out data\health_kb.txt
```

### Query with the health KB

```powershell
python main.py --db health.db --load data\health_kb.txt --mode auto
```

## Regenerating derived files

### Rebuild `meaning.db`

Run this after changing the external verb JSON source or the meaning DB builder.
The verb JSON files are not stored in this repo, so pass the source directory explicitly:

```powershell
python build_meaning_db.py --src C:\dev\nameless_vector\verb_state --db meaning.db --fresh --report
```

### Rebuild browser demo data

Run this after changing `SAMPLE_KB`, `CHAT_KB`, or `EXPANSIONS`:

```powershell
python build_demo_data.py
```

## Run tests

```powershell
python -m unittest -q
```

## Local browser demo

If you want to open the demo locally instead of using the GitHub link:

```powershell
python -m http.server 8000 --directory docs
```

Then open:

- `http://localhost:8000`

## Key files

- **`main.py`**
  - main CLI entry point

- **`srm/cli.py`**
  - REPL, CLI flags, query dispatch

- **`srm/pipeline.py`**
  - query pipeline, auto-mode selection, reconstruction path

- **`srm/meaning.py`**
  - meaning-aware encoding, verb polarity masks, meaning extraction

- **`build_meaning_db.py`**
  - builds `meaning.db` from verb JSON files

- **`build_health_kb.py`**
  - builds `health_kb.txt` from public health/nutrition datasets

- **`crawl_chat_kb.py`**
  - opt-in conversational crawler ingestion

- **`docs/index.html`**
  - browser demo
