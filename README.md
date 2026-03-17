# SRM — Stochastic Resonance Memory

A small algorithmic memory-and-response system built with **Python + SQLite + NumPy**.

It stores lines of text as binary SimHash-style codes, retrieves memories through stochastic Hamming-space search, and assembles responses without using an LLM.

## Why this approach is novel

Most modern chat systems are built around **generative pretrained transformers**: large neural networks that compress vast training corpora into model weights, then generate the next token autoregressively.

This project explores a very different path:

- **Explicit memory instead of hidden weights**
  - Knowledge stays as inspectable text rows in SQLite.
  - You can list, delete, seed, and audit exactly what the system knows.

- **Algorithmic retrieval instead of forward-passing a giant model**
  - Queries are encoded into 128-bit SimHash-style codes.
  - Retrieval happens through noisy stochastic traversal in Hamming space, then TF-IDF re-ranking.

- **Meaning-aware masking instead of dense latent semantics**
  - Optional `meaning.db` injects verb-driven polarity masks derived from final object states.
  - That lets the system separate semantically different actions without training a neural embedding model.

- **Constructive response assembly instead of free-form token generation**
  - Responses are synthesised from retrieved memories or reconstructed from cast outputs.
  - The system is valuable precisely because you can inspect which source memories were used.

- **Small-data, controllable adaptation instead of large-scale pretraining**
  - You improve behaviour by editing KB lines, expansions, polarity knowledge, or safe learning rules.
  - That makes failures easier to diagnose than in opaque neural model weights.

## How SRM differs from generative pretrained transformers

| Dimension | SRM | Generative pretrained transformers |
| --- | --- | --- |
| Core mechanism | Retrieval + stochastic traversal + synthesis | Neural next-token generation |
| Knowledge storage | Explicit text memories in SQLite | Implicitly compressed into model parameters |
| Inspectability | High: list exact memories and attractors | Low: weights are not human-auditable knowledge units |
| Update path | Add/edit/delete memories or rules | Fine-tune, retrain, or rely on prompting/RAG layers |
| Failure mode | Misses, weak retrieval, over-literal stitching | Hallucination, drifting generations, opaque errors |
| Resource profile | Small CPU-friendly Python process | Usually much larger RAM/VRAM + model runtime |
| Determinism | Reproducible with fixed seed | Often only partially reproducible |
| Best use case | Transparent local memory/retrieval experiments | Broad generative reasoning and language coverage |

## Example outputs, inferred-data share, and runtime footprint

`Inferred data %` below is an **approximate** measure of how much of the final response is assembled by the pipeline rather than copied verbatim from one single memory line. High percentages usually mean stronger synthesis/reconstruction; low percentages mean the system found a very direct memory and mostly returned it.

All examples below are from the current local pipeline shape and are intended to illustrate the system's value proposition: **small, transparent, inspectable inference with modest resources**.

| Input | Example response shape | Approx. inferred data % | What produced the answer | Typical resources to run |
| --- | --- | ---: | --- | --- |
| `Who are you` | `If you ask: Who are you, I can explain that I am a text-based system that uses stored memories to respond.` | 5-10% | Mostly a direct chat memory retrieval | CPU only, Python + NumPy + SQLite, local DB with tens of rows |
| `I have headache` | `If you have a headache, it may help to rest, drink water, and reduce bright light. Furthermore, ... seek urgent medical care. Altogether, ... say where it hurts ...` | 45-65% | Multiple retrieved memories stitched by synthesis connectives and selection | CPU only, same stack, local DB with tens of rows |
| `How does DNA replication work?` | `DNA replication copies genetic material before cell division using complementary base pairing and polymerase enzymes. Furthermore, ...` | 35-55% | One direct science memory plus supporting retrieved lines | CPU only, same stack, local DB with tens of rows |
| Fragment-style KB query | Several short cast-selected facts merged into one reply | 60-80% | Cast-level reconstruction from unique noisy traversals | CPU only, same stack, local DB with fragment memories |

## Performance / resource difference at a glance

| System style | Typical runtime requirements | Knowledge transparency | Response generation cost |
| --- | --- | --- | --- |
| SRM | Standard local Python environment, SQLite file, NumPy, no GPU required | High | Retrieval + ranking + assembly over a small explicit KB |
| GPT-style transformer | Model weights, inference runtime, often remote serving or substantial local RAM/VRAM | Low to medium | Full neural forward pass over a large pretrained model |

SRM is **not** trying to out-generate a frontier transformer.
Its novelty is that it offers a practical, inspectable alternative for local memory experiments where:

- **you want to know why an answer happened**
- **you want to change behaviour by editing knowledge directly**
- **you want small-resource execution without heavyweight model serving**
- **you want retrieval and response assembly to be auditable end-to-end**

## Browser demo

- **[Open browser demo](https://htmlpreview.github.io/?https://raw.githubusercontent.com/xwiz/nametest/main/docs/index.html)**

The browser demo uses the current built-in demo KB and vocabulary expansions exported from this repo.

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
