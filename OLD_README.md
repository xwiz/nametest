# SRM — Stochastic Resonance Memory

A pure-algorithmic associative memory system.  
**No LLM. No embeddings. No external NLP libraries. `stdlib + numpy` only.**

---

## How It Works

```
Query text
   │
   ▼  expand_query()          bridge vocabulary gap (e.g. "brain" → "neuron synapse …")
   │
   ▼  encode()                project into 128-bit Hamming space via IDF-weighted SimHash
   │
   ▼  traverse()              fire 40 noisy probes; each lands on its nearest memory (votes)
   │
   ▼  hybrid re-rank          0.55 × vote_share  +  0.45 × TF-IDF cosine
   │
   ▼  synthesise() / RWEA     score sentences, MMR de-duplicate, add connectives
   │
   ▼  Response paragraph
```

### SimHash encoding
Each text (memory or query) is projected to a 128-bit binary code.  
Token types and weights:

| Token type  | Weight        | Purpose                              |
|-------------|---------------|--------------------------------------|
| Unigram     | 2.5 × IDF     | Primary semantic signal              |
| Bigram      | 1.5 × mean IDF| Captures phrase proximity            |
| Char 3-gram | 0.8           | Morphological fallback for rare terms|

Bit *b* of the code = 1 iff the IDF-weighted sum of MD5 projections along *b* > 0.

### Stochastic traversal
40 independent probes are fired from the query code, each with random bit-flips
at rate 0.12. Each probe "lands on" its nearest memory (argmin Hamming distance).
True attractors accumulate votes consistently; noise prevents false-positive lock-in.

### RWEA synthesis
Sentence scoring: `resonance × cosine(query, sent) × pos_weight × content_ratio`  
Selection: greedy MMR — skip any sentence with cosine > 0.68 to an already-chosen one.  
Assembly: lightweight discourse connectives (support / contrast / conclude).

---

## Quickstart

```bash
pip install numpy
python main.py --seed                         # load 25-entry KB, enter REPL
python main.py --seed -q "DNA replication"    # single query
python main.py --seed --verbose               # REPL + attractor debug info
python main.py --load my_facts.txt            # custom knowledge base
python main.py --stats                        # inspect store
```

### REPL commands

| Command          | Action                               |
|------------------|--------------------------------------|
| `<question>`     | Query the memory store               |
| `/add <text>`    | Store a new memory                   |
| `/delete <id>`   | Remove a memory by DB id             |
| `/list`          | List all memories with their ids     |
| `/stats`         | Show configuration & counts          |
| `/clear`         | Wipe all memories                    |
| `/quit`          | Exit                                 |

### CLI flags

| Flag               | Default          | Description                         |
|--------------------|------------------|-------------------------------------|
| `--db PATH`        | srm_memory.db    | SQLite path                         |
| `--seed`           | —                | Load built-in 25-entry KB           |
| `--clear`          | —                | Wipe store before starting          |
| `--load FILE`      | —                | Load memories from text file        |
| `--list`           | —                | List memories and exit              |
| `--stats`          | —                | Print store stats and exit          |
| `-q TEXT`          | —                | Single query and exit               |
| `--casts N`        | 40               | Stochastic probes per query         |
| `--noise F`        | 0.12             | Per-bit flip probability            |
| `--top-k K`        | 5                | Max attractor candidates            |
| `-v / --verbose`   | —                | Show attractor details + scores     |

---

## File Structure

```
srm_project/
├── main.py               Entry point
├── requirements.txt      numpy only
│
├── srm/
│   ├── __init__.py       Public API: MemoryStore, srm_query, SAMPLE_KB
│   ├── config.py         All constants, stopwords, expansions, KB
│   ├── nlp.py            tokenise, IDF, TF-IDF, cosine, expand_query
│   ├── encoding.py       SimHash encode, Hamming distance
│   ├── traversal.py      Stochastic casting (_cast, traverse)
│   ├── synthesis.py      RWEA sentence scoring + MMR assembly
│   ├── store.py          SQLite-backed MemoryStore
│   ├── pipeline.py       srm_query() end-to-end pipeline
│   └── cli.py            Argument parser, REPL, box renderer
│
└── tests/
    └── test_srm.py       36 unit + integration tests
```

---

## Using SRM as a library

```python
from srm import MemoryStore, srm_query

store = MemoryStore("my.db")
store.add("The mitochondria is the powerhouse of the cell.")
store.add("Neural networks learn via backpropagation.")

result = srm_query("how do cells produce energy", store)
print(result["response"])
# → "The mitochondria is the powerhouse of the cell …"

# Rich result dict
for a in result["attractor_details"]:
    print(f"  #{a['mem_id']}  votes={a['votes']}  cos={a['cosine']}  {a['text'][:60]}")
```

---

## Extending SRM

### Add more vocabulary expansions
Edit `srm/config.py` → `EXPANSIONS` dict.  
Keys are surface query terms; values are domain synonyms present in your KB.

### Larger knowledge base
```bash
python main.py --load facts.txt   # one fact per line
```

Or call `store.add(text)` programmatically.

### Tune retrieval behaviour

| Parameter   | Location          | Effect                                   |
|-------------|-------------------|------------------------------------------|
| `NUM_CASTS` | config.py         | More casts → higher vote signal, slower  |
| `NOISE`     | config.py         | Higher noise → wider search radius       |
| `W_VOTE`    | config.py         | Weight of vote share in hybrid score     |
| `W_COS`     | config.py         | Weight of cosine similarity              |
| `SIM_THRESH`| config.py         | MMR threshold — lower = more diverse     |
| `MAX_WORDS` | config.py         | Response word budget                     |

### Swap the synthesis layer
`synthesise()` in `srm/synthesis.py` is self-contained.  
You can replace it with any function matching:

```python
def synthesise(
    query: str,
    top_attractors: list[tuple[int, int, str]],
    **kwargs,
) -> str: ...
```

For example, you could pass `top_attractors` straight to an LLM for
generative synthesis while keeping the algorithmic retrieval layer.

---

## Running tests

```bash
python -m pytest tests/ -v          # requires pytest
python tests/test_srm.py            # stdlib unittest, no pytest needed
```

36 tests cover: tokenisation, IDF, cosine, encoding determinism,
Hamming geometry, traversal vote totals, store CRUD, cache invalidation,
end-to-end domain retrieval accuracy.

---

## Bugs fixed vs. original monolith

| # | Bug | Fix |
|---|-----|-----|
| 1 | `_box()` used `textwrap.fill(body)` which swallowed embedded `\n`, merging the verbose attractor badge and memory text onto one line | `_box()` now splits body on `\n` first, then wraps each paragraph independently |
| 2 | No `/delete` command in REPL | Added `/delete <id>` |
| 3 | `--stats` only available inside REPL | Added `--stats` CLI flag |
| 4 | `VOTE_FLOOR` and `MIN_COS` were module-level magic numbers | Moved to `config.py` as named constants |
