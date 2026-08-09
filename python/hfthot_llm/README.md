# hfthot-llm (Python connectors)

Thin, pragmatic Python connectors for the `hfthot-llm` Rust pipeline
(`crates/llm-corpus`, `crates/llm-tokenize`). No pyo3 build step: the
connectors shell out to the compiled Rust binaries via `subprocess` and read
the resulting lakehouse / tokenized data with **polars** (parquet / Arrow IPC).

## Install

```sh
pip install polars pyarrow requests tokenizers
# or, from this directory:
pip install -e .
```

Requires Python >= 3.10.

## CLI

From `python/hfthot_llm`:

```sh
# List the datasets.* tables present in the lakehouse (mode RÉEL).
python -m hfthot_llm tables

# Rows / columns / bytes for a table (parquet footers, no full scan).
python -m hfthot_llm stats --table datasets.sessions

# Run the compiled llm-corpus binary against the SQLite source DBs.
# (Binary auto-detected: target/release/llm-corpus, falls back to debug.
#  Build it first with: cd hfthot-llm && cargo build --release)
python -m hfthot_llm ingest --historia /path/historia.db --thotbook /path/thotbook.db
```

All commands accept `--lakehouse <dir>` to point at a different lakehouse.

## Library

```python
from hfthot_llm import CorpusClient, load_tokenizer, tokenize_doc, redact_secrets

# Lakehouse reads (parquet via polars)
client = CorpusClient()                      # defaults to data/lakehouse
client.tables()                              # ['datasets.citations', 'datasets.corpus', ...]
df = client.read("datasets.sessions")        # polars.DataFrame
client.stats("datasets.corpus")              # {'rows': ..., 'columns': ..., 'bytes': ...}

# Ingest via the Rust binary (returns subprocess.CompletedProcess)
res = client.ingest("historia.db", "thotbook.db")
assert res.returncode == 0, res.stderr

# Tokenization (data/tokenizers/qwen3-tokenizer.json by default)
tok = load_tokenizer()
ids = tokenize_doc("bonjour", tok)           # list[int]

# Secret redaction (parity with redact.ts / redact.rs)
redact_secrets("token=abc123")               # 'token=[REDACTED]'
contains_secrets("token=[REDACTED]")         # False
```

## Layout

| File | Purpose |
| --- | --- |
| `hfthot_llm/corpus.py` | `CorpusClient` — discover/read/stats lakehouse tables, `ingest` via `llm-corpus` |
| `hfthot_llm/tokenizer.py` | `load_tokenizer`, `tokenize_doc` (HuggingFace `tokenizers`) |
| `hfthot_llm/redact.py` | `redact_secrets`, `contains_secrets` (Python port of `redact.ts`/`redact.rs`) |
| `hfthot_llm/cli.py` | `python -m hfthot_llm` entry point (`tables`, `stats`, `ingest`) |
