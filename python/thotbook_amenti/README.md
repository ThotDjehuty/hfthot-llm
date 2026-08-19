# thotbook-AmentI (Python connectors)

Thin, pragmatic Python connectors for the `thotbook-AmentI` Rust pipeline
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

From `python/thotbook_amenti`:

```sh
# List the datasets.* tables present in the lakehouse (mode RÉEL).
python -m thotbook_amenti tables

# Rows / columns / bytes for a table (parquet footers, no full scan).
python -m thotbook_amenti stats --table datasets.sessions

# Run the compiled llm-corpus binary against the SQLite source DBs.
# (Binary auto-detected: target/release/llm-corpus, falls back to debug.
#  Build it first with: cd thotbook-AmentI && cargo build --release)
python -m thotbook_amenti ingest --historia /path/historia.db --thotbook /path/thotbook.db
```

All commands accept `--lakehouse <dir>` to point at a different lakehouse.

## Library

```python
from thotbook_amenti import CorpusClient, load_tokenizer, tokenize_doc, redact_secrets

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
| `thotbook_amenti/corpus.py` | `CorpusClient` — discover/read/stats lakehouse tables, `ingest` via `llm-corpus` |
| `thotbook_amenti/tokenizer.py` | `load_tokenizer`, `tokenize_doc` (HuggingFace `tokenizers`) |
| `thotbook_amenti/redact.py` | `redact_secrets`, `contains_secrets` (Python port of `redact.ts`/`redact.rs`) |
| `thotbook_amenti/cli.py` | `python -m thotbook_amenti` entry point (`tables`, `stats`, `ingest`) |
