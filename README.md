> **EDUCATIONAL PURPOSE ONLY** — This software is provided strictly for
> research, learning, and academic exploration. It is not intended for
> production use, commercial deployment, or any application where failure
> could cause harm. Use at your own risk.

# thotbook-AmentI

> **A private, local-first LLM training-to-serving pipeline in Rust.**
> Ingest your own knowledge corpus → tokenize → fine-tune → retrieve → serve
> an OpenAI-compatible private model. CPU-friendly, redaction-first,
> lakehouse-native.

---

## Overview

thotbook-AmentI turns your private documents, research notes, session transcripts
and paper indexes into a *personal language model* — fully on your own
hardware. It is built as six focused Rust crates wired together by a Polarway
Delta lakehouse:

```text
 M1            M2              M3              M4              M5
┌────────┐   ┌────────┐    ┌────────┐    ┌────────┐    ┌──────────────┐
│ corpus │──▶│tokenize│───▶│ train  │───▶│  rag   │───▶│ serve        │
│ ingest │   │sharded │    │ (LoRA) │    │(HNSW)  │    │ OpenAI API    │
└────────┘   └────────┘    └────────┘    └────────┘    └──────────────┘
     ▲                           │              ▲              │
     │                     M6 citation crawler ──┘              │
     └─────────────────────── SQLite → Arrow → Delta ──────────┘
```

Every stage reads from / writes to an **Apache Arrow → Delta Lake** lakehouse
(`data/lakehouse`), so any step can be re-run idempotently without losing data.

| Milestone | Crate | Purpose | Status |
| --- | --- | --- | --- |
| M1 | `llm-corpus` | Redact + ingest SQLite sources (sessions, papers, notebooks, equations, arxiv) into Delta tables | ✅ implemented |
| M2 | `llm-tokenize` | Tokenize lakehouse tables into sharded Arrow IPC training files | ✅ implemented |
| M3 | `llm-train` | CPU QLoRA SFT of Qwen3-4B on the tokenized corpus | 🚧 in development |
| M4 | `llm-rag` | Embeddings + HNSW retrieval over the corpus | 🚧 in development |
| M5 | `llm-serve` | Private OpenAI-compatible Qwen3-8B server (candle, CPU) | ✅ implemented |
| M6 | `llm-corpus` (crawler) | arXiv reference-closure citation crawler → `datasets.citations` | 🚧 planned |

## Design principles

- **Private by design.** Content is redacted at ingest time
  (`redact.rs` / `redact.py`, parity with the JS reference implementation).
  A write-time leak audit refuses to append any row that still contains
  secrets after redaction.
- **CPU-only.** No GPU required — the whole pipeline targets a 32 GB RAM
  laptop. Quantization and LoRA keep training and inference feasible on CPU.
- **Lakehouse-native.** Every stage produces versioned, queryable Delta
  tables (time-travel included) instead of ad-hoc files.
- **OpenAI-compatible surface.** Drop-in replacement for
  `POST /v1/chat/completions` on your own model.
- **Parity everywhere.** The redaction rules and tokenizer defaults are kept
  in lock-step between Rust, Python and the JS reference implementation.

## Repository layout

```text
thotbook-AmentI/
├── Cargo.toml                  # workspace root (6 members)
├── crates/
│   ├── llm-corpus/             # M1: SQLite → redacted Arrow → Delta
│   ├── llm-tokenize/           # M2: tokenizer + sharded Arrow IPC
│   ├── llm-train/              # M3: QLoRA SFT (placeholder → in dev)
│   ├── llm-rag/                # M4: embeddings + HNSW (placeholder → in dev)
│   ├── llm-serve/              # M5: axum OpenAI-compatible server
│   └── llm-cli/                # unified CLI entrypoint (placeholder)
├── python/thotbook_amenti/          # thin Python connectors (polars/pyarrow)
├── data/                       # lakehouse + tokenized shards (gitignored)
└── docs/                       # Sphinx docs (ReadTheDocs)
```

## Requirements

- **Rust** 1.92+ (stable)
- **Python** 3.10+ (only for the optional Python connectors)
- 16 GB+ RAM recommended (32 GB for M3 training / M5 Qwen3-8B serving)
- macOS or Linux

> **Note on `polarway-lakehouse`.** The workspace resolves
> `polarway-lakehouse` from a sibling checkout via a path dependency
> (`../polarway/polarway-lakehouse`). It is a private companion crate
> (Delta lakehouse primitives: `DeltaStore`, `ensure_table`, `append`,
> `scan`, `sql`, `compaction`). Check it out next to this repository and
> `cargo build` will resolve it. Everything else is crates.io.

## Quick start

```sh
# 1. Build the pipeline binaries
cargo build --release

# 2. Ingest SQLite sources into the lakehouse (redacts, refuses leaked rows)
cargo run --release -p llm-corpus -- ingest \
  --historia  /path/to/historia.db \
  --thotbook  /path/to/thotbook.db \
  --lakehouse data/lakehouse
# → writes datasets.sessions, datasets.corpus, datasets.equations, datasets.citations

# 3. Tokenize a lakehouse table into training shards
#    (downloads the Qwen3 tokenizer on first use)
cargo run --release -p llm-tokenize -- tokenize \
  --lakehouse data/lakehouse \
  --table datasets.sessions

# 4. Serve the private model (OpenAI-compatible, http://127.0.0.1:8100)
cargo run --release -p llm-serve -- --model-dir data/models/qwen3-8b
```

The tokenizer is fetched automatically on first use:
`https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json`
→ `data/tokenizers/qwen3-tokenizer.json`.

### Example ingest output

```text
2026-08-09 INFO  llm_corpus: starting ingest
2026-08-09 INFO  rows=342 version=1 "wrote sessions"
2026-08-09 INFO  rows=256 version=1 "wrote corpus"
2026-08-09 INFO  rows=1943 version=1 "wrote equations"
2026-08-09 INFO  rows=60 version=1 "wrote citations"
2026-08-09 INFO  llm_corpus: ingest complete
```

Leaked rows (secrets still present after redaction) are reported and the
write is refused — the lakehouse never sees them.

## OpenAI-compatible server (M5)

`llm-serve` runs a **Qwen3-8B** model via `candle` with a seeded
greedy / temperature / top-p sampler, wrapped in an axum HTTP server.

| Endpoint | Description |
| --- | --- |
| `GET /health` | liveness probe |
| `GET /v1/models` | model id (e.g. `Qwen/Qwen3-8B`) |
| `POST /v1/chat/completions` | OpenAI-compatible chat completion |

```sh
curl http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Explain a Hawkes process."}],"temperature":0.7}'
```

Configuration is env-driven with sane defaults; every setting can be
overridden by CLI flags:

| Env var | Default | Description |
| --- | --- | --- |
| `LLM_SERVE_MODEL_ID` | `Qwen/Qwen3-8B` | HF model id (shown in `/v1/models`) |
| `LLM_SERVE_MODEL_DIR` | `data/models/qwen3-8b` | dir with `config.json` + `*.safetensors` |
| `LLM_SERVE_TOKENIZER_PATH` | `data/tokenizers/qwen3-tokenizer.json` | tokenizer file |
| `LLM_SERVE_HOST` | `127.0.0.1` | bind interface |
| `LLM_SERVE_PORT` | `8100` | bind port |
| `LLM_SERVE_MAX_TOKENS` | `512` | default generation length |
| `LLM_SERVE_TEMPERATURE` | `0.7` | sampling temperature (0 = greedy) |
| `LLM_SERVE_TOP_P` | `0.9` | nucleus sampling |

Download the weights with:

```sh
huggingface-cli download Qwen/Qwen3-8B-Instruct --local-dir data/models/qwen3-8b
```

## Python connectors

Thin, build-free Python connectors that shell out to the compiled Rust
binaries and read results with **polars** — plus a `ChatClient` for the
local server:

```python
from thotbook_amenti import ChatClient, CorpusClient, load_tokenizer, tokenize_doc, redact_secrets

# Local open-source chat — no API key, no billing, free by design.
with ChatClient() as assistant:            # http://127.0.0.1:8100/v1, Qwen/Qwen3-8B
    if assistant.is_online():
        reply = assistant.ask("Give one line of Kähler geometry intuition.")
        print(reply.text)

client = CorpusClient()              # defaults to data/lakehouse
client.tables()                      # ['datasets.citations', 'datasets.corpus', ...]
df = client.read("datasets.sessions")  # polars.DataFrame

redact_secrets("token=abc123")       # 'token=[REDACTED]'
contains_secrets("token=[REDACTED]") # False
```

```sh
pip install -e python/thotbook_amenti
python -m thotbook_amenti tables
python -m thotbook_amenti health     # server status
python -m thotbook_amenti chat "What is a fibration?"   # local inference
```

## A free Copilot, built by us

thotbook-AmentI is the model side of the HFThot Research Lab toolset: the
same corpus pipeline that ingests our sessions, notebooks and paper indexes
can fine-tune (`llm-train`) and serve a **local, open-source, CPU-only**
assistant — no subscription, no data leaving the machine. It is open-source
(Apache 2.0 weights, MIT/Apache code), free by principle (public research,
no vendor lock-in) and free by design (local-first, private).

## Testing

```sh
cargo test --workspace   # 39 tests: llm-corpus (9), llm-tokenize (10), llm-serve (20)
pytest python/thotbook_amenti/tests   # ChatClient mock + live integration (live auto-skips when server offline)
```

The redaction suite keeps Rust, Python and the JS reference in parity, and
`llm-serve` includes live HTTP tests against the running axum router.

## Documentation

Full documentation is hosted on **ReadTheDocs** (same account that hosts
`optimiz-rs`): **https://thotbook-AmentI.readthedocs.io/** — see the `docs/`
folder for the Sphinx sources.

## Roadmap

- **M3 `llm-train`** — CPU QLoRA SFT of Qwen3-4B on the tokenized corpus,
  reusing the LoRA / Q4-quantization math in `optimiz-rs`.
- **M4 `llm-rag`** — corpus embeddings + HNSW retrieval (the HNSW index is
  already exercised as a reusable primitive).
- **M6 citation crawler** — arXiv reference-closure to enrich
  `datasets.citations` edges.
- **`llm-cli`** — a single entrypoint orchestrating the whole pipeline.

## License

MIT OR Apache-2.0 (see `LICENSE`).
