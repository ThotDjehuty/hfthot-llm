# Changelog

All notable changes to thotbook-AmentI are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-09-25

### Added

- **`llm-auto`** — autonomous orchestration crate (M7): task decomposition,
  sub-agent spawning, self-training loop, and paper generation from the
  research pipeline. Training data is sourced from a locally configured
  session-log corpus directory (`--session-corpus`, generic Markdown
  `## Goal` / `## Summary` documents — no specific dataset is bundled or
  disclosed).
- **`llm-ingest`** — universal document ingestion crate: PDF, Markdown, and
  agent-definition file parsing into the corpus lakehouse.
- **`llm-salviers`** — agent dispatch system: semantic (keyword/BM25) matching
  of incoming queries to registered agent definitions.
- `llm-rag`: dedicated `embed`, `error`, `hnsw`, `search`, and `store` modules,
  splitting retrieval concerns out of `lib.rs`.
- `llm-serve`: `agent.rs` and `rag.rs` modules wiring the RAG pipeline and
  autonomous agents into the OpenAI-compatible server.
- `llm-train`: full training-loop modules — `arrow_dataset`, `checkpoint`,
  `dataset`, `error`, `lora`, `loss`, `metrics`, `model`, `optimizer`,
  `scheduler`, `trainer`.
- `llm-cli`: standalone `main.rs` entry point.
- Sphinx documentation: `docs/source/algorithms/` (embeddings, inference,
  orchestration, rag, training pages) and top-level `algorithms.rst`.
- Sphinx theming: Thot transparent logo, furo light/dark brand colors, and
  GitHub footer/source links, matching the optimiz-rs documentation template.
- `docs/AUTONOMOUS_ARCHITECTURE.md` — design notes for the M7 autonomous pipeline.
- Root `Makefile` with build/test/docs targets.

### Changed

- Workspace version bumped `0.1.0` → `0.2.0` to match the documentation
  release version.

### Status

- `cargo build --workspace` and `cargo test --workspace` are green
  (70/70 tests passing across all 9 crates).
- `cargo clippy --workspace -- -D warnings` is not yet clean (pending
  follow-up cleanup pass).

[0.2.0]: https://github.com/ThotDjehuty/hfthot-llm/releases/tag/v0.2.0
