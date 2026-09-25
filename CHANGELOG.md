# Changelog

All notable changes to thotbook-AmentI are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-09-25

### Added

- **`thotgrad`** — array-level reverse-mode automatic differentiation in
  NumPy (~250 lines), replacing PyTorch entirely. Input derivatives are
  propagated analytically through the tanh recursion and stay on the same
  tape as the value, so one reverse sweep yields parameter gradients of a
  loss containing `u`, `u_x` and `u_xx` (the PINN double-backward).
- **`thotopt`** — AdamW (decoupled decay, mirroring the Rust optimiser) and
  L-BFGS (Nocedal two-loop recursion + Armijo line search), filling the
  gradient-based gap left by optimiz-rs's gradient-free optimisers.
- **`test_thotgrad.py`** — every gradient pinned against central finite
  differences: elementary ops ~1e-10, `u_x` 1e-12, `u_xx` 1e-6, full PINN
  parameter gradient 1.8e-11 max-norm with cosine similarity 1.000000;
  optimisers checked against closed-form optima.

- **Physics-Informed Neural Networks** (`notebooks/04_pinn_physics.ipynb`,
  `pinn_core.py`, `pinn_problems.py`) — solves the Dirac, Brans-Dicke and
  Wheeler-DeWitt equations on CPU, each graded against an independent
  reference (exact closed form / exact Nariai power law / Radau at
  `rtol=1e-12` cross-checked by an Airy asymptotic). Relative L2 errors
  1.7e-3, 4.3e-3 and 8.3e-6 respectively.
- **Membership inference at scale** (`notebooks/02_mia_real_corpus.ipynb`) —
  the privacy audit run over a real research corpus with document-level
  splits, calibrated membership probabilities (Platt scaling + reliability
  diagram + ECE) and block-bootstrap CIs.
- **Federated learning** (`notebooks/03_federated_ensemble.ipynb`,
  `fed_core.py`) — FedAvg with an honest-but-curious server running the
  update-based attack, plus the four defence families of Bai et al.'s survey
  (partial sharing, secure aggregation, noise perturbation, anomaly
  detection) and the measured privacy/utility trade-off.
- `docs/source/algorithms/variational_calculus.rst` — Euler-Lagrange,
  gradient flow, the ELBO, and the mapping from each functional-minimisation
  problem to its optimiz-rs primitive.
- `docs/source/algorithms/membership_inference.rst` — attack/defence theory
  with citations (Shokri 2017, Yeom 2018, Carlini 2022).
- Explicit AdamW update equations in `docs/source/algorithms/training.rst`.
- `crates/llm-train/examples/membership_probe.rs` — real LoRA training probe
  (candle autograd + `AdamWOptimizer`) with `--hidden` / `--limit` controls.

### Fixed

- **`AdamWOptimizer` never applied weight decay.** The `weight_decay` field
  was stored but unused, making the optimiser plain Adam despite the name and
  the documented equation. Decay is now applied decoupled (Loshchilov &
  Hutter 2019), with a regression test that fails if it regresses.
- All 33 `cargo clippy --workspace --all-targets -- -D warnings` errors
  (manual prefix stripping, private-type-in-public-field, `&PathBuf` over
  `&Path`, `loop`/`match` over `while let`).

### Changed

- Workspace version `0.2.0` → `0.3.0`.
- `notebooks/data/` is gitignored: it holds verbatim text extracted from
  private and third-party copyrighted PDFs and must never reach a public
  remote. Regenerate locally via `THOTBOOK_PAPERS_DIR=... python3
  notebooks/extract_corpus.py`.
- `papers/thotbook_amenti_arxiv.tex`: added a System Capabilities section
  covering the autodiff engine, optimisers, document verification by equation
  solving, the privacy audit and federated training, plus an explicit
  negative-results subsection. Corrected the hardware section, which
  claimed an Apple M1 Max; all measurements in this repo were taken on an
  Intel Core i9-8950HK with no GPU. Added a verified-benchmarks section and
  an explicit threats-to-validity subsection marking the inherited throughput
  and retrieval tables as provisional and not re-measured.

### Notes

- 71/71 Rust tests pass; `cargo clippy -D warnings` is clean.
- PyTorch is no longer a dependency anywhere in the project.

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

[0.3.0]: https://github.com/ThotDjehuty/hfthot-llm/releases/tag/v0.3.0
[0.2.0]: https://github.com/ThotDjehuty/hfthot-llm/releases/tag/v0.2.0
