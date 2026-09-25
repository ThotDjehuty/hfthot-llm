# ============================================================================
# thotbook-AmentI Makefile
# ============================================================================
# A private, local-first LLM training-to-serving pipeline in Rust.
# CPU-only, lakehouse-native, OpenAI-compatible.
# ============================================================================

.PHONY: all build release test check fmt clippy clean \
        serve train ingest rag auto paper \
        docs docs-serve \
        install setup help

# Default target
all: build

# ============================================================================
# Build Commands
# ============================================================================

## Build all crates (debug mode)
build:
	cargo build --workspace

## Build all crates (release mode, optimized)
release:
	cargo build --workspace --release

## Run all tests
test:
	cargo test --workspace

## Type-check without building
check:
	cargo check --workspace

## Format all code
fmt:
	cargo fmt --all

## Run clippy linter
clippy:
	cargo clippy --workspace -- -D warnings

## Clean build artifacts
clean:
	cargo clean

# ============================================================================
# CLI Commands (Development)
# ============================================================================

## Show CLI help
help-cli:
	cargo run -p llm-cli -- --help

## Start the inference server (default: localhost:8100)
serve:
	cargo run --release -p llm-cli -- serve

## Start server with custom model directory
serve-model:
	cargo run --release -p llm-cli -- serve --model-dir $(MODEL_DIR)

## Ingest documents into the corpus
ingest:
	cargo run -p llm-cli -- ingest $(SOURCES)

## Start continuous training
train:
	cargo run -p llm-cli -- train --data $(DATA)

## Train with custom parameters
train-custom:
	cargo run -p llm-cli -- train \
		--data $(DATA) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--mode $(MODE)

# ============================================================================
# RAG Commands
# ============================================================================

## Build the RAG index
rag-build:
	cargo run -p llm-cli -- rag build --input $(INPUT)

## Search the RAG index
rag-search:
	cargo run -p llm-cli -- rag search "$(QUERY)" --top-k $(TOP_K)

# ============================================================================
# Autonomous Research Commands
# ============================================================================

## Run autonomous research on a topic
auto-research:
	cargo run -p llm-cli -- auto research "$(TOPIC)"

## Extract training data from the local session-log corpus
auto-extract:
	cargo run -p llm-cli -- auto extract --session-corpus ../session-corpus

## Extract with custom paths
auto-extract-custom:
	cargo run -p llm-cli -- auto extract \
		--session-corpus $(SESSION_CORPUS) \
		--output $(OUTPUT)

## Check self-improvement loop status
auto-improve:
	cargo run -p llm-cli -- auto improve --data-dir $(DATA_DIR)

## Generate a research paper
paper:
	cargo run -p llm-cli -- paper "$(TOPIC)"

## Generate paper with custom sections
paper-custom:
	cargo run -p llm-cli -- paper "$(TOPIC)" \
		--output $(OUTPUT) \
		--sections $(SECTIONS)

# ============================================================================
# System Status
# ============================================================================

## Show system status
status:
	cargo run -p llm-cli -- status

## List available agents
agents:
	cargo run -p llm-cli -- agent

# ============================================================================
# Documentation
# ============================================================================

## Build Sphinx documentation
docs:
	cd docs && make html

## Serve documentation locally
docs-serve:
	cd docs/build/html && python3 -m http.server 8000

## Clean documentation
docs-clean:
	cd docs && make clean

# ============================================================================
# Installation & Setup
# ============================================================================

## Install the CLI globally
install:
	cargo install --path crates/llm-cli

## Download Qwen3-8B model weights
setup-model:
	@echo "Downloading Qwen3-8B model weights..."
	@mkdir -p data/models/qwen3-8b
	@echo "Please download weights from HuggingFace: Qwen/Qwen3-8B"
	@echo "Place safetensors files in data/models/qwen3-8b/"

## Initialize lakehouse directories
setup-lakehouse:
	@mkdir -p data/lakehouse
	@mkdir -p data/training
	@mkdir -p data/ingest
	@echo "Lakehouse directories created."

## Full setup
setup: setup-lakehouse setup-model
	@echo "Setup complete. Run 'make serve' to start the inference server."

# ============================================================================
# Development Helpers
# ============================================================================

## Run a quick smoke test
smoke-test: build
	cargo run -p llm-cli -- status
	@echo "Smoke test passed."

## Watch for changes and rebuild
watch:
	cargo watch -x "check --workspace"

## Generate dependency graph
deps-graph:
	cargo depgraph --workspace | dot -Tpng > docs/source/_static/deps.png

# ============================================================================
# Production Commands (Release Mode)
# ============================================================================

## Start production server
prod-serve:
	RUST_LOG=info cargo run --release -p llm-cli -- serve

## Production training
prod-train:
	RUST_LOG=info cargo run --release -p llm-cli -- train \
		--data $(DATA) \
		--epochs $(EPOCHS) \
		--mode lora

## Production paper generation
prod-paper:
	RUST_LOG=info cargo run --release -p llm-cli -- paper "$(TOPIC)" \
		--output output/papers \
		--sections 7

# ============================================================================
# Help
# ============================================================================

## Show this help message
help:
	@echo "thotbook-AmentI Makefile Commands"
	@echo "=================================="
	@echo ""
	@echo "Build Commands:"
	@echo "  make build          Build all crates (debug)"
	@echo "  make release        Build all crates (release)"
	@echo "  make test           Run all tests"
	@echo "  make check          Type-check without building"
	@echo "  make fmt            Format all code"
	@echo "  make clippy         Run clippy linter"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "CLI Commands:"
	@echo "  make serve          Start inference server"
	@echo "  make train DATA=... Start training"
	@echo "  make ingest SOURCES=... Ingest documents"
	@echo "  make status         Show system status"
	@echo ""
	@echo "Autonomous Research:"
	@echo "  make auto-extract   Extract training data from local session corpus"
	@echo "  make auto-research TOPIC=... Run autonomous research"
	@echo "  make paper TOPIC=... Generate research paper"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build Sphinx documentation"
	@echo "  make docs-serve     Serve docs locally"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Full setup (lakehouse + model)"
	@echo "  make install        Install CLI globally"
	@echo ""
	@echo "Examples:"
	@echo "  make auto-extract"
	@echo "  make paper TOPIC='Rough Volatility Models'"
	@echo "  make train DATA=data/training/session_corpus.jsonl EPOCHS=3"
