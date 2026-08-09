Roadmap
=======

In development
--------------

* **M3 ``llm-train``** — CPU QLoRA SFT of Qwen3-4B on the tokenized corpus,
  reusing the LoRA / Q4-quantization math in ``optimiz-rs``.
* **M4 ``llm-rag``** — corpus embeddings + HNSW retrieval (the HNSW index is
  already available as a reusable primitive in ``optimiz-rs``).
* **``llm-cli``** — a single entrypoint orchestrating the whole pipeline.

Planned
-------

* **M6 citation crawler** — arXiv reference-closure crawl to enrich the
  ``datasets.citations`` edges (arXiv API + OpenAlex / Semantic Scholar).
* **Time-travel audit tooling** — inspect historical versions of any
  ``datasets.*`` table through the Polarway lakehouse.
