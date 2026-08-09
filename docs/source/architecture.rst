Architecture
============

hfthot-llm turns your private documents, research notes, session transcripts
and paper indexes into a *personal language model* — fully on your own
hardware. It is built as six focused Rust crates wired together by a Polarway
Delta lakehouse.

Pipeline
--------

.. code-block:: text

  M1              M2              M3              M4              M5
 +--------+     +--------+     +--------+     +--------+     +--------------+
 | corpus | --> |tokenize| --> | train  | --> |  rag   | --> | serve        |
 | ingest |     |sharded |     | (LoRA) |     |(HNSW)  |     | OpenAI API   |
 +--------+     +--------+     +--------+     +--------+     +--------------+
      ^                            |              ^              |
      |                     M6 citation crawler --|              |
      +------------------------ SQLite -> Arrow -> Delta -------+

Every stage reads from / writes to an Apache Arrow -> Delta Lake lakehouse
(``data/lakehouse``), so any step can be re-run idempotently.

Milestones
----------

.. list-table::
   :header-rows: 1

   * - Milestone
     - Crate
     - Purpose
     - Status
   * - M1
     - ``llm-corpus``
     - Redact + ingest SQLite sources (sessions, papers, notebooks, equations, arxiv) into Delta
     - implemented
   * - M2
     - ``llm-tokenize``
     - Tokenize lakehouse tables into sharded Arrow IPC files
     - implemented
   * - M3
     - ``llm-train``
     - CPU QLoRA SFT of Qwen3-4B on the tokenized corpus
     - in development
   * - M4
     - ``llm-rag``
     - Embeddings + HNSW retrieval over the corpus
     - in development
   * - M5
     - ``llm-serve``
     - Private OpenAI-compatible Qwen3-8B server (candle, CPU)
     - implemented
   * - M6
     - ``llm-corpus`` (crawler)
     - arXiv reference-closure citation crawler
     - planned

Design principles
-----------------

* **Private by design** — content is redacted at ingest time (``redact.rs`` /
  ``redact.py``, parity with the JS reference implementation). A write-time
  leak audit refuses to append any row that still contains secrets after
  redaction.
* **CPU-only** — no GPU required; quantization and LoRA keep training and
  inference feasible on CPU (32 GB RAM recommended).
* **Lakehouse-native** — every stage produces versioned, queryable Delta
  tables with time-travel.
* **OpenAI-compatible surface** — drop-in replacement for
  ``POST /v1/chat/completions`` on your own model.
* **Parity everywhere** — redaction rules and tokenizer defaults stay in
  lock-step between Rust, Python and the JS reference.
