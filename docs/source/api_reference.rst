API reference
=============

Python connectors
-----------------

The ``hfthot_llm`` package provides build-free connectors that shell out to
the compiled Rust binaries and read the resulting lakehouse with polars.

.. automodule:: hfthot_llm.corpus
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hfthot_llm.redact
   :members:
   :undoc-members:

.. automodule:: hfthot_llm.tokenizer
   :members:
   :undoc-members:

HTTP surface (``llm-serve``)
----------------------------

.. list-table::
   :header-rows: 1

   * - Endpoint
     - Description
   * - ``GET /health``
     - Liveness probe
   * - ``GET /v1/models``
     - Model id (e.g. ``Qwen/Qwen3-8B``)
   * - ``POST /v1/chat/completions``
     - OpenAI-compatible chat completion

Delta tables (``datasets.*``)
-----------------------------

* ``datasets.sessions`` — session transcripts (redacted)
* ``datasets.corpus`` — papers, notebooks and arXiv entries
* ``datasets.equations`` — LaTeX equations indexed per source
* ``datasets.citations`` — citation edges (seed set from the arXiv index)
