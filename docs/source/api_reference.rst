API reference
=============

Python connectors
-----------------

The ``thotbook_amenti`` package provides build-free connectors that shell out to
the compiled Rust binaries and read the resulting lakehouse with polars.

.. automodule:: thotbook_amenti.corpus
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: thotbook_amenti.redact
   :members:
   :undoc-members:

.. automodule:: thotbook_amenti.tokenizer
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
