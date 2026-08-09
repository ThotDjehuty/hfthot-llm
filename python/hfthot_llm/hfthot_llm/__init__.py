"""hfthot-llm — thin Python connectors for the Rust pipeline.

These connectors shell out to the compiled Rust binaries (``llm-corpus``,
``llm-tokenize``) via subprocess and read the resulting lakehouse/parquet
data with polars — no pyo3 build step required.
"""

from __future__ import annotations

from .corpus import CorpusClient
from .redact import REDACTED, contains_secrets, redact_secrets
from .tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer, tokenize_doc

__version__ = "0.1.0"

__all__ = [
    "CorpusClient",
    "DEFAULT_TOKENIZER_PATH",
    "REDACTED",
    "__version__",
    "contains_secrets",
    "load_tokenizer",
    "redact_secrets",
    "tokenize_doc",
]
