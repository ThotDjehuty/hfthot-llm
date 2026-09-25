//! `llm-rag` — Retrieval-Augmented Generation for thotbook-AmentI.
//!
//! HNSW vector index + hybrid BM25 search for semantic document retrieval.
//! Integrates with the Qwen3-8B model for embedding generation.

pub mod embed;
pub mod error;
pub mod hnsw;
pub mod search;
pub mod store;

pub use error::RagError;
pub use search::{ SearchEngine, SearchResult, SearchConfig };
pub use store::RagStore;
