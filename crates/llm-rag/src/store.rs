use std::path::{Path, PathBuf};

use crate::error::RagError;
use crate::hnsw::HnswIndex;
use crate::search::SearchEngine;

/// Persistent RAG store: saves/loads the HNSW index and metadata to disk.
pub struct RagStore {
    pub index_path: PathBuf,
    pub engine: SearchEngine,
}

impl RagStore {
    pub fn new(index_dir: &Path) -> Self {
        Self {
            index_path: index_dir.join("rag_index.bin"),
            engine: SearchEngine::new(),
        }
    }

    /// Save the index and metadata to disk.
    pub fn save(&self) -> Result<(), RagError> {
        let data = self.engine.index.read().serialize();
        std::fs::write(&self.index_path, &data).map_err(|e| RagError::Io {
            path: self.index_path.clone(),
            source: e,
        })?;

        // Save metadata as JSON
        let meta_path = self.index_path.with_extension("meta.json");
        let meta = serde_json::json!({
            "doc_count": self.engine.doc_texts.len(),
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
            .map_err(|e| RagError::Io {
                path: meta_path,
                source: e,
            })?;

        Ok(())
    }

    /// Load the index from disk.
    pub fn load(&mut self) -> Result<(), RagError> {
        if !self.index_path.exists() {
            return Err(RagError::NotLoaded(format!(
                "index file not found: {}",
                self.index_path.display()
            )));
        }

        let data = std::fs::read(&self.index_path).map_err(|e| RagError::Io {
            path: self.index_path.clone(),
            source: e,
        })?;

        if let Some(index) = HnswIndex::deserialize(&data) {
            self.engine.index = std::sync::Arc::new(parking_lot::RwLock::new(index));
        }

        Ok(())
    }

    /// Check if an index exists on disk.
    pub fn exists(&self) -> bool {
        self.index_path.exists()
    }
}
