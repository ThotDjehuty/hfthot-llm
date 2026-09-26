use std::path::{Path, PathBuf};

use rayon::prelude::*;
use tracing::info;
use walkdir::WalkDir;

use crate::agent;
use crate::chunker::{self, Chunk, ChunkConfig};
use crate::error::IngestError;
use crate::latex;
use crate::markdown;
use crate::pdf;

/// Configuration for the ingestion pipeline.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct IngestConfig {
    pub chunk_config: ChunkConfig,
    pub sources: Vec<PathBuf>,
}


/// A single ingested document chunk.
#[derive(Debug, Clone)]
pub struct IngestedChunk {
    pub doc_id: String,
    pub source: String,
    pub source_type: SourceType,
    pub chunk: Chunk,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceType {
    Pdf,
    Markdown,
    Agent,
    Notebook,
    Latex,
}

/// Result of an ingestion run.
#[derive(Debug, Default)]
pub struct IngestReport {
    pub files_processed: usize,
    pub files_skipped: usize,
    pub chunks_created: usize,
    pub errors: Vec<String>,
}

/// The ingestion pipeline: discover → extract → chunk → emit.
pub struct IngestPipeline {
    pub config: IngestConfig,
}

impl IngestPipeline {
    pub fn new(config: IngestConfig) -> Self {
        Self { config }
    }

    /// Discover all processable files under configured sources.
    pub fn discover(&self) -> Vec<(PathBuf, SourceType)> {
        let mut files = Vec::new();
        for source in &self.config.sources {
            if source.is_file() {
                if let Some(st) = classify_file(source) {
                    files.push((source.clone(), st));
                }
            } else if source.is_dir() {
                for entry in WalkDir::new(source)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().is_file())
                {
                    let path = entry.path();
                    if let Some(st) = classify_file(path) {
                        files.push((path.to_path_buf(), st));
                    }
                }
            }
        }
        files
    }

    /// Extract text from a single file.
    pub fn extract(&self, path: &Path, source_type: &SourceType) -> Result<String, IngestError> {
        match source_type {
            SourceType::Pdf => pdf::extract_pdf(path),
            SourceType::Latex => latex::extract_latex(path),
            SourceType::Markdown | SourceType::Agent => markdown::extract_markdown(path),
            SourceType::Notebook => {
                // Extract text from .ipynb JSON cells
                let raw = std::fs::read_to_string(path).map_err(|e| IngestError::Io {
                    path: path.to_path_buf(),
                    source: e,
                })?;
                let notebook: serde_json::Value =
                    serde_json::from_str(&raw).map_err(|e| IngestError::Markdown {
                        path: path.to_path_buf(),
                        reason: format!("invalid JSON: {e}"),
                    })?;
                let mut text = String::new();
                if let Some(cells) = notebook["cells"].as_array() {
                    for cell in cells {
                        if let Some(source) = cell["source"].as_array() {
                            for line in source {
                                if let Some(s) = line.as_str() {
                                    text.push_str(s);
                                }
                            }
                            text.push('\n');
                        }
                    }
                }
                Ok(text)
            }
        }
    }

    /// Run the full ingestion pipeline on all sources.
    pub fn run(&self) -> Result<IngestReport, IngestError> {
        let files = self.discover();
        info!(count = files.len(), "discovered files for ingestion");

        let results: Vec<Result<IngestedChunk, IngestError>> = files
            .par_iter()
            .flat_map(|(path, source_type)| {
                let text = match self.extract(path, source_type) {
                    Ok(t) => t,
                    Err(e) => return vec![Err(e)],
                };
                let chunks = chunker::chunk_text(&text, &self.config.chunk_config);
                let doc_id = uuid::Uuid::new_v4().to_string();
                let source = path.to_string_lossy().to_string();

                let meta = match source_type {
                    SourceType::Agent => {
                        if let Ok(agent_doc) = agent::parse_agent_file(path) {
                            serde_json::json!({
                                "agent_name": agent_doc.name,
                                "tools": agent_doc.tools,
                            })
                        } else {
                            serde_json::json!({})
                        }
                    }
                    _ => serde_json::json!({}),
                };

                chunks
                    .into_iter()
                    .map(|chunk| {
                        Ok(IngestedChunk {
                            doc_id: doc_id.clone(),
                            source: source.clone(),
                            source_type: source_type.clone(),
                            chunk,
                            metadata: meta.clone(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut report = IngestReport::default();
        let mut all_chunks = Vec::new();

        for result in results {
            match result {
                Ok(chunk) => {
                    report.chunks_created += 1;
                    all_chunks.push(chunk);
                }
                Err(e) => {
                    report.errors.push(e.to_string());
                }
            }
        }

        // Count unique files
        let mut seen = std::collections::HashSet::new();
        for chunk in &all_chunks {
            if seen.insert(chunk.source.clone()) {
                report.files_processed += 1;
            }
        }

        info!(
            files = report.files_processed,
            chunks = report.chunks_created,
            errors = report.errors.len(),
            "ingestion complete"
        );

        Ok(report)
    }
}

fn classify_file(path: &Path) -> Option<SourceType> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("pdf") => Some(SourceType::Pdf),
        Some("md") => {
            let name = path.file_name()?.to_str()?;
            if name.ends_with(".agent.md") {
                Some(SourceType::Agent)
            } else {
                Some(SourceType::Markdown)
            }
        }
        Some("ipynb") => Some(SourceType::Notebook),
        Some("tex") => Some(SourceType::Latex),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_extensions() {
        assert_eq!(
            classify_file(Path::new("test.pdf")),
            Some(SourceType::Pdf)
        );
        assert_eq!(
            classify_file(Path::new("test.md")),
            Some(SourceType::Markdown)
        );
        assert_eq!(
            classify_file(Path::new("foo.agent.md")),
            Some(SourceType::Agent)
        );
        assert_eq!(
            classify_file(Path::new("test.ipynb")),
            Some(SourceType::Notebook)
        );
        assert_eq!(classify_file(Path::new("test.rs")), None);
    }
}
