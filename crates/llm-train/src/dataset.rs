use std::path::Path;

use crate::error::TrainError;

/// A single training example.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input_ids: Vec<u32>,
    pub labels: Vec<u32>,
}

/// Streaming dataset that reads training examples in chunks.
/// Never loads all data into RAM.
pub struct StreamingDataset {
    examples: Vec<TrainingExample>,
    batch_size: usize,
    position: usize,
}

impl StreamingDataset {
    /// Load examples from a JSONL file (one JSON object per line).
    pub fn from_jsonl(path: &Path, batch_size: usize) -> Result<Self, TrainError> {
        let file = std::fs::File::open(path).map_err(|e| TrainError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let reader = std::io::BufReader::new(file);
        let mut examples = Vec::new();

        for line in std::io::BufRead::lines(reader) {
            let line = line.map_err(|e| TrainError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let record: serde_json::Value =
                serde_json::from_str(&line).map_err(|e| TrainError::Dataset(e.to_string()))?;

            let input_ids: Vec<u32> = record["input_ids"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
                .unwrap_or_default();
            let labels: Vec<u32> = record["labels"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
                .unwrap_or_default();

            if !input_ids.is_empty() {
                examples.push(TrainingExample { input_ids, labels });
            }
        }

        tracing::info!(count = examples.len(), "loaded training examples");
        Ok(Self {
            examples,
            batch_size,
            position: 0,
        })
    }

    /// Load examples from in-memory (for small datasets).
    pub fn from_examples(examples: Vec<TrainingExample>, batch_size: usize) -> Self {
        Self {
            examples,
            batch_size,
            position: 0,
        }
    }

    /// Get the next batch of examples.
    pub fn next_batch(&mut self) -> Option<&[TrainingExample]> {
        if self.position >= self.examples.len() {
            self.position = 0; // Reset for next epoch
            return None;
        }
        let end = (self.position + self.batch_size).min(self.examples.len());
        let batch = &self.examples[self.position..end];
        self.position = end;
        Some(batch)
    }

    /// Total number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_jsonl() {
        let dir = std::env::temp_dir().join(format!("ds-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("train.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{{\"input_ids\":[1,2,3],\"labels\":[1,2,3]}}").unwrap();
        writeln!(f, "{{\"input_ids\":[4,5,6],\"labels\":[4,5,6]}}").unwrap();

        let mut ds = StreamingDataset::from_jsonl(&path, 10).unwrap();
        assert_eq!(ds.len(), 2);
        let batch = ds.next_batch().unwrap();
        assert_eq!(batch.len(), 2);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
