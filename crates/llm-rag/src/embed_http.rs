//! Text embeddings from a local Ollama server.
//!
//! [`crate::embed`] projects an already-computed hidden-state tensor, which
//! presumes the caller has loaded and run a model. Indexing a corpus needs the
//! step before that — text in, vector out — and doing it over HTTP keeps the
//! engine **model-agnostic**: any embedding model Ollama can serve works, and
//! swapping it is an environment variable rather than a rebuild.
//!
//! ```text
//! THOTBOOK_EMBED_MODEL=nomic-embed-text   # default
//! THOTBOOK_EMBED_URL=http://localhost:11434
//! ```

use serde::Deserialize;

use crate::error::RagError;

pub const DEFAULT_MODEL: &str = "nomic-embed-text";
pub const DEFAULT_URL: &str = "http://localhost:11434";

/// A local embedding backend.
pub struct OllamaEmbedder {
    client: reqwest::blocking::Client,
    url: String,
    model: String,
    dim: Option<usize>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    embedding: Vec<f32>,
}

impl OllamaEmbedder {
    /// Build an embedder from the environment, falling back to the defaults.
    pub fn from_env() -> Result<Self, RagError> {
        let url = std::env::var("THOTBOOK_EMBED_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
        let model =
            std::env::var("THOTBOOK_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        Self::new(&url, &model)
    }

    pub fn new(url: &str, model: &str) -> Result<Self, RagError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| RagError::Embedding(format!("building HTTP client: {e}")))?;
        Ok(Self {
            client,
            url: url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            dim: None,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Dimension of this model's output, discovered on first use.
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    /// Embed one string.
    ///
    /// Rejects an empty vector rather than returning it: a zero-dimension
    /// "embedding" indexes silently and makes every subsequent search return
    /// nothing, which is far harder to diagnose than a failure here. Generative
    /// models served by Ollama answer the embed endpoint with an empty array,
    /// so this is the check that catches a mis-set model name.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, RagError> {
        // Safety backstop only: chunks should already fit the embedding
        // model's context (all-minilm = 256 subword tokens). BERT-style
        // tokenizers produce more tokens per word than whitespace splitting,
        // so this clamp is deliberately tight relative to the chunker's
        // whitespace-token count. Oversized input makes the server drop the
        // connection rather than return a clean error.
        const MAX_CHARS: usize = 800;
        let clamped: String = if text.chars().count() > MAX_CHARS {
            text.chars().take(MAX_CHARS).collect()
        } else {
            text.to_string()
        };

        let body = serde_json::json!({ "model": self.model, "input": clamped });

        // One retry: a dropped connection under load is transient, and losing
        // an entire indexing run to it is not worth the simplicity.
        let mut last_err = String::new();
        let mut resp = None;
        for attempt in 0..5 {
            match self
                .client
                .post(format!("{}/api/embed", self.url))
                .json(&body)
                .send()
            {
                Ok(r) => {
                    resp = Some(r);
                    break;
                }
                Err(e) => {
                    last_err = e.to_string();
                    std::thread::sleep(std::time::Duration::from_millis(500 * (attempt + 1)));
                }
            }
        }
        let resp = resp.ok_or_else(|| {
            RagError::Embedding(format!("POST /api/embed failed after 5 attempts: {last_err}"))
        })?;

        let status = resp.status();
        let parsed: EmbedResponse = resp
            .json()
            .map_err(|e| RagError::Embedding(format!("decoding response ({status}): {e}")))?;

        let mut v = parsed
            .embeddings
            .into_iter()
            .next()
            .unwrap_or(parsed.embedding);

        if v.is_empty() {
            return Err(RagError::Embedding(format!(
                "model '{}' returned a zero-length embedding — it is probably a \
                 generative model rather than an embedding model. Try \
                 THOTBOOK_EMBED_MODEL=nomic-embed-text",
                self.model
            )));
        }

        // L2 normalise so cosine similarity is a dot product.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }

        match self.dim {
            None => self.dim = Some(v.len()),
            Some(d) if d != v.len() => {
                return Err(RagError::Embedding(format!(
                    "dimension changed mid-run: {d} then {}",
                    v.len()
                )))
            }
            _ => {}
        }

        Ok(v)
    }

    /// Check the backend is reachable and the model embeds, before a long run.
    pub fn preflight(&mut self) -> Result<usize, RagError> {
        let v = self.embed("preflight")?;
        Ok(v.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let e = OllamaEmbedder::new(DEFAULT_URL, DEFAULT_MODEL).unwrap();
        assert_eq!(e.model(), "nomic-embed-text");
        assert!(e.dim().is_none(), "dimension is unknown until first use");
    }

    #[test]
    fn trailing_slash_is_trimmed() {
        let e = OllamaEmbedder::new("http://localhost:11434/", "m").unwrap();
        assert_eq!(e.url, "http://localhost:11434");
    }
}
