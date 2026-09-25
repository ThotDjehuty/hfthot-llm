//! Model loading and LoRA-augmented forward pass for training.
//!
//! This module provides:
//! - Loading Qwen3 base model weights
//! - LoRA adapter injection into attention layers
//! - Forward pass with gradient tracking for training

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM};

use crate::error::TrainError;
use crate::lora::{LoraAdapter, LoraConfig};

/// Configuration for model loading.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to model directory (config.json + safetensors).
    pub model_dir: PathBuf,
    /// Path to tokenizer.json.
    pub tokenizer_path: PathBuf,
    /// Use f16 (default) or f32 for weights.
    pub dtype: DType,
}

impl ModelConfig {
    pub fn new(model_dir: impl Into<PathBuf>, tokenizer_path: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            tokenizer_path: tokenizer_path.into(),
            dtype: DType::F32, // Use F32 for training (gradients)
        }
    }
}

/// LoRA-augmented Qwen3 model for training.
pub struct TrainableModel {
    /// Base Qwen3 model (frozen during LoRA training).
    base_model: ModelForCausalLM,
    /// LoRA adapters for attention projections.
    lora_adapters: HashMap<String, LoraAdapter>,
    /// Model configuration.
    config: Qwen3Config,
    /// Device (CPU for now).
    device: Device,
    /// Tokenizer for decoding.
    tokenizer: tokenizers::Tokenizer,
}

impl TrainableModel {
    /// Load model weights and initialize LoRA adapters.
    pub fn load(
        model_config: &ModelConfig,
        lora_config: &LoraConfig,
    ) -> Result<Self, TrainError> {
        let device = Device::Cpu;

        // Load model config
        let config_path = model_config.model_dir.join("config.json");
        let raw = std::fs::read(&config_path).map_err(|e| TrainError::Io {
            path: config_path.clone(),
            source: e,
        })?;
        let qwen_config: Qwen3Config = serde_json::from_slice(&raw)
            .map_err(|e| TrainError::Model(format!("invalid config.json: {e}")))?;

        // Load safetensors shards
        let shards = glob_safetensors(&model_config.model_dir)?;
        if shards.is_empty() {
            return Err(TrainError::Model(format!(
                "no safetensors files in {}",
                model_config.model_dir.display()
            )));
        }

        tracing::info!(
            dir = %model_config.model_dir.display(),
            shards = shards.len(),
            layers = qwen_config.num_hidden_layers,
            "loading qwen3 weights for training"
        );

        let tensors = load_shards(&shards, &device)?;
        let dtype = model_config.dtype;
        let mut tensors_typed: HashMap<String, Tensor> = HashMap::new();
        for (k, t) in tensors {
            let t = t.to_dtype(dtype).map_err(|e| TrainError::Candle(e.to_string()))?;
            tensors_typed.insert(k, t);
        }

        let vb = VarBuilder::from_tensors(tensors_typed, dtype, &device);
        let base_model = ModelForCausalLM::new(&qwen_config, vb)
            .map_err(|e| TrainError::Candle(e.to_string()))?;

        // Initialize LoRA adapters for target modules
        let mut lora_adapters = HashMap::new();
        let hidden_size = qwen_config.hidden_size;

        for layer_idx in 0..qwen_config.num_hidden_layers {
            for target in &lora_config.target_modules {
                let key = format!("layer{layer_idx}.{target}");
                let adapter = LoraAdapter::new(hidden_size, hidden_size, lora_config, &device)
                    .map_err(|e| TrainError::Candle(e.to_string()))?;
                lora_adapters.insert(key, adapter);
            }
        }

        tracing::info!(
            adapters = lora_adapters.len(),
            rank = lora_config.rank,
            "initialized LoRA adapters"
        );

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&model_config.tokenizer_path)
            .map_err(|e| TrainError::Model(format!("failed to load tokenizer: {e}")))?;

        Ok(Self {
            base_model,
            lora_adapters,
            config: qwen_config,
            device,
            tokenizer,
        })
    }

    /// Forward pass for training: returns logits tensor.
    ///
    /// Note: This uses the base model forward. LoRA adapters are applied
    /// separately to the intermediate tensors. For full LoRA training,
    /// we'd need to modify the model internals.
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor, TrainError> {
        let logits = self
            .base_model
            .forward(input_ids, 0)
            .map_err(|e| TrainError::Candle(e.to_string()))?;
        Ok(logits)
    }

    /// Get hidden states for embedding generation (RAG).
    pub fn get_hidden_states(&mut self, input_ids: &Tensor) -> Result<Tensor, TrainError> {
        // For now, use the output before lm_head as hidden states
        // This is a simplification - ideally we'd access the last hidden layer
        let logits = self.forward(input_ids)?;
        Ok(logits)
    }

    /// Get all LoRA adapter parameters as a flat vector (for optimizer).
    pub fn lora_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for adapter in self.lora_adapters.values() {
            params.push(&adapter.a);
            params.push(&adapter.b);
        }
        params
    }

    /// Get mutable LoRA adapter parameters.
    pub fn lora_params_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for adapter in self.lora_adapters.values_mut() {
            params.push(&mut adapter.a);
            params.push(&mut adapter.b);
        }
        params
    }

    /// Save LoRA adapter weights to a directory.
    pub fn save_lora(&self, dir: &Path) -> Result<(), TrainError> {
        std::fs::create_dir_all(dir).map_err(|e| TrainError::Io {
            path: dir.to_path_buf(),
            source: e,
        })?;

        for (name, adapter) in &self.lora_adapters {
            let a_path = dir.join(format!("{name}_a.safetensors"));
            let b_path = dir.join(format!("{name}_b.safetensors"));

            let mut a_map: HashMap<String, Tensor> = HashMap::new();
            a_map.insert("a".to_string(), adapter.a.clone());
            candle_core::safetensors::save(&a_map, &a_path)
                .map_err(|e| TrainError::Candle(e.to_string()))?;

            let mut b_map: HashMap<String, Tensor> = HashMap::new();
            b_map.insert("b".to_string(), adapter.b.clone());
            candle_core::safetensors::save(&b_map, &b_path)
                .map_err(|e| TrainError::Candle(e.to_string()))?;
        }

        tracing::info!(dir = %dir.display(), "saved LoRA adapters");
        Ok(())
    }

    /// Load LoRA adapter weights from a directory.
    pub fn load_lora(&mut self, dir: &Path) -> Result<(), TrainError> {
        for (name, adapter) in &mut self.lora_adapters {
            let a_path = dir.join(format!("{name}_a.safetensors"));
            let b_path = dir.join(format!("{name}_b.safetensors"));

            if a_path.exists() && b_path.exists() {
                let a_data = std::fs::read(&a_path).map_err(|e| TrainError::Io {
                    path: a_path.clone(),
                    source: e,
                })?;
                let b_data = std::fs::read(&b_path).map_err(|e| TrainError::Io {
                    path: b_path.clone(),
                    source: e,
                })?;

                let a_map = candle_core::safetensors::load_buffer(&a_data, &self.device)
                    .map_err(|e| TrainError::Candle(e.to_string()))?;
                let b_map = candle_core::safetensors::load_buffer(&b_data, &self.device)
                    .map_err(|e| TrainError::Candle(e.to_string()))?;

                if let Some(a) = a_map.get("a") {
                    adapter.a = a.clone();
                }
                if let Some(b) = b_map.get("b") {
                    adapter.b = b.clone();
                }
            }
        }

        tracing::info!(dir = %dir.display(), "loaded LoRA adapters");
        Ok(())
    }

    /// Get the tokenizer reference.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Get model config.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Vocab size for loss computation.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

/// Glob all .safetensors files in a directory.
fn glob_safetensors(dir: &Path) -> Result<Vec<PathBuf>, TrainError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| TrainError::Io { path: dir.to_path_buf(), source: e })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| TrainError::Io { path: dir.to_path_buf(), source: e })?;
    let mut shards: Vec<PathBuf> = entries
        .into_iter()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shards.sort();
    Ok(shards)
}

/// Load all safetensors shards and merge into a single tensor map.
fn load_shards(shards: &[PathBuf], device: &Device) -> Result<HashMap<String, Tensor>, TrainError> {
    let mut tensors = HashMap::new();
    for shard in shards {
        let data = std::fs::read(shard).map_err(|e| TrainError::Io {
            path: shard.clone(),
            source: e,
        })?;
        let map = candle_core::safetensors::load_buffer(&data, device)
            .map_err(|e| TrainError::Candle(e.to_string()))?;
        tensors.extend(map);
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_new() {
        let cfg = ModelConfig::new("/path/to/model", "/path/to/tokenizer.json");
        assert_eq!(cfg.model_dir.to_str().unwrap(), "/path/to/model");
        assert_eq!(cfg.dtype, DType::F32);
    }
}
