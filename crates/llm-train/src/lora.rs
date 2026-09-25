use candle_core::{Tensor, DType};


/// LoRA adapter layer configuration.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank matrices.
    pub rank: usize,
    /// Scaling factor (alpha / rank).
    pub alpha: usize,
    /// Dropout probability for LoRA layers.
    pub dropout: f32,
    /// Target modules to apply LoRA to (e.g., ["q_proj", "v_proj"]).
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
        }
    }
}

/// LoRA adapter weights for a single layer.
pub struct LoraAdapter {
    /// Low-rank matrix A (in_features × rank).
    pub a: Tensor,
    /// Low-rank matrix B (rank × out_features).
    pub b: Tensor,
    /// Scaling factor.
    pub scale: f32,
}

impl LoraAdapter {
    /// Create a new LoRA adapter with random initialization.
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: &LoraConfig,
        device: &candle_core::Device,
    ) -> Result<Self, candle_core::Error> {
        let rank = config.rank;
        let scale = config.alpha as f32 / rank as f32;

        // Initialize A with Kaiming uniform, B with zeros
        let a = Tensor::randn(0.0f32, 1.0, (in_features, rank), device)?;
        let b = Tensor::zeros((rank, out_features), DType::F32, device)?;

        Ok(Self { a, b, scale })
    }

    /// Apply LoRA: output = input @ A @ B * scale
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let hidden = input.matmul(&self.a)?;
        let adapted = hidden.matmul(&self.b)?;
        adapted.affine(self.scale as f64, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_output_shape() {
        let device = candle_core::Device::Cpu;
        let config = LoraConfig::default();
        let adapter = LoraAdapter::new(64, 32, &config, &device).unwrap();

        let input = Tensor::zeros((10, 64), DType::F32, &device).unwrap();
        let output = adapter.forward(&input).unwrap();
        assert_eq!(output.dims(), &[10, 32]);
    }
}
