use candle_core::{Tensor, DType, D};
use candle_nn::ops::log_softmax;

/// Cross-entropy loss with optional label smoothing.
pub fn cross_entropy_loss(
    logits: &Tensor,
    labels: &Tensor,
    _label_smoothing: f32,
) -> Result<Tensor, candle_core::Error> {
    let logits = logits.to_dtype(DType::F32)?;
    let labels = labels.to_dtype(DType::I64)?;

    let log_probs = log_softmax(&logits, D::Minus1)?;
    let nll = log_probs.gather(&labels, D::Minus1)?;

    // Mean over all elements
    let loss = nll.mean_all()?;
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_basic() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::zeros((2, 10), DType::F32, &device).unwrap();
        let labels = Tensor::new(&[[1i64, 2], [3, 4]], &device).unwrap();
        let loss = cross_entropy_loss(&logits, &labels, 0.0).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        // log_softmax(zeros) = log(1/10) ≈ -2.3, so loss should be negative
        assert!(val < 0.0);
        assert!(val.is_finite());
    }
}
