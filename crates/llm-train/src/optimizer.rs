use std::collections::HashMap;

use candle_core::{Tensor, DType};

/// AdamW optimizer with weight decay.
pub struct AdamWOptimizer {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step_count: usize,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
}

impl AdamWOptimizer {
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Update a single parameter using AdamW.
    pub fn update_param(
        &mut self,
        name: &str,
        param: &mut Tensor,
        grad: &Tensor,
    ) -> Result<(), candle_core::Error> {
        self.step_count += 1;

        let grad = grad.to_dtype(DType::F32)?;
        let param_f32 = param.to_dtype(DType::F32)?;

        // Initialize moments
        if !self.m.contains_key(name) {
            self.m.insert(name.to_string(), Tensor::zeros_like(&param_f32)?);
            self.v.insert(name.to_string(), Tensor::zeros_like(&param_f32)?);
        }

        let m = self.m.get_mut(name).unwrap();
        let v = self.v.get_mut(name).unwrap();

        let g = grad.clone();

        // m_t = beta1 * m_{t-1} + (1 - beta1) * g
        *m = m.affine(self.beta1 as f64, 0.0)?;
        let g_1mb1 = g.affine((1.0 - self.beta1) as f64, 0.0)?;
        *m = (m.clone() + g_1mb1)?;

        // v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
        let g_sq = g.powf(2.0)?;
        *v = v.affine(self.beta2 as f64, 0.0)?;
        let g_sq_1mb2 = g_sq.affine((1.0 - self.beta2) as f64, 0.0)?;
        *v = (v.clone() + g_sq_1mb2)?;

        // Bias correction
        let b1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let b2 = 1.0 - self.beta2.powi(self.step_count as i32);
        let m_hat = m.affine(1.0 / b1 as f64, 0.0)?;
        let v_hat = v.affine(1.0 / b2 as f64, 0.0)?;

        // Update: param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
        //
        // The weight-decay term is applied to the parameter itself rather than
        // folded into the gradient — this is exactly what makes it *decoupled*
        // (Loshchilov & Hutter, 2019) and what distinguishes AdamW from
        // Adam+L2: the decay is not rescaled by the adaptive denominator
        // sqrt(v_hat). See docs/source/algorithms/training.rst.
        let denom = (v_hat.sqrt()?.affine(1.0, self.eps as f64))?;
        let update = m_hat.broadcast_div(&denom)?;
        let decay = param_f32.affine(self.weight_decay as f64, 0.0)?;
        let update = (update + decay)?.affine(self.lr as f64, 0.0)?;
        *param = (param_f32 - update)?.to_dtype(param.dtype())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_decay_is_applied() {
        // With a zero gradient, plain Adam would leave the parameter
        // untouched; AdamW must still shrink it by lr * wd * param.
        let device = candle_core::Device::Cpu;
        let mut opt = AdamWOptimizer::new(0.1, 0.5);
        let mut param = Tensor::ones((4,), DType::F32, &device).unwrap();
        let grad = Tensor::zeros((4,), DType::F32, &device).unwrap();
        opt.update_param("p", &mut param, &grad).unwrap();
        let v = param.to_vec1::<f32>().unwrap()[0];
        // expected: 1 - 0.1 * (0 + 0.5 * 1) = 0.95
        assert!((v - 0.95).abs() < 1e-5, "decay not applied: got {v}");
    }

    #[test]
    fn optimizer_step() {
        let mut opt = AdamWOptimizer::new(0.001, 0.01);
        assert_eq!(opt.step_count, 0);

        let device = candle_core::Device::Cpu;
        let mut param = Tensor::zeros((10,), DType::F32, &device).unwrap();
        let grad = Tensor::ones((10,), DType::F32, &device).unwrap();
        opt.update_param("test", &mut param, &grad).unwrap();
        assert_eq!(opt.step_count, 1);
    }
}
