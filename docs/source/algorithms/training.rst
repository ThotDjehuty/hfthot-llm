Training Algorithms
===================

The ``llm-train`` crate implements **CPU-only LoRA fine-tuning** for
large language models. This document details the mathematical foundations
and implementation strategies that make GPU-free training feasible.

Low-Rank Adaptation (LoRA)
--------------------------

Instead of fine-tuning all :math:`d \times d` weight matrices, LoRA
decomposes the update into two low-rank factors:

.. math::

   W' = W_0 + \Delta W = W_0 + B A

where :math:`A \in \mathbb{R}^{r \times d}`, :math:`B \in \mathbb{R}^{d \times r}`,
and :math:`r \ll d` is the rank (typically 8–64).

**Memory savings:**

.. math::

   \frac{\text{LoRA parameters}}{\text{Full parameters}}
   = \frac{2 r d}{d^2}
   = \frac{2r}{d}
   \approx 0.4\%
   \quad \text{for } r=16, d=4096

**Implementation in llm-train:**

.. code-block:: rust

   pub struct LoraAdapter {
       pub lora_a: Tensor,  // [r, in_dim]
       pub lora_b: Tensor,  // [out_dim, r]
       pub scale: f32,      // α/r scaling factor
   }

   impl LoraAdapter {
       pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
           // W'x = W₀x + scale · B(Ax)
           let ax = x.matmul(&self.lora_a.t())?;
           let bax = ax.matmul(&self.lora_b.t())?;
           Ok(bax * self.scale)
       }
   }

Cross-Entropy Loss with Label Smoothing
---------------------------------------

The training objective is cross-entropy loss with optional label smoothing:

.. math::

   \mathcal{L}
   = -\frac{1}{N} \sum_{i=1}^{N}
     \sum_{c=1}^{V}
     \tilde{y}_{i,c} \log p_{i,c}

where :math:`\tilde{y}` is the smoothed label distribution:

.. math::

   \tilde{y}_{i,c}
   = (1 - \epsilon) \cdot \mathbf{1}[c = y_i]
     + \frac{\epsilon}{V}

**Masked loss for padding:**

Only non-padding tokens (``label != -100``) contribute to the loss:

.. code-block:: rust

   fn cross_entropy_loss(logits: &Tensor, labels: &Tensor) -> Result<f32> {
       let log_probs = log_softmax(logits, D::Minus1)?;
       let flat_labels = labels.reshape(((),))?;
       let flat_log_probs = log_probs.reshape(((), vocab_size))?;

       // Gather log probs at label positions
       let selected = flat_log_probs.gather(&flat_labels, 1)?;

       // Mask: ignore label == -100
       let mask = flat_labels.ge(&0i64)?.to_dtype(DType::F32)?;
       let masked_nll = (selected.neg()? * &mask)?;

       // Mean over valid tokens
       Ok((masked_nll.sum_all()? / mask.sum_all()?.clamp(1.0, f64::MAX)?).to_scalar()?)
   }

AdamW Optimizer
----------------

LoRA parameters :math:`(A, B)` are updated with **AdamW** — Adam with
decoupled weight decay (Loshchilov & Hutter, 2019). At step :math:`t`, for
gradient :math:`g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})`:

.. math::

   m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
   v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
   \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
   \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
   \theta_t &= \theta_{t-1} - \eta_t \left(
     \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda\, \theta_{t-1}
   \right)

with :math:`\beta_1 = 0.9`, :math:`\beta_2 = 0.999`,
:math:`\varepsilon = 10^{-8}`, and decoupled weight decay
:math:`\lambda = 0.01` applied directly to :math:`\theta_{t-1}` rather
than folded into :math:`g_t` — this is what keeps weight decay from
being rescaled by :math:`\hat{v}_t`, the fix that motivated AdamW over
plain Adam+L2. :math:`\eta_t` is the cosine-annealed rate below. This is
the discrete-time update whose continuous-time (gradient-flow) limit is
derived in :doc:`variational_calculus`.

Cosine Learning Rate Schedule
-----------------------------

We use cosine annealing with linear warmup:

.. math::

   \eta_t =
   \begin{cases}
   \eta_{\max} \cdot \frac{t}{T_w} & t < T_w \\
   \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
     \left(1 + \cos\left(\frac{t - T_w}{T - T_w}\pi\right)\right) & t \geq T_w
   \end{cases}

where :math:`T_w` is the warmup steps (typically 10% of total), :math:`T` is
total steps, :math:`\eta_{\max} = 2 \times 10^{-4}`, :math:`\eta_{\min} = 2 \times 10^{-5}`.

.. code-block:: rust

   pub struct CosineScheduler {
       warmup_steps: usize,
       total_steps: usize,
       lr_max: f32,
       lr_min: f32,
   }

   impl CosineScheduler {
       pub fn lr_at(&self, step: usize) -> f32 {
           if step < self.warmup_steps {
               self.lr_max * (step as f32 / self.warmup_steps as f32)
           } else {
               let progress = (step - self.warmup_steps) as f32
                   / (self.total_steps - self.warmup_steps) as f32;
               self.lr_min + 0.5 * (self.lr_max - self.lr_min)
                   * (1.0 + (progress * std::f32::consts::PI).cos())
           }
       }
   }

Gradient Checkpointing
----------------------

To reduce memory usage during backward pass, we checkpoint intermediate
activations at layer boundaries:

.. math::

   \text{Memory}(\text{naive})
   = \mathcal{O}(L \cdot B \cdot T \cdot d)

.. math::

   \text{Memory}(\text{checkpointed})
   = \mathcal{O}(\sqrt{L} \cdot B \cdot T \cdot d)

where :math:`L` = layers, :math:`B` = batch size, :math:`T` = sequence length.

For Qwen3-8B (L=32, d=4096, T=2048):

- Naive: ~28 GB activation memory
- Checkpointed: ~5 GB activation memory

Arrow Dataset Streaming
-----------------------

The ``ArrowDataset`` reader provides memory-mapped access to tokenized
training data stored in Apache Arrow IPC format:

.. code-block:: rust

   pub struct ArrowDataset {
       mmap: Mmap,
       schema: Arc<Schema>,
       current_shard: usize,
   }

   impl ArrowDataset {
       pub fn next_batch(&mut self, batch_size: usize) -> Option<Vec<Example>> {
           // Zero-copy read from memory-mapped file
           let batch = read_record_batch(&self.mmap[offset..], &self.schema)?;
           let input_ids = batch.column(0).as_i32_slice();
           let labels = batch.column(1).as_i32_slice();
           // ...
       }
   }

Benefits of Arrow IPC:

1. **Zero-copy**: Data stays in kernel page cache
2. **Columnar**: Only read needed columns
3. **Compressed**: LZ4 compression reduces I/O
4. **Streaming**: Process larger-than-RAM datasets

Training Configuration
----------------------

Default training configuration:

.. code-block:: toml

   [training]
   mode = "lora"
   lora_rank = 16
   lora_alpha = 32
   batch_size = 4
   gradient_accumulation = 8  # effective batch = 32
   epochs = 3
   learning_rate = 2e-4
   warmup_ratio = 0.1
   weight_decay = 0.01
   max_seq_len = 2048
   checkpoint_interval = 100

Convergence Analysis
--------------------

For a convex approximation of the loss landscape, LoRA with AdamW converges
at rate:

.. math::

   \mathbb{E}[\mathcal{L}(W_T) - \mathcal{L}^*]
   \leq \frac{C}{\sqrt{T}}

where :math:`C` depends on the Lipschitz constant of the gradient and the
effective learning rate.

In practice, we observe:

- **Epoch 1**: Loss drops from ~3.5 to ~1.8
- **Epoch 2**: Loss stabilizes around ~1.2–1.5
- **Epoch 3**: Marginal improvement, ~1.0–1.2

Perplexity tracks closely: :math:`\text{PPL} = e^{\mathcal{L}}`.
