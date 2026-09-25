Inference Algorithms
====================

The ``llm-serve`` crate implements **CPU-optimized autoregressive inference**
for Qwen3-8B using the Candle framework. This document details the
algorithms that make real-time inference feasible on consumer hardware.

Autoregressive Generation
-------------------------

Text generation follows the autoregressive factorization:

.. math::

   p(x_{1:T})
   = \prod_{t=1}^{T} p(x_t \mid x_{1:t-1})

At each step, we compute:

1. **Forward pass**: :math:`\text{logits}_t = f_\theta(x_{1:t})`
2. **Sampling**: :math:`x_{t+1} \sim p(\cdot \mid \text{logits}_t)`
3. **KV-cache update**: Store :math:`K_t, V_t` for future steps

Key-Value Cache
---------------

The KV-cache eliminates redundant computation by storing attention keys
and values from previous tokens:

.. math::

   \text{Attention}(Q_t, K_{1:t}, V_{1:t})
   = \text{softmax}\left(\frac{Q_t K_{1:t}^\top}{\sqrt{d_k}}\right) V_{1:t}

**Memory layout:**

.. code-block:: rust

   pub struct KvCache {
       // [layers, 2, batch, heads, seq_len, head_dim]
       cache: Vec<(Tensor, Tensor)>,
       seq_len: usize,
   }

   impl KvCache {
       pub fn append(&mut self, layer: usize, k: Tensor, v: Tensor) {
           let (k_cache, v_cache) = &mut self.cache[layer];
           *k_cache = Tensor::cat(&[k_cache, &k], D::Minus2)?;
           *v_cache = Tensor::cat(&[v_cache, &v], D::Minus2)?;
           self.seq_len += 1;
       }
   }

**Memory per token:**

.. math::

   \text{Memory}(\text{KV})
   = 2 \cdot L \cdot H \cdot d_h \cdot \text{dtype\_size}
   = 2 \cdot 32 \cdot 32 \cdot 128 \cdot 2
   = 512 \text{ KB/token}

For 2048 tokens: ~1 GB KV-cache memory.

Quantization (Q4_K)
-------------------

We use Q4_K quantization for model weights, reducing memory by 4×:

.. math::

   w_q = \text{round}\left(\frac{w - \min(w)}{\Delta}\right),
   \quad
   \Delta = \frac{\max(w) - \min(w)}{15}

**Dequantization:**

.. math::

   \hat{w} = w_q \cdot \Delta + \min(w)

**Block quantization** (block size 32):

Each block of 32 weights shares a scale and zero-point, reducing
quantization error while maintaining compression.

.. code-block:: rust

   pub struct Q4KBlock {
       // 32 weights packed into 16 bytes (4 bits each)
       data: [u8; 16],
       scale: f16,
       zero: f16,
   }

   impl Q4KBlock {
       pub fn dequantize(&self, output: &mut [f32]) {
           for i in 0..32 {
               let q = if i % 2 == 0 {
                   self.data[i / 2] & 0x0F
               } else {
                   self.data[i / 2] >> 4
               };
               output[i] = f16::to_f32(self.scale) * (q as f32)
                   + f16::to_f32(self.zero);
           }
       }
   }

**Memory comparison:**

.. list-table::
   :header-rows: 1

   * - Precision
     - Model Size
     - Inference Memory
   * - FP32
     - 32 GB
     - 40+ GB
   * - FP16
     - 16 GB
     - 22 GB
   * - Q8_0
     - 8 GB
     - 12 GB
   * - **Q4_K**
     - **4.5 GB**
     - **6 GB**

Sampling Strategies
-------------------

**Temperature scaling:**

.. math::

   p_i = \frac{\exp(\text{logit}_i / \tau)}{\sum_j \exp(\text{logit}_j / \tau)}

- :math:`\tau = 0`: Greedy (argmax)
- :math:`\tau = 1`: Standard softmax
- :math:`\tau > 1`: More uniform (creative)
- :math:`\tau < 1`: Sharper (deterministic)

**Top-p (nucleus) sampling:**

Keep only tokens whose cumulative probability exceeds :math:`p`:

.. math::

   V_p = \min\{V' \subseteq V : \sum_{v \in V'} p(v) \geq p\}

.. code-block:: rust

   fn top_p_sample(logits: &[f32], p: f32, temperature: f32) -> usize {
       let mut probs: Vec<(usize, f32)> = logits
           .iter()
           .enumerate()
           .map(|(i, &l)| (i, (l / temperature).exp()))
           .collect();

       // Sort descending
       probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

       // Normalize
       let sum: f32 = probs.iter().map(|(_, p)| p).sum();
       let mut cumsum = 0.0;

       for &(idx, prob) in &probs {
           cumsum += prob / sum;
           if cumsum >= p {
               return idx;
           }
       }
       probs[0].0
   }

**Repetition penalty:**

Discourage repeating recent tokens:

.. math::

   \text{logit}'_i =
   \begin{cases}
   \text{logit}_i / \alpha & i \in \text{recent} \\
   \text{logit}_i & \text{otherwise}
   \end{cases}

where :math:`\alpha > 1` is the penalty factor.

Rotary Position Embeddings (RoPE)
---------------------------------

Qwen3 uses RoPE for position encoding:

.. math::

   \text{RoPE}(x, m) = R_m x

where :math:`R_m` is a block-diagonal rotation matrix:

.. math::

   R_m = \text{diag}(R_{\theta_1}^m, R_{\theta_2}^m, \ldots, R_{\theta_{d/2}}^m)

with :math:`R_\theta^m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}`.

**Frequency schedule:**

.. math::

   \theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1

**Implementation:**

.. code-block:: rust

   fn apply_rope(x: &Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> Result<Tensor> {
       let (x_real, x_imag) = x.chunk(2, D::Minus1)?;
       let rotated_real = (&x_real * freqs_cos)? - (&x_imag * freqs_sin)?;
       let rotated_imag = (&x_real * freqs_sin)? + (&x_imag * freqs_cos)?;
       Tensor::cat(&[rotated_real, rotated_imag], D::Minus1)
   }

OpenAI-Compatible API
---------------------

The HTTP server exposes three endpoints:

**POST /v1/chat/completions**

.. code-block:: json

   {
     "model": "qwen3-8b",
     "messages": [
       {"role": "user", "content": "Explain LoRA in one line."}
     ],
     "temperature": 0.7,
     "max_tokens": 128
   }

**POST /v1/embeddings**

.. code-block:: json

   {
     "model": "qwen3-8b",
     "input": ["Hello world", "Goodbye world"]
   }

**GET /health**

Returns ``{"status": "ok", "model": "qwen3-8b"}``.

Performance Tuning
------------------

**BLAS backend:**

On Apple Silicon, we use Accelerate framework (vecLib) for matrix
operations. On x86-64, OpenBLAS with AVX-512 is preferred.

.. code-block:: toml

   # Cargo.toml feature flags
   [features]
   accelerate = ["candle-core/accelerate"]
   mkl = ["candle-core/mkl"]

**Thread pool sizing:**

.. code-block:: rust

   // Use physical cores for compute, leave 2 for I/O
   let num_threads = num_cpus::get_physical().saturating_sub(2).max(1);
   rayon::ThreadPoolBuilder::new()
       .num_threads(num_threads)
       .build_global()
       .unwrap();

**Batch inference:**

For embedding multiple texts, batch them for better throughput:

.. math::

   \text{Throughput}(\text{batch}=8)
   \approx 4 \times \text{Throughput}(\text{batch}=1)

due to better BLAS utilization and reduced Python/Rust FFI overhead.
