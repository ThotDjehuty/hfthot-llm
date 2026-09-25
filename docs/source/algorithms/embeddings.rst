Embedding Algorithms
====================

The embedding module extracts dense vector representations from text
using the Qwen3-8B hidden states. These embeddings power the RAG
retrieval system and enable semantic search over the knowledge base.

Mean Pooling
------------

We extract embeddings by mean-pooling the last hidden layer:

.. math::

   \mathbf{e} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{h}_t^{(L)}

where :math:`\mathbf{h}_t^{(L)} \in \mathbb{R}^{d}` is the hidden state
at position :math:`t` in the final layer :math:`L`.

**Masked mean pooling:**

To handle variable-length sequences with padding:

.. math::

   \mathbf{e} = \frac{\sum_{t=1}^{T} m_t \cdot \mathbf{h}_t^{(L)}}{\sum_{t=1}^{T} m_t}

where :math:`m_t \in \{0, 1\}` is the attention mask.

.. code-block:: rust

   pub fn mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
       // hidden_states: [batch, seq_len, hidden_dim]
       // attention_mask: [batch, seq_len]

       let mask = attention_mask.unsqueeze(D::Minus1)?;  // [batch, seq_len, 1]
       let masked = (hidden_states * &mask)?;
       let summed = masked.sum(1)?;                      // [batch, hidden_dim]
       let counts = mask.sum(1)?.clamp(1.0, f64::MAX)?;
       summed.broadcast_div(&counts)
   }

Dimensionality Reduction
------------------------

The raw hidden dimension (4096) is too large for efficient retrieval.
We project to 384 dimensions using a learned linear layer:

.. math::

   \mathbf{e}_{384} = W_p \cdot \mathbf{e}_{4096} + \mathbf{b}_p

where :math:`W_p \in \mathbb{R}^{384 \times 4096}`.

**Projection initialization:**

We use PCA on a sample of embeddings to initialize :math:`W_p`:

.. math::

   W_p = V_{:384}^\top

where :math:`V` are the right singular vectors of the centered
embedding matrix.

Normalization
-------------

Embeddings are L2-normalized for cosine similarity:

.. math::

   \hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}

After normalization:

.. math::

   \cos(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2)
   = \hat{\mathbf{e}}_1^\top \hat{\mathbf{e}}_2
   = \langle \hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2 \rangle

This allows using dot product for similarity, which is faster than
computing full cosine similarity.

Embedding Quantization
----------------------

For large-scale retrieval (>100k documents), we quantize embeddings
to int8:

.. math::

   e_q = \text{round}\left(\frac{e - \min}{\max - \min} \times 255\right) - 128

**Memory savings:**

.. list-table::
   :header-rows: 1

   * - Representation
     - Bytes/embedding
     - 100k docs
   * - FP32
     - 1536
     - 150 MB
   * - FP16
     - 768
     - 75 MB
   * - **INT8**
     - **384**
     - **37 MB**

**Approximate similarity:**

.. math::

   \text{sim}(e_{q,1}, e_{q,2})
   \approx \frac{1}{255^2} \sum_i (e_{q,1,i} + 128)(e_{q,2,i} + 128)

Using SIMD (AVX-512), we compute 64 int8 dot products per cycle.

Batch Embedding
---------------

The ``/v1/embeddings`` endpoint supports batch embedding:

.. code-block:: json

   {
     "model": "qwen3-8b",
     "input": [
       "First document text",
       "Second document text",
       "Third document text"
     ]
   }

Response:

.. code-block:: json

   {
     "object": "list",
     "data": [
       {"embedding": [0.023, -0.041, ...], "index": 0},
       {"embedding": [-0.012, 0.089, ...], "index": 1},
       {"embedding": [0.056, 0.003, ...], "index": 2}
     ],
     "model": "qwen3-8b",
     "usage": {"prompt_tokens": 42, "total_tokens": 42}
   }

**Throughput:**

.. math::

   \text{Throughput} \approx 150\text{-}200 \text{ embeddings/sec}

for average document length of 256 tokens on M1 Max.

Semantic Similarity
-------------------

Given embeddings, compute pairwise similarities:

.. code-block:: rust

   pub fn cosine_similarity_matrix(a: &Tensor, b: &Tensor) -> Result<Tensor> {
       // a: [n, dim], b: [m, dim]
       // Returns [n, m] similarity matrix

       let a_norm = a.normalize(1.0, D::Minus1)?;
       let b_norm = b.normalize(1.0, D::Minus1)?;
       a_norm.matmul(&b_norm.t()?)
   }

For finding the top-k most similar documents:

.. code-block:: rust

   pub fn top_k_similar(query: &Tensor, corpus: &Tensor, k: usize) -> Vec<(usize, f32)> {
       let sims = cosine_similarity_matrix(query, corpus)?;
       let (values, indices) = sims.topk(k)?;
       // ...
   }

Embedding Cache
---------------

To avoid recomputing embeddings, we cache them in the lakehouse:

.. code-block:: sql

   CREATE TABLE embeddings (
       doc_id STRING PRIMARY KEY,
       embedding BINARY,  -- 384 x f32 = 1536 bytes
       model_version STRING,
       created_at TIMESTAMP
   )

Cache invalidation triggers on:

1. Document content change (hash mismatch)
2. Model version update
3. Manual cache clear

Integration with HNSW
---------------------

Embeddings feed into the HNSW index for approximate nearest neighbor
search. See :doc:`rag` for details on the retrieval pipeline.
