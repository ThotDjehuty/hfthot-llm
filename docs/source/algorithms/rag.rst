Retrieval-Augmented Generation
===============================

The ``llm-rag`` crate implements **hybrid retrieval** combining dense
vector search (HNSW) with sparse keyword matching (BM25). This enables
accurate retrieval from the personal knowledge base.

Architecture Overview
---------------------

.. code-block:: text

   Query --+--> HNSW (dense)  --+--> RRF Fusion --> Top-k Docs --> LLM
           |                     |
           +--> BM25 (sparse) --+

**Reciprocal Rank Fusion (RRF):**

.. math::

   \text{RRF}(d)
   = \sum_{r \in \{r_{\text{dense}}, r_{\text{sparse}}\}}
     \frac{1}{k + r(d)}

where :math:`r(d)` is the rank of document :math:`d` in ranking :math:`r`,
and :math:`k = 60` is a smoothing constant.

HNSW Index
----------

We use Hierarchical Navigable Small World graphs for approximate
nearest neighbor search:

**Graph structure:**

- **Layers**: :math:`L = \lfloor \log_m N \rfloor` layers
- **Connections**: Each node has :math:`M` connections per layer
- **Entry point**: Top layer has single entry node

**Search algorithm:**

.. code-block:: text

   1. Start at entry point in top layer
   2. Greedily descend to layer 0, following closest neighbors
   3. At layer 0, perform beam search with ef_search candidates
   4. Return top-k results

**Complexity:**

.. math::

   \text{Search}: O(\log N \cdot M \cdot \text{ef\_search})

.. math::

   \text{Insert}: O(\log N \cdot M^2)

**Configuration:**

.. code-block:: rust

   pub struct HnswConfig {
       pub m: usize,           // Connections per layer (default: 16)
       pub ef_construction: usize,  // Build-time beam width (default: 200)
       pub ef_search: usize,   // Query-time beam width (default: 50)
       pub max_elements: usize,
   }

**Recall-latency tradeoff:**

.. list-table::
   :header-rows: 1

   * - ef_search
     - Recall@10
     - Latency (ms)
   * - 20
     - 0.85
     - 0.5
   * - 50
     - 0.95
     - 1.2
   * - 100
     - 0.98
     - 2.5
   * - 200
     - 0.99
     - 5.0

BM25 Sparse Retrieval
---------------------

BM25 scores documents by term frequency and inverse document frequency:

.. math::

   \text{BM25}(q, d)
   = \sum_{t \in q}
     \text{IDF}(t) \cdot
     \frac{f(t, d) \cdot (k_1 + 1)}
          {f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}

where:

- :math:`f(t, d)` = term frequency of :math:`t` in :math:`d`
- :math:`\text{IDF}(t) = \log \frac{N - n(t) + 0.5}{n(t) + 0.5}`
- :math:`k_1 = 1.2`, :math:`b = 0.75` (tuning parameters)
- :math:`\text{avgdl}` = average document length

**Implementation:**

.. code-block:: rust

   pub struct Bm25Index {
       inverted_index: HashMap<String, Vec<(DocId, f32)>>,
       doc_lengths: Vec<usize>,
       avg_doc_length: f32,
       k1: f32,
       b: f32,
   }

   impl Bm25Index {
       pub fn score(&self, query: &[String], doc_id: DocId) -> f32 {
           let doc_len = self.doc_lengths[doc_id];
           let len_norm = 1.0 - self.b + self.b * (doc_len as f32 / self.avg_doc_length);

           query.iter().map(|term| {
               let idf = self.idf(term);
               let tf = self.tf(term, doc_id);
               idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
           }).sum()
       }
   }

Chunking Strategy
-----------------

Documents are split into overlapping chunks for retrieval:

.. code-block:: rust

   pub struct ChunkConfig {
       pub chunk_size: usize,      // Target tokens per chunk (default: 512)
       pub chunk_overlap: usize,   // Overlap between chunks (default: 64)
       pub min_chunk_size: usize,  // Minimum chunk size (default: 100)
   }

**Sliding window chunking:**

.. math::

   \text{chunks}(D) = \{D[i \cdot (s - o) : i \cdot (s - o) + s]\}_{i=0}^{\lceil |D|/s \rceil}

where :math:`s` = chunk_size, :math:`o` = overlap.

**Semantic chunking:**

For better coherence, we also detect section boundaries:

.. code-block:: rust

   fn semantic_chunk(text: &str) -> Vec<String> {
       // Split at headers, blank lines, or sentence boundaries
       let boundaries = detect_boundaries(text);
       merge_small_chunks(split_at(text, &boundaries))
   }

Context Window Construction
---------------------------

After retrieval, we construct the context window for the LLM:

.. code-block:: text

   <|system|>
   You are a helpful assistant with access to a knowledge base.
   Use the following context to answer the question.

   <|context|>
   [Retrieved chunk 1]
   ---
   [Retrieved chunk 2]
   ---
   [Retrieved chunk 3]

   <|user|>
   {original query}

   <|assistant|>

**Context budget:**

Given a 4096-token context window:

- System prompt: ~100 tokens
- Question: ~50 tokens
- Response budget: ~500 tokens
- **Context budget: ~3400 tokens**

We fill the context with top-ranked chunks until the budget is exhausted.

Reranking
---------

Optional cross-encoder reranking improves precision:

.. math::

   \text{score}(q, d) = \sigma(f_{\text{rerank}}([q; d]))

where :math:`f_{\text{rerank}}` is a smaller cross-encoder model and
:math:`[\cdot; \cdot]` denotes concatenation.

**Two-stage retrieval:**

1. **Recall stage**: HNSW + BM25 → 100 candidates
2. **Precision stage**: Cross-encoder → top 5

This achieves ~10% higher precision with <50ms additional latency.

Evaluation Metrics
------------------

We evaluate retrieval quality using:

**Recall@k:**

.. math::

   \text{Recall@}k = \frac{|\text{retrieved}_k \cap \text{relevant}|}{|\text{relevant}|}

**Mean Reciprocal Rank (MRR):**

.. math::

   \text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(q)}

**Normalized Discounted Cumulative Gain (nDCG):**

.. math::

   \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}

where :math:`\text{DCG@}k = \sum_{i=1}^k \frac{2^{r_i} - 1}{\log_2(i+1)}`.

**Our benchmarks on the session-log corpus:**

.. list-table::
   :header-rows: 1

   * - Method
     - Recall@5
     - MRR
     - nDCG@10
   * - BM25 only
     - 0.72
     - 0.58
     - 0.64
   * - HNSW only
     - 0.81
     - 0.65
     - 0.71
   * - **Hybrid (RRF)**
     - **0.89**
     - **0.74**
     - **0.79**

Lakehouse Integration
---------------------

The RAG index is stored in Delta Lake tables:

.. code-block:: sql

   -- Document chunks
   CREATE TABLE rag.chunks (
       chunk_id STRING PRIMARY KEY,
       doc_id STRING,
       content TEXT,
       embedding BINARY,
       metadata JSON,
       created_at TIMESTAMP
   );

   -- HNSW index serialized
   CREATE TABLE rag.hnsw_index (
       index_id STRING PRIMARY KEY,
       serialized_graph BINARY,
       config JSON,
       version INT,
       created_at TIMESTAMP
   );

Benefits:

1. **Time travel**: Query past versions of the index
2. **Incremental updates**: Only re-index changed documents
3. **Audit trail**: Track what was retrieved for each query
