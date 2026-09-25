Autonomous Orchestration
========================

The ``llm-auto`` crate implements **autonomous research orchestration**,
enabling thotbook-AmentI to decompose complex tasks, coordinate
sub-agents, and improve itself through training on its own outputs.

Multi-Agent Architecture
------------------------

The system employs six specialized agents:

.. list-table::
   :header-rows: 1

   * - Agent
     - Role
     - Task Types
   * - Researcher
     - Literature review, ArXiv search
     - ``Research``
   * - Mathematician
     - Derivations, proofs
     - ``Mathematics``
   * - Coder
     - Implementation, testing
     - ``Coding``
   * - Writer
     - LaTeX, documentation
     - ``Writing``
   * - Critic
     - Quality evaluation
     - ``Critique``
   * - Trainer
     - Fine-tuning orchestration
     - ``Training``

**Agent selection:**

.. math::

   \text{agent}(t) = \arg\max_{a \in A} \text{match}(a.\text{skills}, t.\text{type})

Task Decomposition
------------------

Complex tasks are decomposed into a directed acyclic graph (DAG) of
subtasks:

.. code-block:: text

   "Write a paper on rough volatility"
   │
   ├─▶ Research: Literature review
   │   └─▶ Research: ArXiv search "rough volatility"
   │
   ├─▶ Mathematics: Derive key equations
   │   ├─▶ Math: Volterra process definition
   │   └─▶ Math: Fractional Riccati solution
   │
   ├─▶ Writing: Generate LaTeX sections
   │   ├─▶ Write: Introduction
   │   ├─▶ Write: Main results (depends on Math)
   │   └─▶ Write: Conclusion (depends on all)
   │
   └─▶ Critique: Quality review

**Decomposition prompt:**

.. code-block:: text

   Task: {task_description}

   Decompose this task into subtasks. For each subtask, specify:
   - description: What to do
   - task_type: Research | Mathematics | Coding | Writing | Critique
   - depends_on: List of subtask indices this depends on

   Output JSON: {"subtasks": [...]}

**Topological execution:**

Subtasks are executed in topological order, respecting dependencies:

.. code-block:: rust

   pub async fn execute_dag(&mut self, dag: &TaskDag) -> Result<(), AutoError> {
       let order = dag.topological_sort()?;

       for task_id in order {
           let task = dag.get(task_id)?;

           // Wait for dependencies
           for dep_id in &task.dependencies {
               self.wait_for(dep_id).await?;
           }

           // Execute
           let agent = self.agents.find_agent(task.task_type)?;
           let result = agent.execute(&task).await?;

           self.completed.insert(task_id, result);
       }

       Ok(())
   }

Self-Improvement Loop
---------------------

The system improves through training on its own high-quality outputs:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                                                         │
   │  Execute Task ──▶ Evaluate Quality ──▶ Score ≥ 70?     │
   │       │                                    │            │
   │       │                              ┌─────┴─────┐      │
   │       │                              │    Yes    │      │
   │       │                              ▼           │      │
   │       │                        Add to Queue      │      │
   │       │                              │           │      │
   │       │                    Queue ≥ 100?          │      │
   │       │                              │           │      │
   │       │                        ┌─────┴─────┐     │      │
   │       │                        │    Yes    │     │      │
   │       │                        ▼           │     │      │
   │       │                   Fine-tune LoRA   │     │      │
   │       │                        │           │     │      │
   │       └────────────────────────┴───────────┘     │      │
   │                                                  │      │
   └──────────────────────────────────────────────────┘      │

**Quality evaluation:**

.. math::

   Q(t, o) = w_1 \cdot \text{coherence}(o)
           + w_2 \cdot \text{completeness}(o, t)
           + w_3 \cdot \text{accuracy}(o)

where :math:`w_1 + w_2 + w_3 = 1`.

The Critic agent evaluates outputs:

.. code-block:: text

   Evaluate the following output for the given task.

   Task: {task_description}
   Output: {output}

   Rate on a scale of 0-100 for:
   - Coherence: Is it well-structured and readable?
   - Completeness: Does it fully address the task?
   - Accuracy: Are the claims factually correct?

   Output JSON: {"score": <0-100>, "feedback": "..."}

**Training data extraction:**

From session-log documents, we extract (task, response) pairs:

.. code-block:: rust

   fn extract_training_pair(doc: &str) -> Option<(String, String)> {
       let goal = extract_section(doc, "## Goal")?;
       let summary = extract_section(doc, "## Summary")
           .or_else(|| extract_section(doc, "## Solution"))?;

       if summary.len() < 100 {
           return None;  // Too short
       }

       Some((goal, summary))
   }

Paper Generation Pipeline
-------------------------

The paper generator produces arXiv-ready LaTeX:

**Stage 1: Research**

.. code-block:: rust

   let research = agent.execute(&Task::new(
       "Literature review",
       TaskType::Research,
       format!("Survey papers on: {topic}")
   )).await?;

**Stage 2: Outline**

.. code-block:: rust

   let outline = agent.execute(&Task::new(
       "Generate outline",
       TaskType::Writing,
       format!("Outline for paper on {topic} with {n} sections")
   )).await?;

**Stage 3: Section generation**

For each section, use the appropriate agent:

- Introduction, Conclusion → Writer
- Main Results, Proofs → Mathematician
- Implementation → Coder

**Stage 4: Abstract**

Summarize all sections into a 150-250 word abstract.

**Stage 5: Bibliography**

Extract citations from research and format as BibTeX.

**Stage 6: LaTeX compilation**

.. code-block:: latex

   \documentclass[11pt,a4paper]{article}
   \usepackage{amsmath,amssymb,amsthm}
   \usepackage{hyperref}

   \title{A Study of {topic}}
   \author{ThotBook-AmentI Autonomous Research System}
   \date{\today}

   \begin{document}
   \maketitle

   \begin{abstract}
   {abstract}
   \end{abstract}

   {sections}

   \bibliographystyle{plain}
   \begin{thebibliography}{99}
   {bibliography}
   \end{thebibliography}

   \end{document}

Orchestrator Configuration
--------------------------

.. code-block:: rust

   pub struct OrchestratorConfig {
       /// LLM inference endpoint
       pub inference_url: String,       // "http://127.0.0.1:8100"

       /// Session-log corpus directory for training data
       pub session_corpus_dir: PathBuf, // "../session-corpus"

       /// Lakehouse for outputs
       pub lakehouse_dir: PathBuf,      // "data/lakehouse"

       /// Quality threshold for training (0-100)
       pub quality_threshold: u8,       // 70

       /// Enable automatic retraining
       pub auto_retrain: bool,          // true

       /// Minimum samples before retraining
       pub retrain_min_samples: usize,  // 100
   }

Metrics & Monitoring
--------------------

The orchestrator tracks:

- **Task completion rate**: % of tasks successfully completed
- **Average quality score**: Mean score across all outputs
- **Training queue size**: Pending high-quality examples
- **Retraining frequency**: How often the model is fine-tuned
- **Latency**: Time per task, broken down by agent

These metrics are logged to the lakehouse for analysis:

.. code-block:: sql

   CREATE TABLE orchestrator.metrics (
       timestamp TIMESTAMP,
       metric_name STRING,
       metric_value DOUBLE,
       task_id STRING,
       agent_type STRING
   );

Future Directions
-----------------

Planned enhancements:

1. **Parallel execution**: Run independent subtasks concurrently
2. **Caching**: Skip repeated subtasks with same inputs
3. **Active learning**: Prioritize uncertain examples for training
4. **Multi-model ensemble**: Use different models for different agents
5. **Human-in-the-loop**: Request feedback for low-confidence outputs
