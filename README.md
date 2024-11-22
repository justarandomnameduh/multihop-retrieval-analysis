# README: Exploring Retrieval-Augmented Generation (RAG) Systems

## Overview

This repository contains the codebase and experimental results for a project exploring **Retrieval-Augmented Generation (RAG)** systems in multi-hop question-answering (QA) tasks. The project systematically evaluates the trade-offs between different retrieval, reranking, and generation algorithms to identify optimal configurations for effective short-answer QA.

The experiments focus on the two primary stages of RAG:
1. **Retrieval Stage**: Identifying top-k relevant documents using various retrieval models.
2. **Generation Stage**: Synthesizing accurate, concise answers using large language models (LLMs) with retrieved documents as context.

Through a combination of prompt engineering, lightweight post-processing, and systematic analysis, this study highlights the nuanced interplay between retrieval and generation for factoid QA.

---

## Features

### Retrieval Algorithms
- **BAAI/LLM-Embedder**: Dense vector-based retrieval optimized for similarity-based search.
- **BAAI/BGE-Reranker-base**: Direct relevance scoring via cross-entropy loss for precise ranking.
- **Sentence-Transformer (all-mpnet-base-v2)**: Semantic search using cosine similarity on embeddings.
- **BM25**: A term frequency-based model with normalization for document length.

### Generation Models
- **LLaMA-2**: Causal language model for text generation and dialogue tasks.
- **Mistral-7B**: Lightweight and resource-efficient model for real-time applications.
- **Starling-7B Alpha**: Uses Reinforcement Learning from AI Feedback (RLAIF) for improved alignment with user intent.
- **Zephyr-7B Beta**: Implements Distilled Direct Preference Optimization (dDPO) and fine-tuning for instruction following.

---

## Key Results

### Retrieval Stage
The retrieval algorithms were evaluated on metrics like **MRR@10**, **NDCG@10**, and **Hits@10**:
- **BAAI/BGE-Reranker-base** outperformed other models in ranking quality and relevance across all metrics.
- **BM25**, though older, demonstrated robustness in exact keyword matching tasks.

### Generation Stage
Evaluation metrics included **Precision**, **Recall**, **F1 Score**, **ROUGE-1**, **ROUGE-L**, and **BLEU**:
- **Starling-7B** achieved the best results, providing highly precise and contextually relevant answers.
- **Zephyr-7B Beta** performed well but exhibited bias in binary (yes/no) responses.
- **LLaMA-2** and **Mistral-7B** struggled with concise answers, highlighting the importance of prompt engineering.

---

## Code Structure

### Retrieval Models
- **`bm25_ranking.py`**: BM25 retrieval pipeline.
- **`all_mpnet_base_v2.py`**: Semantic ranking with Sentence-Transformer.
- **`baai_llm_embedder.py`**: Retrieval with dense embeddings from BAAI's LLM.
- **`baai_llm_embedder_reranker.py`**: Reranking pipeline for retrieval outputs.

### Generation Models
- **`llama2_7b_hf.py`**: Text generation with LLaMA-2.
- **`mistral_7b_instruct_v03.py`**: Fine-tuned inference with Mistral-7B.
- **`starling_7b_alpha.py`**: Prompt-based QA with Starling-7B.
- **`zephyr_7b_beta.py`**: Contextual response generation with Zephyr-7B.

### Visualization and Analysis
- **`data_visualize.ipynb`**: Visualizations of performance metrics and experimental results.

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-system-experiments.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run retrieval experiments:
   ```bash
   bash run_stage_1.sh
   ```
4. Evaluate generation models:
   ```bash
   python run_rag.sh
   ```

---

## Future Directions

- Investigating RAG systems under constrained environments (e.g., low-memory GPUs).
- Exploring bias mitigation techniques for binary QA tasks.
- Extending RAG applications to real-world domains like legal or medical QA.

---

## References
- **Mars, M.** (2022). From Word Embeddings to Pre-Trained Language Models: A State-of-the-Art Walkthrough. *Applied Sciences*. [DOI:10.3390](https://doi.org/10.3390).

---

Contributions and feedback are welcome! 🚀
