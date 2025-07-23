# Exploring Retrieval-Augmented Generation (RAG) Systems

## Overview

This repository contains the codebase and experimental results for a project exploring **Retrieval-Augmented Generation (RAG)** systems in multi-hop question-answering (QA) tasks. The project systematically evaluates the trade-offs between different retrieval, reranking, and generation algorithms to identify optimal configurations for effective short-answer QA.

The experiments focus on the two primary stages of RAG:
1. **Retrieval Stage**: Identifying top-k relevant documents using various retrieval models.
2. **Generation Stage**: Synthesizing accurate, concise answers using large language models (LLMs) with retrieved documents as context.

Through a combination of prompt engineering, lightweight post-processing, and systematic analysis, this study highlights the interplay between retrieval and generation for RAG.

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
## Experimental Results

### Retrieval
Evaluation metrics included **MRR@10**, **NDCG@10**, and **Hits@10**:
- **BAAI/BGE-Reranker-base** outperformed other models with superior ranking precision and relevance.
- **BM25**, while less modern, delivered strong results in keyword-dependent queries.

| Model                 | Hits@10 | MAP@10 | MRR@10 | NDCG@10 |
|-----------------------|---------|--------|--------|---------|
| BGE-Reranker-base     | 0.6204  | 0.2149 | 0.4936 | 0.6614  |
| all-mpnet-base-v2     | 0.4523  | 0.1169 | 0.2645 | 0.3833  |
| BM25                  | 0.5060  | 0.1505 | 0.3293 | 0.2344  |
| LLM-Embedder-ranker   | 0.5348  | 0.1408 | 0.3299 | 0.4690  |

### Generation
Key metrics: **Precision**, **F1 Score**, **ROUGE-1**, **ROUGE-L**, and **BLEU**:
- **Starling-7B Alpha** provided the most precise and contextually relevant answers, making it ideal for short-answer tasks.
- **Zephyr-7B Beta** performed well but exhibited biases in binary yes/no answers.
- **LLaMA-2** and **Mistral-7B** showed limitations in generating concise responses.

| Model          | Precision | Recall | F1 Score | ROUGE-1 | ROUGE-L | BLEU   |
|----------------|-----------|--------|----------|---------|---------|--------|
| LLaMA-2        | 0.4182    | 0.2267 | 0.2334   | 0.2358  | 0.2356  | 0.0554 |
| Mistral-7B     | 0.4350    | 0.3220 | 0.3316   | 0.3349  | 0.3344  | 0.0731 |
| Starling-7B    | 0.5783    | 0.5739 | 0.5739   | 0.5742  | 0.5740  | 0.1043 |
| Zephyr-7B Beta | 0.4469    | 0.4170 | 0.4196   | 0.4245  | 0.4239  | 0.0529 |

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
- **`MyRAGEval.ipynb`**: Visualizations of performance metrics of RAG stage.
- **`MyRetEval.ipynb`**: Visualizations of performance metrics of Retrieval stage.
---

## Installation and Usage

4. (Optional) To re-produce all the figures and tables from the report, run all the cells in:
    - MyRetEval.ipynb
    - MyRAGEval.ipynb
    - data_visualize.ipynb
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-system-experiments.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run retrieval experiments:
   - To run all the retrieval, execute: `bash run_stage_1.sh`
      - or you can run each file separately: e.g. `python baai_llm_embedder_reranker.py`
   - To evaluate a stage one's output (e.g. `output/baai_llm_embedder_reranker.json`): `python MyRetEval.py output/baai_llm_embedder_reranker.json`
   - To run all the RAG, execute: `bash run_rag.sh`
      - or you can run each file separately: e.g. `python llama2_7b_hf.py`
   - (Optional) To re-produce all the figures and tables from the report, run all the cells in:
      - `MyRetEval.ipynb`, `MyRAGEval.ipynb`, `data_visualize.ipynb`
