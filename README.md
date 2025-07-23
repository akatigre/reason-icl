## Overview
Based on the <a href="https://github.com/Shark-NLP/OpenICL/tree/main">OpenICL repository</a>.

This project implements an in-context learning (ICL) experiment with four different methods for retrieving relevant examples:

1. **Tag Trace Based ICL**: Uses the reasoning tag traces (e.g., "thinking", "focusing", "conclude") to find examples with similar reasoning patterns.
2. **Qwen 2.5 Embedding Similarity Based ICL**: Uses cosine similarity between raw LLM embeddings to find semantically similar examples.
3. **VAE Trained Embedding Similarity Based ICL**: Uses a trained VAE to transform the raw embeddings and then computes similarity in the learned latent space.
4. **Clustering Based ICL**: Clusters the VAE embeddings and selects examples based on cluster membership probability.

## Installation
Python 3.9, transformers > 4.49.0, install uv beforehand

**Installation for local development:**
```
cd reason-icl
uv venv --python 3.9
source .venv/bin/activate
uv pip install -e .
uv pip install -U transformers==4.53.1 numpy==1.26.4 seaborn matplotlib
uv pip install vllm
uv pip install flash-attn --no-build-isolation
```
numpy should be 1.x.x, transformers>4.53 to support qwen3


### Running experiments
```python
# predict
accelerate launch --multi_gpu --num_processes GPU_NUM exp/run.py
```