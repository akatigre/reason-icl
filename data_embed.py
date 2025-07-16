import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


ds = load_dataset("json", data_files="data/MATH500/test_perturbed.jsonl")

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

ds_with_embeddings = ds.map(lambda example: {'embeddings': model.encode(example["question"]).numpy()})