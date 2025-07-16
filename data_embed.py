import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

with open("data/MATH500/test_perturbed.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
for line in data:
    line["cot_with_tag_correct"] = str(line.get("cot_with_tag_correct", [""])[0])
    line["cot_wo_tag_correct"] = str(line.get("cot_wo_tag_correct", [""])[0])
    
with open("data/MATH500/test.jsonl", "w") as f:
    for line in data:
        f.write(json.dumps(line) + "\n")

ds = load_dataset("json", data_files="data/MATH500/test.jsonl")

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

ds_with_embeddings = ds.map(lambda example: {'solution_emb': model.encode(example["solution"]).numpy()})
breakpoint()