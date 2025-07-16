import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def save_into_jsonl():
    with open("data/AOKVQA/val.json", "r") as f:
        data = json.load(f)
        
    lines = []
    for key, line in data.items():
        if "cot_with_tag_variants" in line:
            lines.append({
                "id": key,
                "cot_wo_tag": " ".join(line["cot_wo_tag"]),
                "cot_with_tag": " ".join(line["cot_with_tag"]),
                "question": line["query"],
                "answer": line["choices"][line["correct_choice_idx"]],
                "cot_with_tag_correct": line["cot_with_tag_correct"],
                "cot_wo_tag_correct": line["cot_wo_tag_correct"],
                "cot_with_tag_negatives": line["cot_with_tag_variants"]["EntitySwap"] + line["cot_with_tag_variants"]["Paraphrasing"],
                "cot_wo_tag_negatives": line["cot_wo_tag_variants"]["EntitySwap"] + line["cot_wo_tag_variants"]["Paraphrasing"]
            })
    
    with open("data/AOKVQA/val.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    ds = load_dataset("json", data_files="data/AOKVQA/val.jsonl")

    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", model_kwargs={"torch_dtype": "bfloat16"})

    # keys_to_embed = ["solution", ""]
    pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1"])
    # Encode with multiple GPUs
    keys_to_embed = ["cot_wo_tag", "cot_wo_tag_negatives", "question"]
    for key in keys_to_embed:
        if isinstance(ds[key], list):
            inputs_nested_list = [line[key] for line in ds]
            inputs_list = None
        else:
            inputs_list = [line[key] for line in ds]
        if inputs_list is not None:
            embeddings = model.encode(inputs_list, pool=pool)
        else:
            embeddings_list = [model.encode(list_of_strings) for list_of_strings in inputs_nested_list]

        for line in ds:
            ds[""]
    model.stop_multi_process_pool(pool)
    
    
    ds_with_embeddings.save_faiss_index('embeddings', 'my_index.faiss')
    ds = load_dataset('crime_and_punish', split='train[:100]')
    ds.load_faiss_index('embeddings', 'my_index.faiss')