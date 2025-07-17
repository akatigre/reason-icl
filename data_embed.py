import os
import json
import pickle
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Array2D
from sentence_transformers import SentenceTransformer


def embed_and_save_faiss(ds, save_to_path, emb_model, gen_model):
    model = SentenceTransformer(emb_model, model_kwargs={"torch_dtype": "bfloat16"})
    pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1"])
    # Encode with multiple GPUs
    keys_to_embed = ["cot_wo_tag", "question"]
    embeddings = {}
    for key in keys_to_embed:
        key_to_save = key + "_emb"
        inputs_list = [line[key] for line in ds]
        
        embeddings = model.encode(inputs_list, pool=pool)
        ds = ds.add_column(key_to_save, embeddings.tolist())
        ds.add_faiss_index(column=key_to_save)
        ds.save_faiss_index(key_to_save, os.path.join(save_to_path, emb_model.split("/")[-1], key + ".faiss"))
    
    inputs_nested_list = [line["cot_wo_tag_negatives"] for line in ds]
    embedding_list = []
    for list_of_strings in tqdm(inputs_nested_list):
        embeddings = model.encode(list_of_strings, pool=pool)
        embedding_list.append(embeddings.tolist())
    model.stop_multi_process_pool(pool)
    
    M, D = len(embedding_list[0]), len(embedding_list[0][0])
    
    key_to_save = "cot_wo_tag_negatives_emb"
    ds = ds.add_column(key_to_save, embedding_list)
    ds = ds.cast_column(key_to_save, Array2D(shape=(M, D), dtype="float32"))
    ds.save_to_disk(os.path.join(save_to_path, gen_model.split("/")[-1]))

def load_data_and_embeddings(save_to_path, gen_model, emb_model):
    path_to_dataset = os.path.join(save_to_path, gen_model.split("/")[-1])
    ds = load_from_disk(path_to_dataset)
    path_to_emb = os.path.join(save_to_path, emb_model.split("/")[-1])
    faiss_keys = ["question.faiss", "cot_wo_tag.faiss"]
    emb_paths = [os.path.join(path_to_emb, key) for key in faiss_keys]
    for faiss_key, emb_path in zip(faiss_keys, emb_paths):
        key_name = faiss_key.replace(".faiss", "")+"_emb"
        ds.add_faiss_index(column=key_name)
        ds.load_faiss_index(key_name, emb_path)
    ds = ds.with_format("numpy") # converts list in list to numpy array
    return ds

if __name__ == "__main__":
    # data_path_json = "data/AOKVQA/val.json"
    # save_into_jsonl(data_path_json)
    # gen_model = "Qwen/Qwen2.5-7B-Instruct"
    # data_path_jsonl = f"data/AOKVQA/{gen_model.split('/')[-1]}/val.jsonl"
    # emb_model = "Qwen/Qwen3-Embedding-4B"
    # save_to_path = f"data/AOKVQA/{emb_model.split('/')[-1]}"
    # ds = load_dataset("json", data_files=data_path_jsonl)["train"]
    # embed_and_save_faiss(ds, save_to_path, emb_model)
    data_path_jsonl = f"data/AOKVQA/Qwen2.5-7B-Instruct/val.jsonl"
    emb_model = "Qwen/Qwen3-Embedding-4B"
    gen_model = "Qwen/Qwen2.5-7B-Instruct"
    save_to_path = f"data/AOKVQA/"
    ds = load_dataset("json", data_files=data_path_jsonl)["train"]
    embed_and_save_faiss(ds, save_to_path, emb_model, gen_model)
    ds = load_data_and_embeddings(save_to_path, gen_model, emb_model)
    
    