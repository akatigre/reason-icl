import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Array2D
from sentence_transformers import SentenceTransformer
from data_template import DATA_MAP

def embed_and_save_faiss(ds, save_to_path, emb_model, gen_model, keys_to_embed):
    model = SentenceTransformer(
        emb_model, 
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1"])
    # Encode with multiple GPUs
    embeddings = {}
    for key in keys_to_embed:
        key_to_save = key + "_emb"
        inputs_list = [line[key] for line in ds]
        
        embeddings = model.encode(inputs_list, pool=pool)
        ds = ds.add_column(key_to_save, embeddings.tolist())
        ds.add_faiss_index(column=key_to_save)
        ds.save_faiss_index(key_to_save, os.path.join(save_to_path, emb_model.split("/")[-1], key + ".faiss"))
    
    # inputs_nested_list = [line["negatives"] for line in ds]
    # embedding_list = []
    # for list_of_strings in tqdm(inputs_nested_list):
    #     embeddings = model.encode(list_of_strings, pool=pool)
    #     embedding_list.append(embeddings.tolist())
    # model.stop_multi_process_pool(pool)
    
    # M, D = len(embedding_list[0]), len(embedding_list[0][0])
    
    # key_to_save = "negatives_emb"
    # ds = ds.add_column(key_to_save, embedding_list)
    # ds = ds.cast_column(key_to_save, Array2D(shape=(M, D), dtype="float32"))
    # ds.save_to_disk(os.path.join(save_to_path, gen_model.split("/")[-1]))

def load_data_and_embeddings(ds, save_to_path, emb_model, keys_to_embed):

    path_to_emb = os.path.join(save_to_path, emb_model.split("/")[-1])
    faiss_keys = [key+".faiss" for key in keys_to_embed]
    emb_paths = [os.path.join(path_to_emb, key) for key in faiss_keys]
    for faiss_key, emb_path in zip(faiss_keys, emb_paths):
        key_name = faiss_key.replace(".faiss", "")+"_emb"
        ds.add_faiss_index(column=key_name)
        ds.load_faiss_index(key_name, emb_path)
    ds = ds.with_format("numpy") # converts list in list to numpy array
    return ds

if __name__ == "__main__":
    
    data_type = "gsm8k"
    
    data_config = DATA_MAP[data_type]
    dataset = load_dataset(data_config["data_path"], 'main')
    emb_model = "Qwen/Qwen3-Embedding-4B"
    gen_model = "Qwen/Qwen2.5-7B-Instruct"
    
    keys_to_embed = data_config["input_columns"] + [data_config["output_column"]]
    splits = [data_config["train_split"], data_config["test_split"]]
    for split in splits:
        data = dataset[split]
        output_path = f"data/{data_type}/{split}/{split}.jsonl"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for row in data:
                row = data_config["data_processor"](row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
        ds = load_dataset("json", data_files=output_path)["train"]
        embed_and_save_faiss(ds, os.path.dirname(output_path), emb_model, gen_model, keys_to_embed)
        ds = load_data_and_embeddings(ds, os.path.dirname(output_path), emb_model, keys_to_embed)
        
    # save_into_jsonl(data_path_json)
    # gen_model = "Qwen/Qwen2.5-7B-Instruct"
    # data_path_jsonl = f"data/AOKVQA/{gen_model.split('/')[-1]}/val.jsonl"
    # emb_model = "Qwen/Qwen3-Embedding-4B"
    # save_to_path = f"data/AOKVQA/{emb_model.split('/')[-1]}"
    # ds = load_dataset("json", data_files=data_path_jsonl)["train"]
    
    # data_path_json = "data/AOKVQA/val.json"
    
    # embed_and_save_faiss(ds, save_to_path, emb_model)
    # data_path_jsonl = f"data/AOKVQA/Qwen2.5-7B-Instruct/val.jsonl"
    # emb_model = "Qwen/Qwen3-Embedding-4B"
    # gen_model = "Qwen/Qwen2.5-7B-Instruct"
    # save_to_path = f"data/AOKVQA/"
    # ds = load_dataset("json", data_files=data_path_jsonl)["train"]
    # embed_and_save_faiss(ds, save_to_path, emb_model, gen_model)
    # ds = load_data_and_embeddings(save_to_path, gen_model, emb_model)
    
    