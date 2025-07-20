import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk, Array2D
from data_template import DATA_MAP
from data.perturb_reasoning import generate_negatives_variations, generate_positives_variations
import gc


import torch  # 🔧 추가

def load_model(model_path, max_model_token_num, tensor_parallel_size=4, multi_modal=False):
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)

    if multi_modal:
        temp, top_p, top_k, rep = 0.1, 0.001, 50, 1.05
    else:
        temp, top_p, top_k, rep = 0.7, 0.8, 20, 1.0

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=rep,
        max_tokens=max_model_token_num,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.padding_side = "left"

    return llm, sampling_params, processor


def prepare_data(dataset, data_config, split, output_path):
    data = dataset[split]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for row in data:
            row = data_config["data_processor"](row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_variations(llm, processor, sampling_params, ds, output_path):
    idxs = list(range(len(ds)))

    with open(output_path, "w") as f:
        for row in tqdm(ds, desc="Generating variations"):
            try:
                negatives = generate_negatives_variations(llm, processor, sampling_params, row["reason_answer"], num_variants=30)
                random_idxs = random.sample(idxs, 30)
                ds_random = [ds[i]["reason_answer"] for i in random_idxs]
                positives = generate_positives_variations(llm, processor, sampling_params, row["reason_answer"], ds_random)

                row["negatives"] = negatives
                row["positives"] = positives
            except Exception as e:
                print(f"Error processing row: {e}")
                row["negatives"] = []
                row["positives"] = []

            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_and_save_faiss(ds, save_to_path, emb_model, gen_model, keys_to_embed):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        emb_model,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16"
        },
        tokenizer_kwargs={"padding_side": "left"},
    )

    try:
        # Use fewer GPUs for sentence transformer to avoid conflicts
        available_gpus = ["cuda:0", "cuda:1"]  # Use only 2 GPUs instead of 4
        pool = model.start_multi_process_pool(target_devices=available_gpus)
        print(f"✅ Multi-process pool started on {available_gpus}")
    except RuntimeError as e:
        print("❌ Multi-process pool failed to start. Falling back to single-process.")
        pool = None

    # Flat text fields
    for key in keys_to_embed:
        key_emb = key + "_emb"
        inputs_list = [line[key] for line in ds]
        print(f"🔄 Embedding {key} field...")
        emb = model.encode(inputs_list, pool=pool)
        ds = ds.add_column(key_emb, emb.tolist())
        ds.add_faiss_index(column=key_emb)
        ds.save_faiss_index(key_emb, os.path.join(save_to_path, emb_model.split("/")[-1], key + ".faiss"))

    # Nested field
    nested_inputs = [line["negatives"] for line in ds]
    nested_embs = []
    for lines in tqdm(nested_inputs, desc="Embedding negatives"):
        embs = model.encode(lines, pool=pool)
        nested_embs.append(embs.tolist())
    if pool:
        model.stop_multi_process_pool(pool)

    M, D = len(nested_embs[0]), len(nested_embs[0][0])
    ds = ds.add_column("negatives_emb", nested_embs)
    ds = ds.cast_column("negatives_emb", Array2D(shape=(M, D), dtype="float32"))
    ds.save_to_disk(os.path.join(save_to_path, gen_model.split("/")[-1]))


def load_data_and_embeddings(ds, save_to_path, emb_model, keys_to_embed):
    path_to_emb = os.path.join(save_to_path, emb_model.split("/")[-1])
    for key in keys_to_embed:
        key_emb = key + "_emb"
        faiss_path = os.path.join(path_to_emb, key + ".faiss")
        ds.add_faiss_index(column=key_emb)
        ds.load_faiss_index(key_emb, faiss_path)
    return ds.with_format("numpy")


def main(data_type):
    data_config = DATA_MAP[data_type]
    dataset = load_dataset(data_config["data_path"], 'main')

    emb_model = "Qwen/Qwen3-Embedding-4B"
    gen_model = "Qwen/Qwen2.5-7B-Instruct"

    keys_to_embed = data_config["input_columns"] + [data_config["output_column"]]
    splits = [data_config["train_split"], data_config["test_split"]]

    for split in splits:
        print(f"\n▶ Processing split: {split}")
        raw_path = f"data/{data_type}/{split}/{split}.jsonl"
        enhanced_path = f"data/{data_type}/{split}/{split}_enhanced.jsonl"
        prepare_data(dataset, data_config, split, raw_path)

        ds = load_dataset("json", data_files=raw_path)["train"]
        
        # Load vLLM model for generation
        print("🔄 Loading vLLM model for generation...")
        llm, sampling_params, processor = load_model(
            gen_model, max_model_token_num=2048, tensor_parallel_size=4, multi_modal=False
        )
        
        generate_variations(llm, processor, sampling_params, ds, enhanced_path)

    print("🔄 Shutting down vLLM engine and freeing CUDA memory...")
    # 1) Explicitly shut down the engine (kills worker procs & ZMQ contexts)
    try:
        # if vLLM wrapper has shutdown() method
        if hasattr(llm, "shutdown"):
            llm.shutdown()
        # fallback: call into the engine core directly
        elif hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
    except Exception:
        pass

    # 2) Delete and GC–collect the Python objects
    del llm, sampling_params, processor
    gc.collect()

    # 3) Empty and reset *all* CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for split in splits:
        raw_path = f"data/{data_type}/{split}/{split}.jsonl"
        enhanced_path = f"data/{data_type}/{split}/{split}_enhanced.jsonl"
        
        ds_enhanced = load_dataset("json", data_files=enhanced_path)["train"]
        embed_and_save_faiss(ds_enhanced, os.path.dirname(raw_path), emb_model, gen_model, keys_to_embed)
        # ds_enhanced = load_data_and_embeddings(ds_enhanced, os.path.dirname(raw_path), emb_model, keys_to_embed)



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    main(data_type="gsm8k")
