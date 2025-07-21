import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import random
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk, Array2D
from data_template import DATA_MAP
from data.perturb_reasoning import generate_negatives_variations, generate_positives_variations
from data.generate_cot import plan_wo_tags
from utils.parse_utils import extract_last_boxed_text
import gc
from PIL import Image




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
            
def prepare_aokvqa():
    import os
    import json
    from utils.aokvqa_utils import get_coco_path
    aokvqa_dir = "/home/server08/hdd1/yoonjeon_workspace/aokvqa/"
    coco_dir = "/home/server08/hdd1/yoonjeon_workspace/coco/"
    split = "train"
    dataset = json.load(open(
            os.path.join(aokvqa_dir, f"aokvqa_v1p0_{split}.json")
        ))

    data_list = []
    for data_item in dataset:
        choices = ", ".join(data_item["choices"])
        data_list.append({
            "image_path": get_coco_path(split, data_item["image_id"], coco_dir),
            "question": f"{data_item['question']}. Answer from one of the choices {choices}. When the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase.",
            "answer": data_item["choices"][data_item["correct_choice_idx"]],
        })

    with open(f"AOKVQA/{split}/{split}.jsonl", "w") as f:
        for data_item in data_list:
            f.write(json.dumps(data_item, ensure_ascii=False) + "\n")

def generate_cots(llm, processor, sampling_params, ds, gts):
    prompts = []
    ds = ds.select(range(10))
    for row in tqdm(ds, desc="Generating COTs without answer"):
        message = plan_wo_tags(question=row["question"], answer=None)
        prompt = processor.apply_chat_template(message, tokenize=False)

        image = row.get("image_path", None)
        if image:
            pil_image = Image.open(image)
            prompts.append({
                "prompt": prompt,
                "multi_modal_data": {"image": pil_image},
            })
        else:
            prompts.append(prompt)
        
    
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() + " The answer" for output in outputs]
    extracted = [extract_last_boxed_text(response) for response in responses]
    correct_idxs = [i for i, answer in enumerate(extracted) if answer == gts[i]]
    ds = ds.add_column("reason", responses)
    
    prompts = []
    for idx, row in tqdm(enumerate(ds), desc="Generating COTs with answer"):
        if idx in correct_idxs:
            continue
        else:
            for row in tqdm(ds, desc="Generating COTs without answer"):
                message = plan_wo_tags(question=row["question"], answer=None)
                prompt = processor.apply_chat_template(message, tokenize=False)

                image = row.get("image_path", None)
                if image:
                    pil_image = Image.open(image)
                    prompts.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": pil_image},
                    })
                else:
                    prompts.append(prompt)
        
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    responses = [output.outputs[0].text.strip() for output in outputs]
    for idx, (row, response) in enumerate(zip(ds, responses)):
        if idx in correct_idxs:
            continue
        else:
            row["reason"] = response
    return ds

def generate_variations(llm, processor, sampling_params, ds, output_path):
    idxs = list(range(len(ds)))
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
    return ds

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
    for key in ["negatives", "positives"]:  
        nested_inputs = [line[key] for line in ds]
        nested_embs = []
        for lines in tqdm(nested_inputs, desc=f"Embedding {key}"):
            embs = model.encode(lines, pool=pool)
            nested_embs.append(embs.tolist())
            
        M, D = len(nested_embs[0]), len(nested_embs[0][0])
        ds = ds.add_column(key + "_emb", nested_embs)
        ds = ds.cast_column(key + "_emb", Array2D(shape=(M, D), dtype="float32"))
    

    if pool:
        model.stop_multi_process_pool(pool)
    
    ds.save_to_disk(os.path.join(save_to_path, gen_model.split("/")[-1]))


def load_data_and_embeddings(ds, save_to_path, emb_model, keys_to_embed):
    path_to_emb = os.path.join(save_to_path, emb_model.split("/")[-1])
    for key in keys_to_embed:
        key_emb = key + "_emb"
        faiss_path = os.path.join(path_to_emb, key + ".faiss")
        ds.add_faiss_index(column=key_emb)
        ds.load_faiss_index(key_emb, faiss_path)
    return ds.with_format("numpy")


def main(data_type, multi_modal=False, tensor_parallel_size=4, emb_model=None, gen_model=None):
    from pathlib import Path
    data_config = DATA_MAP[data_type]

    keys_to_embed = data_config["input_columns"] + [data_config["output_column"]]
    splits = [data_config["train_split"], data_config["test_split"]]

    for split in splits:
        print(f"\n▶ Processing split: {split}")
        enhanced_path = f"data/{data_type}/{split}/{split}_enhanced.jsonl"
        if os.path.exists(enhanced_path):
            ds = load_dataset('json', data_files=enhanced_path)['train']
        else:
            ds = load_dataset('json', data_files=str(Path(data_config["data_path"]) / split / f"{split}.jsonl"))['train']
        
        # Load vLLM model for generation
        print("🔄 Loading vLLM model for generation...")
        llm, sampling_params, processor = load_model(gen_model, max_model_token_num=3072, tensor_parallel_size=tensor_parallel_size, multi_modal=multi_modal)
        if "reason_answer" not in ds[0]:
            gts = [row["answer"] for row in ds]
            ds = generate_cots(llm, processor, sampling_params, ds, gts)
            with open(enhanced_path, "w") as f:
                for row in ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if "negatives" not in ds[0]:
            ds = generate_variations(llm, processor, sampling_params, ds, enhanced_path)
            with open(enhanced_path, "w") as f:
                for row in ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="gsm8k")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    args = parser.parse_args()
    
    multi_modal = True if args.data_type == "AOKVQA" else False
    emb_model = "Qwen/Qwen3-Embedding-4B"
    gen_model = "Qwen/Qwen2.5-VL-7B-Instruct" if multi_modal else "Qwen/Qwen2.5-7B-Instruct"
    
    data_config = DATA_MAP[args.data_type]
    # keys_to_embed = data_config["input_columns"] + [data_config["output_column"]]
    main(args.data_type, tensor_parallel_size=args.tensor_parallel_size, multi_modal=multi_modal, emb_model=emb_model, gen_model=gen_model)
    # raw_path = f"data/{args.data_type}/{data_config['train_split']}/{data_config['train_split']}.jsonl"
    # enhanced_path = f"data/{args.data_type}/{data_config['train_split']}/{data_config['train_split']}_enhanced.jsonl"
    
    # ds_enhanced = load_dataset("json", data_files=enhanced_path)["train"]
    # embed_and_save_faiss(ds_enhanced, os.path.dirname(raw_path), emb_model, gen_model, keys_to_embed)