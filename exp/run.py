
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
from openicl import DatasetReader
from openicl import RandomRetriever, BM25Retriever, ConERetriever, TopkRetriever
from datasets import load_dataset
from data_template import DATA_MAP, IN_CONTEXT_EXAMPLE_TOKEN
from sentence_transformers import SentenceTransformer
from utils.parse_utils import extract_last_boxed_text
from utils.math_utils import grade_answer_sympy
from utils.model_utils import load_model
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--icl_shot", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", choices=["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"])
    parser.add_argument("--retrieve_model_path", type=str, default="Qwen/Qwen3-Embedding-4B")
    
    args = parser.parse_args()
    
    task_names = list(DATA_MAP.keys())
    TEST = False
    
    for task_name in task_names:
        logger.info(f"Running {task_name} with retrieve model {args.retrieve_model_path} and inference model {args.model_path}")
        config = DATA_MAP[task_name]
        dataset = load_dataset(config["data_path"], config["subset_name"])
        dataset[config["train_split"]] = dataset[config["train_split"]].map(config["data_processor"])
        dataset[config["test_split"]] = dataset[config["test_split"]].map(config["data_processor"])
        ds_size = 50 if TEST else None
        data = DatasetReader(dataset, input_columns=config["input_columns"], output_column=config["output_column"], ds_size=ds_size)
        logger.info(f"Loading {args.icl_shot} shot TopK retriever")
        model = SentenceTransformer(
            args.retrieve_model_path,
            model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"},
            tokenizer_kwargs={"padding_side": "left"},
        )
        dim = model.get_sentence_embedding_dimension()
        retriever = TopkRetriever(
            data, 
            ice_num=args.icl_shot, 
            model=model,
            tokenizer_name=args.retrieve_model_path,
            batch_size=4,
            index_split=config["train_split"], 
            test_split=config["test_split"]
            )

        ice_idx_list = retriever.retrieve()
        model.cpu()
        del model
        torch.cuda.empty_cache()
        
        logger.info(f"Loading inference model {args.model_path} into VLLM gpus: {args.num_gpus}")
        llm, sampling_params, processor = load_model(args.model_path, max_model_token_num=2048, tensor_parallel_size=args.num_gpus, multi_modal=False)
        
        prompt_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx, ice_template=config["template"])
            prompt = retriever.generate_prompt_for_generate_task(
                idx, 
                ice, 
                gen_field_replace_token=IN_CONTEXT_EXAMPLE_TOKEN,
                prompt_template=config["template"]
            )
            prompt_list.append(prompt)
        
        # images = [retriever.test_ds[i]['image'] for i in range(len(retriever.test_ds))] if 'image' in retriever.test_ds[0] else None
        question_list = [retriever.test_ds[i]['question'] for i in range(len(retriever.test_ds))]
        answer_list = [retriever.test_ds[i]['answer'] for i in range(len(retriever.test_ds))]
        
        prompt_list = [processor.apply_chat_template(
            [
                {"content": "You are a helpful assistant.", "role": "system"},
                {"content": prompt, "role": "user"}
            ], add_generation_prompt=True, tokenize=False) for prompt in prompt_list]
        
        
        logger.info(f"Generating {len(prompt_list)} responses")
        outputs = llm.generate(prompt_list, sampling_params=sampling_params, use_tqdm=True)
        results = []
        
        logger.info(f"Processing responses")
        for i, (question, answer, icl_prompt, output) in enumerate(zip(question_list, answer_list, prompt_list, outputs)):
            response = output.outputs[0].text
            ext_answer = extract_last_boxed_text(response)
            results.append({
                "question": question,
                "gt_answer": answer,
                "icl_prompt": icl_prompt,
                "full_response": response,
                "extracted_answer": ext_answer,
                "correct": grade_answer_sympy(ext_answer, answer)[0]
            })
        
        output_json_filepath = f'results/{task_name}/{args.model_path.split("/")[-1]}_{args.retrieve_model_path.split("/")[-1]}_TOPK_{args.icl_shot:02d}shot_seed_{args.seed}.jsonl'
        os.makedirs(os.path.dirname(output_json_filepath), exist_ok=True)
        logger.info(f"Saving results to {output_json_filepath}")
        with open(output_json_filepath, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
                
                