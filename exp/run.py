import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
from accelerate import Accelerator
from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import RandomRetriever, BM25Retriever, ConERetriever, TopkRetriever, PPLInferencer, AccEvaluator, GenInferencer
from datasets import load_dataset
from pathlib import Path

from utils.math_utils import grade_answer_sympy
from utils.parse_utils import extract_last_boxed_text
from data_template import DATA_MAP, IN_CONTEXT_EXAMPLE_TOKEN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--icl_shot", type=int, default=8)
    parser.add_argument("--batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--retrieve_model_path", type=str, default="Qwen/Qwen3-Embedding-4B")
    
    args = parser.parse_args()
    
    task_names = list(DATA_MAP.keys())
    
    seeds = [args.seed]
    for model_path in [args.model_path]:
        accelerator = Accelerator()
        for seed in seeds:
            for task_name in task_names:
                config = DATA_MAP[task_name]
                output_json_filepath = f'results/{task_name}_{model_path.split("/")[-1]}_{args.retrieve_model_path.split("/")[-1]}_{args.icl_shot:02d}shot'
                
                os.makedirs(output_json_filepath, exist_ok=True)
                dataset = load_dataset(config["data_path"], config["subset_name"])
                dataset[config["train_split"]] = dataset[config["train_split"]].map(config["data_processor"])
                dataset[config["test_split"]] = dataset[config["test_split"]].map(config["data_processor"])
                data = DatasetReader(dataset, input_columns=config["input_columns"], output_column=config["output_column"], ds_size=10)
                topk_retriever = TopkRetriever(
                    data, ice_num=args.icl_shot, sentence_transformers_model_name=args.retrieve_model_path, tokenizer_name=args.retrieve_model_path,
                    batch_size=1, index_split=config["train_split"], test_split=config["test_split"], accelerator=accelerator
                    )

                # cone_retriever = ConERetriever(data, ice_num=icl_shot, candidate_num=30, sentence_transformers_model_name=retrieve_model_name, tokenizer_name=retrieve_model_name, model_tokenizer_name=model_path, ce_model_name=model_path, ice_template=config["template"], select_time=candidate_num, seed=seed, batch_size=batch_size, test_split='test', accelerator=accelerator)
                inferencer = GenInferencer(
                    model_name=model_path, tokenizer=model_path,
                    output_json_filepath=output_json_filepath,
                    batch_size=args.batch_size_per_gpu * args.num_gpus,
                    accelerator=accelerator,
                    generation_kwargs={"max_new_tokens": 2048},
                    gen_field_replace_token=IN_CONTEXT_EXAMPLE_TOKEN
                    )

                topk_predictions = inferencer.inference(topk_retriever, ice_template=config["template"], output_json_filename=f'topk_seed_{seed}')
                print(output_json_filepath)