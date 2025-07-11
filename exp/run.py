import os
from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import RandomRetriever, BM25Retriever, ConERetriever, TopkRetriever, PPLInferencer, AccEvaluator, GenInferencer
from datasets import load_dataset
from accelerate import Accelerator


def processing_answer(str):
    str = str.split(' ')[::-1]
    flag = False
    ret = ''
    for i in range(len(str)):
        s = str[i]
        for i in range(len(s)):
            if s[i].isdigit():
                flag = True
                ret = s
                break
        if flag:
            break
    ret1 = ''
    for i in range(len(ret)):
        if ret[i].isdigit():
            ret1 += ret[i]
    return ret1
    
if __name__ == '__main__':
    qwen_template = "<|im_start|>system\nLet's think step by step and output answer in \\boxed{}<|im_end|>\n<|im_start|>user\nQuestion: </Q> <|im_end|>\n<|im_start|>assistant\n </A> <|im_end|>\n"
    # qwenvl_template = "<|im_start|>system\nLet's think step by step and output answer in \\boxed{}<|im_end|>\n<|im_start|>user\nQuestion: </Q> <|im_end|>\n<|im_start|>assistant\n </A> <|im_end|>\n"
    in_context_placeholder_token_map = {'question':'</Q>', 'answer':'</A>'}
    in_context_example_token = '</E>'
    data_map = {
        "gsm8k": {
            "template": PromptTemplate(
                template = in_context_example_token + qwen_template,
                column_token_map = in_context_placeholder_token_map,
                ice_token = in_context_example_token,
            ),
            "input_columns": ["question"],
            "output_column": "answer",
            "train_split": "train",
            "test_split": "test",
            "data_path": "openai/gsm8k",
            "subset_name": "main",
            "data_processor": lambda example: {'answer': example['answer'].split("#### ")[0].replace(',', '').strip() + "\\boxed{" + example['answer'].split("#### ")[1].replace(',', '').strip() + "}"}
        },
        "math500": {
            "template": PromptTemplate(
                template = in_context_example_token + qwen_template,
                column_token_map = in_context_placeholder_token_map,
                ice_token = in_context_example_token,
            ),
            "input_columns": ["problem"],
            "output_column": "answer",
            "train_split": "train",
            "test_split": "test",
            "data_path": "HuggingFaceH4/MATH-500",
            "subset_name": "main",
            # "data_processor": lambda example: {'reasoning': example['answer'].split("#### ")[0].replace(',', '').strip(), 'answer': example['answer'].split("#### ")[1].replace(',', '').strip()}
        },
        # "theoremQA": {
        #     "template": PromptTemplate(
        #         template = in_context_example_token + qwen_template,
        #         column_token_map = in_context_placeholder_token_map,
        #         ice_token = in_context_example_token,
        #     ),
        #     "input_columns": ["Question"],
        #     "output_column": "Answer",
        #     "train_split": "train",
        #     "test_split": "test",
        #     "data_path": "TIGER-Lab/TheoremQA",
        #     "subset_name": "main",
        # }
    }
    
    task_names = list(data_map.keys())
    infer_model_names = ['Qwen/Qwen2.5-1.5B-Instruct']
    retrieve_model_name = 'Qwen/Qwen3-Embedding-4B'
    icl_shot = 8
    batch_size = 4
    seeds = [1]

    for model_path in infer_model_names:
        accelerator = Accelerator()
        for seed in seeds:
            for task_name in task_names:
                output_json_filepath = f'results/{task_name}_{model_path.split("/")[-1]}_{retrieve_model_name.split("/")[-1]}_{icl_shot:02d}shot'
                os.makedirs(output_json_filepath, exist_ok=True)
                dataset = load_dataset(data_map[task_name]["data_path"], data_map[task_name]["subset_name"])
                # dataset[data_map[task_name]["train_split"]] = dataset[data_map[task_name]["train_split"]].map(data_map[task_name]["data_processor"])
                # dataset[data_map[task_name]["test_split"]] = dataset[data_map[task_name]["test_split"]].map(data_map[task_name]["data_processor"])
                data = DatasetReader(dataset, input_columns=data_map[task_name]["input_columns"], output_column=data_map[task_name]["output_column"], ds_size=10)
                topk_retriever = TopkRetriever(data, ice_num=icl_shot, sentence_transformers_model_name=retrieve_model_name, tokenizer_name=retrieve_model_name, 
                                               batch_size=1, index_split=data_map[task_name]["train_split"], test_split=data_map[task_name]["test_split"], accelerator=accelerator)
                # cone_retriever = ConERetriever(data, ice_num=icl_shot, candidate_num=30, sentence_transformers_model_name=retrieve_model_name, tokenizer_name=retrieve_model_name, model_tokenizer_name=model_path, ce_model_name=model_path, ice_template=data_map[task_name]["template"], select_time=candidate_num, seed=seed, batch_size=batch_size, test_split='test', accelerator=accelerator)
                # inferencer = PPLInferencer(model_name=model_path, tokenizer=model_path, output_json_filepath=output_json_filepath, batch_size=batch_size, accelerator=accelerator)
                inferencer = GenInferencer(model_name=model_path, tokenizer=model_path, output_json_filepath=output_json_filepath, batch_size=batch_size, accelerator=accelerator)
                topk_predictions = inferencer.inference(topk_retriever, ice_template=data_map[task_name]["template"], output_json_filename=f'topk_seed_{seed}')
                topk_predictions = [processing_answer(pred.split('\n\n')[0]) for pred in topk_predictions]
                score = AccEvaluator().score(predictions=topk_predictions, references=data.references)
                print(score)