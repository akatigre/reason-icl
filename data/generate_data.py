import os
import argparse
import logging
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from utils import read_json, save_json, evaluate_code
from evaluation.build_query import create_query_data
from utils.parse_utils import extract_last_boxed_text
from reasoning import MyCoT
from utils.vqa_utils import EvalAIAnswerProcessor

eval_ai_processor = EvalAIAnswerProcessor()
def load_model(model_path, multi_modal=False):
    llm = LLM(
            model=model_path,
            tensor_parallel_size=4
        )
    if multi_modal:
        temp = 0.1
        top_p = 0.001
        top_k = 50
        repetition_penalty = 1.05
    else:
        temp = 0.7
        top_p = 0.8
        top_k = 20
        repetition_penalty = 1
    sampling_params = SamplingParams(
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_tokens=9182,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return llm, sampling_params, processor

def save_response(response, problem_ids, results, save_as, shot_type='solution'):
    for i, (problem_id, output) in enumerate(zip(problem_ids, response)):
        response = output.outputs[0].text
        results[problem_id][save_as] = response
        gt_answer = results[problem_id]['answer']
        ext_answer = extract_last_boxed_text(response)
        ext_answer = eval_ai_processor(ext_answer)
        if ext_answer == gt_answer:
            results[problem_id][f'{save_as}_correct'] = True
        else:
            results[problem_id][f'{save_as}_correct'] = False
    return results

    
def main(args):
    llm, sampling_params, processor = load_model(args.model_path, args.debug)
    cot = MyCoT()
    logging.info(f"Model loaded.")
    if args.dataset_name == "MathVista":
        assert args.test_split_name in ["testmini", "test"]
        data_list = load_dataset("AI4Math/MathVista", split=args.test_split_name)
        # Convert Hugging Face data into dictionary to match local data format
        data = {item['pid']: item for item in data_list}
        dataset = create_query_data(data, caption_data={}, ocr_data={}, args=args)
    elif args.dataset_name == "AOKVQA":
        from utils.aokvqa_utils import load_aokvqa
        aokvqa_dir = "./data/aokvqa/"
        assert args.split in ["val", "train"]
        dataset = load_aokvqa(aokvqa_dir, args.split)
        if isinstance(dataset, list):
            dataset = {dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }
    elif args.dataset_name == "REMI":
        from utils.remi_utils import load_remi
        assert args.split in ["val", "train"]
        dataset = load_remi(args.split)
        tasks = list(dataset.keys())
        new_dataset = {}
        for task in tasks:
            for problem_id, data in dataset[task].items():
                new_dataset[problem_id] = {
                    "question": data['question'],
                    "answer": data['label'],
                    "task": task,
                    "image_map": data['image_map']
                }
        dataset = new_dataset
    pids = list(dataset.keys())

    logging.info(f"Number of test problems to run: {len(pids)}")
    inputs_raw, inputs_cot_tag, inputs_cot_wo_tag = {}, {}, {}
    results = {}
    for retry in range(3):
        for i, problem_id in enumerate(tqdm(pids)):
            problem: dict = data[problem_id].copy()    
            query = problem[problem_id]
            image_path = str(Path("/data/yoonjeon_workspace/") / problem["image"])
            problem.pop('decoded_image')
            results[problem_id] = problem
            results[problem_id]['query'] = query
                
            messages = [{
                "role": "user", 
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_path),
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    },
                    {
                        "type": "text", 
                        "text": query
                    },
                ],
            }]
            
            if "vanilla" not in results[problem_id] or not results[problem_id].get('vanilla_correct', False):
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)
                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_inputs},
                }
                inputs_raw[problem_id] = llm_inputs
            if "cot_tag" not in results[problem_id] or not results[problem_id].get('cot_tag_correct', False):
                answer = results[problem_id]['answer']
                cot_with_tag_messages = cot.plan_with_tags(messages[0]["content"], answer=results[problem_id]['answer'])
                cot_with_tag_prompt = processor.apply_chat_template(
                    cot_with_tag_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                llm_inputs = {
                    "prompt": cot_with_tag_prompt,
                    "multi_modal_data": {"image": image_inputs},
                }
                inputs_cot_tag[problem_id] = llm_inputs
                
            if "cot_wo_tag" not in results[problem_id] or not results[problem_id].get('cot_wo_tag_correct', False):
                cot_messages = cot.plan_wo_tags(messages[0]["content"], answer=results[problem_id]['answer'])
                cot_prompt = processor.apply_chat_template(
                    cot_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                llm_inputs = {
                    "prompt": cot_prompt,
                    "multi_modal_data": {"image": image_inputs},
                }
                inputs_cot_wo_tag[problem_id] = llm_inputs
            
            if args.shot_num > 0:
                pass
            if args.debug and i == 3:
                break
        
        response_raw = llm.generate(list(inputs_raw.values()), sampling_params=sampling_params)
        response_cot_tag = llm.generate(list(inputs_cot_tag.values()), sampling_params=sampling_params)
        response_cot_wo_tag = llm.generate(list(inputs_cot_wo_tag.values()), sampling_params=sampling_params)
        
        results = save_response(response_raw, list(inputs_raw.keys()), results, "response")
        results = save_response(response_cot_tag, list(inputs_cot_tag.keys()), results, "cot_tag_raw")
        results = save_response(response_cot_wo_tag, list(inputs_cot_wo_tag.keys()), results, "cot_wo_tag_raw")

    save_json(results, os.path.join(args.output_dir, args.output_file))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, choices=['Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-32B-Instruct', "InternVL/InternVL-32B-Instruct", 'OpenGVLab/InternVL3-9B', "google/gemma-3-27b-it"])
    parser.add_argument('--dataset_name', type=str, required=True, choices=['MathVista', 'AOKVQA', 'REMI'])
    parser.add_argument('--test_split_name', type=str, required=False, default="minitest", choices=['testmini', 'test'])
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--output_dir', type=str, required=False, default="./results")
    parser.add_argument('--output_file', type=str, required=False, default="minitest.json")
    parser.add_argument('--icl_method', type=str, required=False, default='random', choices=['random', 'clip', 'vae', 'vae_tag', 'vae_wo_tag'])
    parser.add_argument('--seed', type=int, required=False, default=42)
    # query
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    args = parser.parse_args()
    main(args)