import os
import io
import sys
import base64
from PIL import Image
import logging
import argparse

from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from rich.logging import RichHandler
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import read_json, save_json, evaluate_code
from utils.parse_utils import extract_last_boxed_text
from evaluation.build_query import create_query_data

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--test_split_name', type=str, required=True, choices=['testmini', 'test'])
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    # Local Model
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', choices=['Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-32B-Instruct'])
    # query
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    return args


def main():
    logging.info("MathVista: Generating Responses - Start")
    args = parse_args()
    # load data
    dataset_name = 'AI4Math/MathVista'
    logging.info(f"Loading dataset {dataset_name}, split {args.test_split_name}...")
    data_list = load_dataset(dataset_name, split=args.test_split_name)
    # Convert Hugging Face data into dictionary to match local data format
    # TODO: Convert scripts not to depend on dictionary .json format. Update to use .jsonl format
    data = {item['pid']: item for item in data_list}

    # load or create query data
   
    logging.info("Creating new query...")

    caption_data = {}
    if args.use_caption:
        caption_file = args.caption_file
        if os.path.exists(caption_file):
            logging.info(f"Reading {caption_file}...")
            try:
                caption_data = read_json(caption_file)["texts"]
                logging.info("Caption data loaded.")
            except Exception as e:
                logging.info("Caption data not found!! Please Check.")

    ocr_data = {}
    if args.use_ocr:
        ocr_file = args.ocr_file
        if os.path.exists(ocr_file):
            logging.info(f"Reading {ocr_file}...")
            try:
                ocr_data = read_json(ocr_file)["texts"]
                logging.info("OCR data loaded.")
            except Exception as e:
                logging.info("OCR data not found!! Please Check.")

    query_data = create_query_data(data, caption_data, ocr_data, args)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=2,
        limit_mm_per_prompt={"image": 5, "video": 5},
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=1024,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"

    logging.info(f"Model loaded.")

    full_pids = list(data.keys())

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file_path):
        logging.info("Results already exist.")
        logging.info(f"Reading {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}

    test_pids = full_pids

    logging.info(f"Number of test problems to run: {len(test_pids)}")
    inputs = []
    for i, problem_id in enumerate(tqdm(test_pids)):
        problem: dict = data[problem_id].copy()
        
        query = query_data[problem_id].replace(" Solution:", "") + "\n Output the final answer inside \\boxed{}."
        
        image_path = str(Path("/data/yoonjeon_workspace/") / problem["image"])
        problem.pop('decoded_image')
        results[problem_id] = problem
        results[problem_id]['query'] = query
        if "response" in results[problem_id]:
            results[problem_id]['answer'] = extract_last_boxed_text(results[problem_id]['response'])
        else:
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
            
            inputs.append(llm_inputs)
            if args.debug and i == 3:
                break

    response = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(response):
        response = output.outputs[0].text
        if args.shot_type == 'solution':
            results[test_pids[i]]['response'] = response
            results[test_pids[i]]['execution'] = ''
            results[test_pids[i]]['error'] = ''
        else:
            output, error = evaluate_code(response)
            results[test_pids[i]]['response'] = response
            results[test_pids[i]]['execution'] = output
            results[test_pids[i]]['error'] = str(error)
    save_json(results, output_file_path)

    logging.info("MathVista: Generating Responses - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
