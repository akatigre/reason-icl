
import json

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from utils.math_utils import grade_answer_sympy
from utils.parse_utils import extract_last_boxed_text

output_json_filepath = "./results/gsm8k_Qwen2.5-1.5B-Instruct_Qwen3-Embedding-4B_08shot/topk_seed_1.json"
with open(output_json_filepath, 'r') as f:
    topk_predictions = json.load(f)
    
for k, pred in topk_predictions.items():
    pred["parsed_prediction"] = extract_last_boxed_text(pred['prediction']) # args.retrieve_model_path
    
list_of_tuples = [
    grade_answer_sympy(pred['parsed_prediction'], pred['origin_answer']) for pred in topk_predictions.values()
]

tf = [t[0] for t in list_of_tuples]
acc = sum(tf) / len(tf)
print(f"Task: {output_json_filepath}, Acc: {acc}")