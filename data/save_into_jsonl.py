import json

def save_into_jsonl(data_path_json):
    with open(data_path_json, "r") as f:
        data = json.load(f)
    lines = []
    for key, line in data.items():
        if "cot_with_tag" in line:
            lines.append({
                "id": key,
                "cot_wo_tag": " ".join(line["cot_wo_tag"]),
                "cot_with_tag": " ".join(line["cot_with_tag"]),
                "question": line["query"],
                "answer": line["choices"][line["correct_choice_idx"]],
                "cot_with_tag_correct": line["cot_with_tag_correct"],
                "cot_wo_tag_correct": line["cot_wo_tag_correct"],
                # "cot_with_tag_negatives": line["cot_with_tag_variants"]["EntitySwap"] + line["cot_with_tag_variants"]["Paraphrasing"],
                # "cot_wo_tag_negatives": line["cot_wo_tag_variants"]["EntitySwap"] + line["cot_wo_tag_variants"]["Paraphrasing"]
            })
    
    with open(data_path_json.replace(".json", ".jsonl"), "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
            
if __name__ == "__main__":
    data_path = "./data/MATH500/test_perturbed.json"