"""
https://github.com/allenai/aokvqa/tree/main
"""
import os
import json
import statistics
from utils.vqa_utils import EvalAIAnswerProcessor
from PIL import Image
import matplotlib.pyplot as plt
    
coco_dir = "./data/coco"

def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

def load_aokvqa_sample(data, split):
    image_id = data['image_id']    
    image_path = get_coco_path(split, image_id, coco_dir)

    question = data['question']
    choices = data['choices']
    if split == "test":
        question_id = data['question_id']
        correct_choice_idx = None
        correct_choice = None
        direct_diffucult = data['difficult_direct_answer']
        meta_data = {
            "question_id": question_id,
            "direct_diffucult": direct_diffucult
        }
    else:
        correct_choice_idx = data['correct_choice_idx']
        correct_choice = data['choices'][ correct_choice_idx ]
        meta_data = {
            # "rationales": rationales,
            "answer": correct_choice
        }
    
    content = [
        {
            "type": "image",
            "image": image_path
        },
        {
            "type": "text",
            "text": f"{question}. Answer from one of the choices {choices}. When the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
        }
    ]
    return content, meta_data

def _eval_acc(preds):
    acc = []
    wrong_qids_pred = {}
    for q in preds.keys():

        answer = preds[q]['answer']
        pred = preds[q]['pred']
        if answer == pred:
            acc.append(1)
        else:
            acc.append(0)
            wrong_qids_pred[q] = pred

    acc = sum(acc) / len(acc) * 100

    return acc, wrong_qids_pred

def ok_vqa_process_results(doc, result):
    
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        if gtAcc:
            accuracy = statistics.mean(gtAcc)
        else:
            accuracy = 0

    return {
        "exact_match": accuracy,
        "submission": {
            "image": f"{doc['question_id']}.jpg",
            "answer": resAns,
        },
    }
   

def visualize_aokvqa_sample(data, pred, split, save_as):
    
    rationales = data['rationales']
    image_id = data['image_id']    
    image_path = get_coco_path(split, image_id, coco_dir)
    question = data['question']
    correct_choice = data['choices'][ data['correct_choice_idx'] ]
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.axis('off')
    
    # Fix the attribute error - set_aspect is a method of the axis, not plt
    ax.set_aspect('equal', adjustable='box')
    
    # Add question and answer information to the visualization
    plt.figtext(0.5, 0.01, f"Question: {question}\nCorrect Answer: {correct_choice}\nPrediction: {pred}\nRationales: {''.join(rationales)}", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close(fig)  # Close the figure to free memory
    
def analyze_answer(gpt_answer, all_choices):
    """
    extracts the multiple choice answer from a long paragraph of model output if there is only one choice; otherwise, query GPT3.5 turbo to extract the choice. If the model output is short and only contains the choice, reformats the choice in the correct format e.g. (A) and returns the choice as is.

    Parameters:
    - d : data, the data containing the question and choices.
    - gpt_answer: String, the model output.
    - all_choices: List of strings, the list of all choices.

    Returns:
    - prediction, the extracted answer.
    """
    eval_ai_processor = EvalAIAnswerProcessor()
    all_choices = [eval_ai_processor(choice) for choice in all_choices]
    gpt_answer = eval_ai_processor(gpt_answer)
    intersect = list(set(all_choices).intersection(set(gpt_answer.split())))
    if gpt_answer in all_choices:
        prediction = gpt_answer
    elif len(intersect) == 1:
        prediction = intersect[0]
    elif len(intersect) != 1:
        from model_utils.run_chatgpt import ChatGPT
        gpt = ChatGPT()
        gpt_fn = gpt.create_fn()
        contents = [
            {
                "type": "text",
                "text": f"From the given text {gpt_answer}, extract the final answer from given options {all_choices}. The answer should be in a single word or phrase."
            }
        ]
        extracted_answer = gpt_fn(
            messages = [
                    {
                        "role": "user",
                        "content": contents
                    }
                ],
            )
        
        prediction = eval_ai_processor(extracted_answer)
    else:
        prediction = "unanswerable"
    return prediction

