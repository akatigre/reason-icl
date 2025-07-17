from openicl import PromptTemplate

llm_template = "Question: </Q> \n Please think step by step and output your answer in \\boxed{}. </A> \n"
vlm_template = "Question: </Q> \n Image: </I> \n Please think step by step and output your answer in \\boxed{}. </A> \n"
in_context_placeholder_token_map = {'question': '</Q>', 'reason_answer': '</A>'}

DATA_MAP = {
    "gsm8k": {
        "template": PromptTemplate(
            template=llm_template,
            column_token_map=in_context_placeholder_token_map,
        ),
        "input_columns": ["question"],
        "output_column": "reason_answer",
        "train_split": "train",
        "test_split": "test",
        "data_path": "openai/gsm8k",
        "subset_name": "main",
        "data_processor": lambda example: {
            'reason_answer': example['answer'].split("#### ")[0].replace(',', '').strip() + " The answer is \\boxed{" + example['answer'].split("#### ")[1].replace(',', '').strip() + "}",
            'answer': example['answer'].split("#### ")[1].replace(',', '').strip()
        }
    },
    # "math500": {
    #     "template": PromptTemplate(
    #         template=in_context_example_token + qwen_template,
    #         column_token_map=in_context_placeholder_token_map,
    #         ice_token=in_context_example_token,
    #     ),
    #     "input_columns": ["problem"],
    #     "output_column": "reason_answer",
    #     "train_split": "train",
    #     "test_split": "test",
    #     "data_path": "./data/MATH500",
    #     "subset_name": "main",
    #     "data_processor": lambda example: {'reason_answer': example["solution"] + " The answer is \\boxed{" + example["answer"] + "}"},
    # },
    # "aokvqa": {
    #     "template": PromptTemplate(
    #         template=IN_CONTEXT_EXAMPLE_TOKEN + qwenvl_template,
    #         column_token_map=in_context_placeholder_token_map,
    #         ice_token=IN_CONTEXT_EXAMPLE_TOKEN,
    #     ),
    #     "input_columns": ["question", "image"],
    #     "output_column": "reason_answer",
    #     "train_split": "train",
    #     "test_split": "val",
    #     "data_path": "openai/gsm8k",
    #     "data_processor": lambda example, split: {
    #         "question": f"{example['question']}. Answer from one of the choices {example['choices']}. When the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase.",
    #         "image": get_coco_path(split, example['image_id'], "./data/coco"), 
    #         "answer": example['choices'][ example['correct_choice_idx'] ] if split == "train" else example["difficult_direct_answer"]}
    # }
}

# from utils.aokvqa_utils import load_aokvqa, get_coco_path
# aokvqa_dir = "./data/aokvqa/"
# assert args.split in ["val", "train"]
# dataset = load_aokvqa(aokvqa_dir, args.split)
# if isinstance(dataset, list):
#     dataset = {dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }
    
# image_id = data['image_id']    
# image_path = get_coco_path(split, image_id, coco_dir)

# question = data['question']
# choices = data['choices']
# if split == "test":
#     question_id = data['question_id']
#     correct_choice_idx = None
#     correct_choice = None
#     direct_diffucult = data['difficult_direct_answer']
#     meta_data = {
#         "question_id": question_id,
#         "direct_diffucult": direct_diffucult
#     }
# else:
#     correct_choice_idx = data['correct_choice_idx']
#     correct_choice = data['choices'][ correct_choice_idx ]
#     meta_data = {
#         # "rationales": rationales,
#         "answer": correct_choice
#     }
