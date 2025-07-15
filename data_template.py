from openicl import PromptTemplate

qwen_template = "<|im_start|>system\nLet's think step by step and output answer in \\boxed{}.<|im_end|>\n<|im_start|>user\nQuestion: </Q> Output your answer in \\boxed{}.<|im_end|>\n<|im_start|>assistant\n </A> <|im_end|>\n"
in_context_placeholder_token_map = {'question': '</Q>', 'reason_answer': '</A>'}
IN_CONTEXT_EXAMPLE_TOKEN = '</E>'

DATA_MAP = {
    "gsm8k": {
        "template": PromptTemplate(
            template=IN_CONTEXT_EXAMPLE_TOKEN + qwen_template,
            column_token_map=in_context_placeholder_token_map,
            ice_token=IN_CONTEXT_EXAMPLE_TOKEN,
        ),
        "input_columns": ["question"],
        "output_column": "reason_answer",
        "train_split": "train",
        "test_split": "test",
        "data_path": "openai/gsm8k",
        "subset_name": "main",
        "data_processor": lambda example: {'reason_answer': example['answer'].split("#### ")[0].replace(',', '').strip() + " The answer is \\boxed{" + example['answer'].split("#### ")[1].replace(',', '').strip() + "}",
                                            'answer': example['answer'].split("#### ")[1].replace(',', '').strip()}
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
}