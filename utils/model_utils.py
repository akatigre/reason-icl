from vllm import LLM, SamplingParams
from transformers import AutoProcessor

def load_model(model_path, max_model_token_num, tensor_parallel_size=4, multi_modal=False):
    llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size
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
        max_tokens=max_model_token_num,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.padding_side = "left"
    return llm, sampling_params, processor