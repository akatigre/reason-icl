from typing import List

def generate_negatives_variations(llm, processor, sampling_params, prompt: str, num_variants: int = 30):
    system_message = (
        "You are a helpful assistant tasked with rewriting a given prompt in a different way."
    )
    entity_swap_prompt = "Replace key entities in the prompt with plausible but different entities.\n"
    paraphrasing_prompt = "Rephrase the sentence while preserving the original meaning.\n"    
    user_message = f"METHOD \n Given prompt: {prompt}\n Modified prompt:"
    # Apply chat template with thinking enabled
    
    messages = lambda method: [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message.replace("METHOD", method)}
    ]
    
    inputs = [processor.apply_chat_template(
        messages(entity_swap_prompt),
        tokenize=False,
        add_generation_prompt=True
    ) for _ in range(num_variants // 2)] + [processor.apply_chat_template(
        messages(paraphrasing_prompt),
        tokenize=False,
        add_generation_prompt=True
    ) for _ in range(num_variants // 2)]

    outputs = llm.generate(inputs, sampling_params)

    # Parse JSON from the first output (they should all be identical in structure)
    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses


def generate_positives_variations(llm, processor, sampling_params, original_cot: str, random_cots: List[str]):
    system_message = (
        "You are a helpful assistant tasked with rewriting a given prompt into a similar variant."
    )
    user_message = (
        "From the original sentence, extract the main entities and words, and inject these words naturally into the target sentence.\n"
        f"original: {original_cot}\n"
        f"target: RANDOM_COT\n"
        "Return the modified target sentence."
    )

    messages = [[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message.replace("RANDOM_COT", random)}
    ] for random in random_cots]
    
    text = [processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for message in messages]
    outputs = llm.generate(text, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses