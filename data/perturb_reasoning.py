import re
import json
from argparse import ArgumentParser
from tqdm import tqdm
import faiss
from transformers import AutoTokenizer, AutoModel

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize vLLM tokenizer and engine for Qwen3-32B
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_tokens=4096
)
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")


def generate_negatives_variations(prompt: str, num_variants: int = 30):
    system_message = (
        "You are a helpful assistant tasked with rewriting a given prompt into multiple variants."
        " For each variant, create a variation of the original prompt using the following methods:\n"
        "- Entity Swap: Replace key entities in the prompt with plausible but different entities.\n"
        "- Paraphrasing: Rephrase the sentence while preserving the original meaning.\n"
        "Return the output as a JSON array of objects, each with keys 'EntitySwap' and 'Paraphrasing'."
    )
    user_message = f"Prompt: {prompt}"

    # Apply chat template with thinking enabled
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Duplicate prompt to get multiple outputs
    inputs = [text] * num_variants
    outputs = llm.generate(inputs, sampling_params)

    # Parse JSON from the first output (they should all be identical in structure)
    raw = outputs[0].outputs[0].text.strip()
    try:
        variants = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from model output: {raw}")

    entity_swap_list = [v["EntitySwap"] for v in variants]
    paraphrasing_list = [v["Paraphrasing"] for v in variants]
    return entity_swap_list, paraphrasing_list


def generate_positives_variations(original_cot: str, random_cot: str):
    system_message = (
        "You are a helpful assistant tasked with rewriting a given prompt into a similar variant."
        " From the target sentence, maintain the logical structure but change the entities and words"
        " to make it similar to the original sentence."
    )
    user_message = (
        f"original_cot: {original_cot}\n"
        f"target_cot: {random_cot}\n"
        "Return the modified sentence."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    outputs = llm.generate([text], sampling_params)
    return outputs[0].outputs[0].text.strip()


if __name__ == "__main__":
    from parser import get_args

    args = get_args()
    remove_tags = lambda text: re.sub(r'</?[^>]+>', '', text)

    tokenizer_embed = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Embedding")
    model_embed = AutoModel.from_pretrained("Qwen/Qwen3-4B-Embedding")
    model_embed.eval()

    # Load data
    with open(args.predictions, "r") as f:
        data = json.load(f)
        
    for p_id, item in tqdm(data.items()):
        cot_with_tag = remove_tags(" ".join(item["cot_with_tag"]))
        cot_without_tag = remove_tags(" ".join(item["cot_wo_tag"]))

        if "cot_with_tag_variants" not in item:
            es, para = generate_negatives_variations(cot_with_tag, num_variants=30)
            item["cot_with_tag_variants"] = {"EntitySwap": es, "Paraphrasing": para}

        if "cot_wo_tag_variants" not in item:
            es, para = generate_negatives_variations(cot_without_tag, num_variants=30)
            item["cot_wo_tag_variants"] = {"EntitySwap": es, "Paraphrasing": para}

        if "positives_with_tag" not in item["cot_with_tag_variants"]:
            positives_with_tag = []
            positives_wo_tag = []

            # Compute lowest-similarity PIDs via CLIP
            sims = pid_clip_dict[p_id] @ clip_embeds_array.T
            lowest_sim_pids = sims.argsort(axis=-1)[:30]

            for pid in lowest_sim_pids:
                cot_with_tag_rand = remove_tags(" ".join(data[pid]["cot_with_tag"]))
                cot_wo_tag_rand = remove_tags(" ".join(data[pid]["cot_wo_tag"]))

                pos_with = generate_positives_variations(cot_with_tag, cot_with_tag_rand)
                positives_with_tag.append(pos_with)

                pos_wo = generate_positives_variations(cot_without_tag, cot_wo_tag_rand)
                positives_wo_tag.append(pos_wo)

            item["cot_with_tag_variants"]["positives_with_tag"] = positives_with_tag
            item["cot_wo_tag_variants"]["positives_wo_tag"] = positives_wo_tag

            # Save perturbed results
            output_path = args.predictions.replace(".json", "_perturbed.json")
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)
