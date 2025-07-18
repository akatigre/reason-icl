# Set your OpenAI API key
import os
import re
import json
from argparse import ArgumentParser
from openai import OpenAI
import dotenv
from pydantic import BaseModel
from tqdm import tqdm
import random
dotenv.load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=openai_api_key
)

def generate_negatives_variations(prompt: str, num_variants: int = 30):
    system_message = (
        "You are a helpful assistant tasked with rewriting a given prompt into multiple variants."
        " For each variant, create a variation of the original prompt using following methods:\n"
        "- Entity Swap: Replace key entities in the prompt with plausible but different entities.\n"
        "- Paraphrasing: Rephrase the sentence while preserving the original meaning.\n"
        "Return each variation as a numbered list. Each variation should be unique and different from the original prompt."
    )
    class Variation(BaseModel):
        EntitySwap: str
        Paraphrasing: str

    user_message = f"Prompt: {prompt}"

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.9,
        n=num_variants,
        response_format=Variation,
        frequency_penalty=1
    )
    entity_swap_list = []
    paraphrasing_list = []
    for choice in response.choices:
        parsed = choice.message.parsed
        entity_swap_list.append(parsed.EntitySwap)
        paraphrasing_list.append(parsed.Paraphrasing)
    return entity_swap_list, paraphrasing_list

def generate_positives_variations(original_cot: str, random_cot: str):
    system_message = (
        "You are a helpful assistant tasked with rewriting a given prompt into a similar variant."
        "From target sentence, maintain the logical structure but change the entities and words to make it similar to the original sentence.\n"
    )
    user_message = (
        f"original_cot: {original_cot}\n"
        f"target_cot: {random_cot}\n"
        "Return the modified sentence."
    )

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    from parser import get_args
    import pickle
    import numpy as np
    args = get_args()

    remove_tags = lambda text: re.sub(r'</?[^>]+>', '', text)
    with open(args.predictions, "r") as f:
        data = json.load(f)
    with open(args.embedding_output_dir, "rb") as f:
        clip_embeds = pickle.load(f)
        
    pid_cot_dict = {p_id: clip_embeds[p_id]["meta"]["cot_wo_tag"] for p_id in data.keys()}
    pid_clip_dict = {p_id: clip_embeds[p_id]["embeddings"]["clip"] for p_id in data.keys()}
    clip_embeds = np.array(list(pid_clip_dict.values()))
    
    for p_id, item in tqdm(data.items()):
        cot_with_tag, cot_without_tag = " ".join(item["cot_with_tag"]), " ".join(item["cot_wo_tag"])
        cot_with_tag = remove_tags(cot_with_tag)
        if not "cot_with_tag_variants" in data.get(p_id, {}):
            
            EntitySwap, Paraphrasing = generate_negatives_variations(cot_with_tag, num_variants=30)
            data[p_id]["cot_with_tag_variants"] = {
                "EntitySwap": EntitySwap,
                "Paraphrasing": Paraphrasing
            }
            
        if not "cot_wo_tag_variants" in data.get(p_id, {}):
            
            EntitySwap, Paraphrasing = generate_negatives_variations(cot_without_tag, num_variants=30)
            data[p_id]["cot_wo_tag_variants"] = {
                "EntitySwap": EntitySwap,
                "Paraphrasing": Paraphrasing
            }
            
        if not "positives_with_tag" in data[p_id]["cot_with_tag_variants"]:
            positives_with_tag = []
            positives_wo_tag = []

            lowest_sim_pids = (pid_clip_dict[p_id] @ clip_embeds.T).argsort(dim=-1, descending=False)[:30]

            #! Get negatives by CLIP and change the entities and words to make it similar to the original sentence.
            for pid in lowest_sim_pids:
                cot_with_tag_random, cot_without_tag_random = " ".join(data[pid]["cot_with_tag"]), " ".join(data[pid]["cot_wo_tag"])
                cot_with_tag_random = remove_tags(cot_with_tag_random)
                positive_variants = generate_positives_variations(cot_with_tag, cot_with_tag_random)
                
                positives_with_tag.append(positive_variants)
                positive_variants = generate_positives_variations(cot_without_tag, cot_without_tag_random)
                positives_wo_tag.append(positive_variants)
            data[p_id]["cot_with_tag_variants"].update({
                "positives_with_tag": positives_with_tag,
            })
            data[p_id]["cot_wo_tag_variants"].update({
                "positives_wo_tag": positives_wo_tag
            })
            with open(args.predictions.replace(".json", "_perturbed.json"), "w") as f:
                json.dump(data, f, indent=4)
            