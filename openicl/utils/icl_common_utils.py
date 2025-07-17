from torch.utils.data import DataLoader
from openicl.icl_retriever import BaseRetriever
from typing import List, Union, Optional
from openicl import PromptTemplate
from accelerate import Accelerator


def get_dataloader(datalist: List[List], batch_size: int) -> DataLoader:
    dataloader = DataLoader(datalist, batch_size=batch_size)
    return dataloader


def get_generation_prompt_list_from_retriever_indices(ice_idx_list: List[List[int]], retriever: BaseRetriever,
                                                      tokenizer, gen_field_replace_token: str,
                                                      max_model_token_num: Optional[int] = None,
                                                      ice_template: Optional[PromptTemplate] = None,
                                                      prompt_template: Optional[PromptTemplate] = None):
    prompt_list = []
    for idx, ice_idx in enumerate(ice_idx_list):
        ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
        prompt = retriever.generate_prompt_for_generate_task(idx, ice, gen_field_replace_token=gen_field_replace_token,
                                                             ice_template=ice_template, prompt_template=prompt_template)
        prompt_list.append(prompt)
    return prompt_list


def get_input_token_num(tokenizer, input):
    return len(tokenizer(input, verbose=False)['input_ids'])
