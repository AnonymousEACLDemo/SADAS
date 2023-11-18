import json
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from m2m100 import M2M100ForConditionalGeneration
from tqdm import tqdm
import logging
import torch

def generated_response(input_data, model, tokenizer, device):
    generated_responses = []
    for line in tqdm(input_data["context"]):
        input_ids = tokenizer.encode(line, truncation=True, return_tensors='pt')
        input_ids = input_ids.to(device)
        generated_output = model.generate(
            input_ids,
            max_length=200,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True)

        context = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        response = tokenizer.decode(generated_output[0], skip_special_tokens=False)
        response = response.replace("<s>","").replace("</s>","").replace("__en__", "")

        dialogue_history = context.split("</s> <s>")
        one_example = {}
        one_example["dialogue_history"] = dialogue_history
        one_example["generated_response"] = response
        generated_responses.append(one_example)
    return generated_responses

def main(context):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logging.debug(f"GPU detected. Running TA2 component (dialogue manager) in GPU")
    else:
        logging.debug(f"No GPU detected. Running TA2 component (dialogue manager) in CPU")
        
    GENERATOR_MODEL = "model/"

    config = AutoConfig.from_pretrained(GENERATOR_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    
    model = M2M100ForConditionalGeneration.from_pretrained(GENERATOR_MODEL, config=config)
    model.to(device)
    
    input_data = {
        "context" : [context]
    }

    generated_responses = generated_response(input_data, model, tokenizer, device)
    return generated_responses