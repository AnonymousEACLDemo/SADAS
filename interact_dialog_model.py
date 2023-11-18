import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


def score_response_candidates(context, response, model, tokenizer):
    texts = (context, response)
    inputs = tokenizer(*texts, padding=True, max_length=500, truncation=True, return_tensors="pt")
    inputs.to(model.device)
    outputs = model(**inputs)
    predictions = softmax(outputs.logits, dim=-1)
    relevant_score = predictions.data[:, 1][0]
    return round(relevant_score.cpu().detach().item(), 4)


def gpt2_causal_text_generation(dialogue_history, model, tokenizer, classifier, classifier_tokenizer):
    gen_kwargs = {
        "max_length": 300,
        "length_penalty": 5.0,
        "num_beams": 10,
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
    }
    preceding_utterance = dialogue_history[-1]
    for utterance in dialogue_history[-4:-1]:
        context = utterance + tokenizer.sep_token + preceding_utterance + tokenizer.sep_token + "[RED]"
        context_encoding = tokenizer(context, max_length=200, truncation=True, return_tensors='pt')
        context_encoding.to(model.device)
        context_length = context_encoding["input_ids"].shape[-1]
        gen_kwargs["max_length"] = context_length+100
        generated_tokens = model.generate(context_encoding["input_ids"], **gen_kwargs,
                                          pad_token_id=tokenizer.pad_token_id,
                                          eos_token_id=tokenizer.sep_token_id)
        decoded_preds = tokenizer.decode(generated_tokens[:, context_encoding["input_ids"].shape[-1]:][0],
                                         skip_special_tokens=False)
        decoded_preds = decoded_preds.replace(" ", "")
        relevant_score = score_response_candidates(context, decoded_preds, classifier, classifier_tokenizer)
        print(f"response: {decoded_preds} {relevant_score}")
    if len(dialogue_history) < 2:
        context = preceding_utterance + tokenizer.sep_token
        context_encoding = tokenizer(context, max_length=200, truncation=True, return_tensors='pt')
        context_encoding.to(model.device)
        context_length = context_encoding["input_ids"].shape[-1]
        gen_kwargs["max_length"] = context_length + 100
        generated_tokens = model.generate(context_encoding["input_ids"], **gen_kwargs,
                                          pad_token_id=tokenizer.pad_token_id,
                                          eos_token_id=tokenizer.sep_token_id)
        decoded_preds = tokenizer.decode(generated_tokens[:, context_encoding["input_ids"].shape[-1]:][0],
                                         skip_special_tokens=False)
        decoded_preds = decoded_preds.replace(" ", "")
        relevant_score = score_response_candidates(context, decoded_preds, classifier, classifier_tokenizer)
        print(f"response: {decoded_preds} {relevant_score}")


# load generator model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator_tokenizer = AutoTokenizer.from_pretrained("models/gpt2-chinese-ccu-annotatedData")
generator = AutoModelForCausalLM.from_pretrained("models/gpt2-chinese-ccu-annotatedData").to(device)
generator.eval()
# load classifier model
classifier_tokenizer = AutoTokenizer.from_pretrained("models/chinese-roberta-wwm-dialog")
classifier = AutoModelForSequenceClassification.from_pretrained("models/CI_classifier").to(device)
classifier.eval()
"""
dialogue_history = [
    "你好。",
    "你好，欢迎过来。",
    "你们的经理怎么到现在还没来。",
    "可能马上就到了。",
    "你们迟到太久了，这样太没礼貌了。"
]

dialogue_history = [
    "王工，我发现我们的系统响应时间越来越长了。",
    "我看了下数据，确实是的。",
]

dialogue_history = [
    "你怎么现在才把姚总要的文件送过来？你知不知道姚总为了今天的会议花了多少心血？",
]
"""
dialogue_history = [
    "你怎么现在才把姚总要的文件送过来？你知不知道姚总为了今天的会议花了多少心血？",
]
gpt2_causal_text_generation(dialogue_history, generator, generator_tokenizer, classifier, classifier_tokenizer)