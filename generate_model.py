import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from constant import *


class PhraseModel(object):
    def __init__(self):
        # load generator model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        generator_tokenizer_path = PROJECT_ABSOLUTE_PATH + "/models/gpt2-chinese-ccu-annotatedData_criticism_2.0"
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_path)

        self.cri_generator = AutoModelForCausalLM.from_pretrained(generator_tokenizer_path).to(self.device)
        self.cri_generator.eval()

        apo_generator_path = PROJECT_ABSOLUTE_PATH + "/models/gpt2-chinese-ccu-annotatedData_apo_2.0"
        self.apo_generator = AutoModelForCausalLM.from_pretrained(apo_generator_path).to(self.device)
        self.apo_generator.eval()

        # load classifier model
        roberta_path = PROJECT_ABSOLUTE_PATH + "/models/chinese-roberta-wwm-dialog"
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(roberta_path)
        classifier_path = PROJECT_ABSOLUTE_PATH + "/models/CI_classifier"
        self.classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path).to(self.device)
        self.classifier.eval()

    def score_response_candidates(self, context, response):
        texts = (context, response)
        inputs = self.classifier_tokenizer(*texts, padding=True, max_length=500, truncation=True, return_tensors="pt")
        inputs.to(self.classifier.device)
        outputs = self.classifier(**inputs)
        predictions = softmax(outputs.logits, dim=-1)
        relevant_score = predictions.data[:, 1][0]
        return round(relevant_score.cpu().detach().item(), 4)

    def seperate_gpt2_causal_text_generation(self, dialogue_history):
        beam_gen_kwargs = {
            "max_length": 300,
            "length_penalty": 5.0,
            "num_beams": 10,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
        }
        sample_gen_kwargs = {
            "max_length": 300,
            "top_p": 0.8,
            "do_sample": True
        }
        norm_category = ["[APO]", "[CRI]"]  # , "[GRE]", "[REQ]", "[PSU]"]
        preceding_utterance, decoded_pred_list = dialogue_history[-1], []

        for utterance in dialogue_history[-3:-1]:
            for norm in norm_category:
                if norm == "[APO]":
                    model = self.apo_generator
                else:
                    model = self.cri_generator
                context = utterance + self.generator_tokenizer.sep_token + preceding_utterance + self.generator_tokenizer.sep_token + norm
                context_encoding = self.generator_tokenizer(context, max_length=200, truncation=True, return_tensors='pt')
                context_encoding.to(model.device)
                context_length = context_encoding["input_ids"].shape[-1]
                sample_gen_kwargs["max_length"] = context_length + 100
                beam_gen_kwargs["max_length"] = context_length + 100
                generated_tokens = model.generate(context_encoding["input_ids"], **sample_gen_kwargs,
                                                  pad_token_id=self.generator_tokenizer.pad_token_id,
                                                  eos_token_id=self.generator_tokenizer.sep_token_id)
                decoded_preds = self.generator_tokenizer.decode(generated_tokens[:, context_encoding["input_ids"].shape[-1]:][0],
                                                 skip_special_tokens=True)
                decoded_preds = decoded_preds.replace(" ", "")
                # relevant_score = score_response_candidates(context, decoded_preds, classifier, classifier_tokenizer)
                decoded_pred_list.append(decoded_preds)
                print(f"response: {norm} {decoded_preds} ")

        # len(dialogue_history) < 2:
        for norm in norm_category:
            if norm == "[APO]":
                model = self.apo_generator
            else:
                model = self.cri_generator
            context = dialogue_history[-1] + self.generator_tokenizer.sep_token + norm
            context_encoding = self.generator_tokenizer(context, max_length=200, truncation=True, return_tensors='pt')
            context_encoding.to(model.device)
            context_length = context_encoding["input_ids"].shape[-1]
            sample_gen_kwargs["max_length"] = context_length + 100
            beam_gen_kwargs["max_length"] = context_length + 100
            generated_tokens = model.generate(context_encoding["input_ids"], **sample_gen_kwargs,
                                              pad_token_id=self.generator_tokenizer.pad_token_id,
                                              eos_token_id=self.generator_tokenizer.sep_token_id)
            decoded_preds = self.generator_tokenizer.decode(generated_tokens[:, context_encoding["input_ids"].shape[-1]:][0],
                                             skip_special_tokens=True)
            decoded_preds = decoded_preds.replace(" ", "")
            # relevant_score = score_response_candidates(context, decoded_preds, classifier, classifier_tokenizer)
            decoded_pred_list.append(decoded_preds)
            print(f"response: {norm} {decoded_preds} ")

        return decoded_pred_list


if __name__ == "__main__":
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
    generate_model = PhraseModel()
    generate_model.seperate_gpt2_causal_text_generation(dialogue_history)




