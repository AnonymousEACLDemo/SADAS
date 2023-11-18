from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import os

class norm_rule_model:

	def __init__(self, model_name, path, output_size=11):
		self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
		self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=output_size)
		self.model.load_state_dict(torch.load(os.path.join(path, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)
		self.norm_type_chinese_mapping = {
            "apology": "道歉",
            "greetings": "问候",
            "request": "请求"
        }
		self.norm_rule_to_masking = {
            "apology": [1,1,1,0,0,0,0,0,0,0,0],
            "greetings": [0,0,0,1,1,1,0,0,0,0,0],
            "request": [0,0,0,0,0,0,1,1,1,1,1],
        }
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = self.model.to(self.device)

	def execute(self, context_list, voliate_norm):
		text = "违反规范：%s " % self.norm_type_chinese_mapping[voliate_norm]
		for index, utterance in enumerate(context_list):
			if index % 2 == 0:
				text += "A: %s " % utterance
			else:
				text += "B: %s " % utterance
		text = text.strip()

		encodings = self.tokenizer([text], padding=True, truncation=True)
		input_ids = torch.tensor(encodings['input_ids'][0]).long().unsqueeze(0)
		input_ids = input_ids.to(self.device)

		logit_mask = torch.tensor(self.norm_rule_to_masking[voliate_norm]).float().unsqueeze(0)
		logit_mask = logit_mask.to(self.device)

		logits = self.model(input_ids=input_ids)[0]
		_, preds = torch.max(logits * logit_mask - 1e9 * (1. - logit_mask), dim=-1)

		return preds[0].item()


if __name__ == "__main__":
	model = norm_rule_model('bert-base-chinese', 'bert_all_data')
	print(model.execute(["你怎么一直瞪着我，也不说话，吓死我了！", "我又没盯着你看。"], 'apology'))
