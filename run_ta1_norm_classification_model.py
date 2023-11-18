from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import os

class norm_classification_model:

	def __init__(self, model_name, path, output_size=6):
		self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
		self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=output_size)
		self.model.load_state_dict(torch.load(os.path.join(path, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = self.model.to(self.device)
		self.norm_type_mapping = {
		    0: "OK",
		    1: "apology",
		    2: "greetings",
		    3: "request",
		    4: "criticism",
		    5: "persuasion"
		}

	def execute(self, context_list, related_norm, is_voliation, prob):
		text = "GPT3 Detected Norm %s Violation %s Probility %.2f " % (related_norm, is_voliation, prob)
		for index, utterance in enumerate(context_list):
			if index % 2 == 0:
				text += "A: %s " % utterance
			else:
				text += "B: %s " % utterance
		text = text.strip()

		encodings = self.tokenizer([text], padding=True, truncation=True)
		input_ids = torch.tensor(encodings['input_ids'][0]).long().unsqueeze(0)
		input_ids = input_ids.to(self.device)

		logits = self.model(input_ids=input_ids)[0]
		_, preds = torch.max(logits, dim=-1)

		return self.norm_type_mapping[preds[0].item()]


if __name__ == "__main__":
	model = norm_classification_model('bert-base-chinese', 'models/bert_ta1_norm_classification')
	print(model.execute(["你怎么一直瞪着我，也不说话，吓死我了！", "我又没盯着你看。"], 'No norm', 'No', 0.0))
