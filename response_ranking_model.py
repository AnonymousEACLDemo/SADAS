from transformers import BertTokenizerFast
from bert_model import BertForRanking
import torch
import os
import random

generic_response = {
    0: ['不好意思', '抱歉啊'],
    1: ['我对这件事情十分抱歉', '非常抱歉，我错了'],
    2: ['非常抱歉，这是我们的错'],
    3: ['您好，很高兴见到您', '您好，幸会！'],
    4: ['您好，很高兴为您服务'],
    5: ['你好', '吃了吗？'],
    6: ['您能给我介绍一下吗？', '麻烦您一下'],
    7: ['我可能需要麻烦您一下', '劳驾您一下'],
    8: ['我需要麻烦您一下', '劳驾您一下', '不好意思，麻烦您一下'],
    9: ['我能不能占用您一点时间说点事儿'],
    10: ['我能不能占用您一点时间说点事儿', '这样是不是更好一些'],
    11: ['这样做是不是可能会有问题'],
    12: ['这样做肯定是不合适的', '还需要加把劲'],
    13: ['相信我，这样做没错的', '我有个建议', '我有个想法'],
    14: ['对于这件事，我有个小小的建议', '对于这件事，我希望您能采纳我的建议']
}

def has_repeat(v):
	ch_dict = {}
	for ch in v:
		if ch not in ch_dict:
			ch_dict[ch] = 0
		ch_dict[ch] += 1

	for v in ch_dict.values():
		if v > 3:
			return True
	return False

class res_ranking_model:

	def __init__(self, model_name, path, output_size=1, added_score_for_generic_response=1.0):
		self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
		self.model = BertForRanking.from_pretrained(model_name, num_labels=output_size, margin_value=0.5)
		self.model.load_state_dict(torch.load(os.path.join(path, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.added_score_for_generic_response = added_score_for_generic_response
		self.model = self.model.to(self.device)

	def execute(self, context_list, candidate_outputs, rule_index):
		if rule_index in generic_response:
			g_response = generic_response[rule_index]
		else:
			g_response = generic_response[0]

		try:
			output = self._inner_execute(context_list, candidate_outputs, rule_index)
		except:
			output = random.choice(g_response)

		return output

	def _inner_execute(self, context_list, candidate_outputs, rule_index):
		candidate_outputs = [v for v in candidate_outputs if len(v) <= 2.0 * len(context_list[-1]) and (not has_repeat(v))]

		original_len = len(candidate_outputs)
		if rule_index in generic_response:
			candidate_outputs += generic_response[rule_index]

		context_text = ""
		for index, utterance in enumerate(context_list):
			if index % 2 == 0:
				context_text += "A: %s " % utterance
			else:
				context_text += "B: %s " % utterance
		
		text_list = []
		for candidate in candidate_outputs:
			if len(context_list) % 2 == 0:
				text = context_text + "A: %s" % candidate
			else:
				text = context_text + "B: %s" % candidate
			text_list.append(text)

		encodings = self.tokenizer(text_list, padding=True, truncation=True)
		input_ids = torch.cat([torch.tensor(val).long().unsqueeze(0) for val in encodings['input_ids']], dim=0)
		attention_mask = torch.cat([torch.tensor(val).unsqueeze(0) for val in encodings['attention_mask']], dim=0)
		
		input_ids = input_ids.to(self.device)
		attention_mask = attention_mask.to(self.device)
		logits = self.model(stronger_input_ids=input_ids, stronger_attention_mask=attention_mask).logits
		score = logits.squeeze(-1)
		score[original_len:] += self.added_score_for_generic_response
		_, max_index = torch.max(score, dim=0)

		print('The response ranking score is:')
		for score_, response_ in zip(score, candidate_outputs) :
			print('{} -> {}'.format(score_, response_))
		return candidate_outputs[max_index.item()]


if __name__ == "__main__":
	model = res_ranking_model('bert-base-chinese', 'models/bert_response_ranking')
	context = ['陈总，您好！我是小李，我们老板在上面让我来机场接您，他在公司等您呢。', '哦，这样子。']
	candidates = ['这是一个非常具有挑战性的工作环境。我敢打赌这会让你感到压力和焦虑。你有没有考虑过与您的经理讨论这个问题呢?', '这是一个非常困难的工作环境。我不知道该怎么处理这个问题。有什么建议吗', '这是一个非常困难的情况。我很抱歉听到这个消息。您是否尝试过联系当地的慈善机构寻求帮助或提供建议?', '你这么做是不对的，你应该为自己的行为负责任，而不是为了别人的安全着想。如果这是你的错误做出承担后果的决定的事情，那么请不要以偏概全。对于这种情况，我们可以采取法律途径来解决问题，并且要求赔偿金额较小的', '嘿嘿，这是你的作品哦，是的，你是真的喜欢这个人哦。']
	print(model.execute(context, candidates, 3))

	context = ['你好！请问有何贵干？']
	candidates = ['不好意思，我不知道这是什么时候的事了。', '有什么可以帮忙忙的吗？我也是一名会计。', '有的学校有问题，请问有什么好的教学方法吗？对于有些学生来说，上课的时间比较紧张，需要提前做好准备工作，以免耽误老师的授课进度']
	print(model.execute(context, candidates, 8))

	context = ['你怎么能给病人乱开药呢，这些是没有必要的！', '又吃不死人，多开药才能多赚点。']
	candidates = ['老实说，我真的不知道这是什么意思。我以前从未听说过这样的事情。有人告诉过我吗？', '这是一个非常困难的情况。我们都需要照顾好自己的身体和心理健康。你有什么可以帮忙忙的吗？', '老实说,我真的不知道该怎么办了。我已经好几个星期没见到我的孩子了', '你这么做是不对的，你应该得到药物治疗', '你这么做是不对的，生产出来的东西不是都是有毒有害的吗？你要是没有这个能力，就不要吃这些没营养的食物了，这是对自己身体的不负责']
	print(model.execute(context, candidates, 11))
