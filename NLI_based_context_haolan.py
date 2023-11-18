#!/usr/bin/env python
# -*- coding: utf-8 -*-


#from transformers import AutoModelForSequenceClassification, AutoTokenizer
#import torch
import numpy as np
import json
import codecs

rule_details = {
	0: {
		'norm_type': 'apology',
		'context': 'a minor offence has been committed',
		'action': 'the person should give a light apology'
	},
	1: {
		'norm_type': 'apology',
		'context': 'a major offence has been committed',
		'action': 'the person should use stronger apology words'
	},
	2: {
		'norm_type': 'apology',
		'context': 'a person commonly associated with a group deeply offends another person who has a relationship with that group, but the offender is not necessarily sorry for it',
		'action': 'a person associated with the offender’s group, especially a person who bears responsibility, extend an apology to the offended party on behalf of the offender'
	},
	3: {
		'norm_type': 'greetings',
		'context': 'a formal/business setting, or in a situation with a low-status person addressing a high status person, or in a situation with unfamiliar people meeting,',
		'action': 'it is obligatory to show acknowledgement of the other party and use a standard greeting along with physical acknowledgements such as smiles, handshakes, or bows.'
	},
	4: {
		'norm_type': 'greetings',
		'context': 'a public setting, one person is serving the other in a professional capacity,',
		'action': 'it is obligatory to acknowledge the customer with a standard greeting (words or smile/bow), but it is not necessary to use words or have extensive interaction beyond what is required for the transaction; it is acceptable to get right to the point of the interaction.'
	},
	5: {
		'norm_type': 'greetings',
		'context': 'an informal setting between people already familiar with each other and who see each other often',
		'action': 'acceptable to omit a formal greeting expression and instead comment on their or one’s own arrival/presence or other current action.'
	},
	6: {
		'norm_type': 'request',
		'context': 'a higher status/power party and a lower status/party who are already familiar with each other,',
		'action': 'use a direct command to make requests.'
	},
	7: {
		'norm_type': 'request',
		'context': 'an equal, lower status, or unfamiliar person gives a direct command,',
		'action': 'people are preferable to use a politeness marker to give polite commands'
	},
	8: {
		'norm_type': 'request',
		'context': 'any context that requires extra politeness, especially interactions between unfamiliar people / lower to higher status, or if the request is more significant or costly to fulfill,',
		'action': 'people should ask about preparatory conditions for a request'
	},
	9: {
		'norm_type': 'request',
		'context': 'any context that requires politeness, especially interactions between unfamiliar people / lower to higher status persons, or if the request is more significant or costly to fulfill',
		'action': 'a speaker implies a request through strong hints rather than directly requesting it.'
	},
	10: {
		'norm_type': 'request',
		'context': 'a situation that requires extra politeness, due to status, power, or possibly gender',
		'action': 'a speaker implies a request through strong hints rather than directly requesting it.'
	},
	11: {
	    'norm_type': 'criticism',
        'context': 'In a professional setting with lower status speaking to higher status OR peer speaking to peer, it is preferred to use indirect, subtle language',
        'action': 'people can use use indirect, subtle language when criticizing while being respectful',
	},
	12: {
		'norm_type': 'criticism',
        'context': 'In a professional setting with higher status speaking to lower status, it is permitted to use direct language',
        'action': 'people can use direct criticism, strong tone and display emotions when criticizing',
	},
	13: {
	    'norm_type': 'persuasion',
        'context': 'the process of trying to convince someone to agree to do something',
        'action': 'via logic, comparisons of advantages and disadvantages, Historical precedence, a chain of syllogisms',
	},
	14: {
		'norm_type': 'persuasion',
        'context': 'When a lower status person is persuading a higher status person',
        'action': 'it is preferable to use indirect persuasion, Use indirect methods to persuade others',
	}
}

rule_normalization = {
	v['context']: key for (key, v) in rule_details.items()
}

rule_normalization["In a public setting, if one person is serving the other in a professional capacity,"] = 4
rule_normalization['In a professional setting with higher status speaking to lower status, it is permitted to use direct language,'] = 12
rule_normalization['When a lower status person is persuading a higher status person.'] = 14


rule_set = {}
# model = AutoModelForSequenceClassification.from_pretrained("symanto/xlm-roberta-base-snli-mnli-anli-xnli")
# tokenizer = AutoTokenizer.from_pretrained("symanto/xlm-roberta-base-snli-mnli-anli-xnli")

with open('./final_version_1027.json') as out:
	data = json.load(out)

	for dialogue in data['annotations']:
		norm_rule = dialogue['norm rule']['context']
		index = rule_normalization[norm_rule]
		dialogue['norm rule'] = rule_details[index]
		dialogue['norm rule']['index'] = index
		# dialogue_text = ""
		# for index, text in enumerate(dialogue['dialogue']):
		# 	translation = text['translation']
		# 	if index % 2 == 0:
		# 		 dialogue_text += "A: %s " % translation
		# dialogue_text = dialogue_text.strip()
		# input_pairs = [(dialogue_text, norm_rule)]
		# inputs = tokenizer(input_pairs, truncation="only_first", return_tensors="pt", padding=True)
		# logits = model(**inputs).logits
		# _, index = torch.max(logits, dim=1)
		# print(index[0].item())

with codecs.open('./final_version_1027_updated.json', 'w', encoding='utf-8') as out:
	json.dump(data, out, indent=4, ensure_ascii=False)

