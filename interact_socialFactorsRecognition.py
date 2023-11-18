import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from modeling_t5 import T5ForTokenAttentionCLS_Multihead
from norm_violation_detection import chatgpt_norm_violation_detection_fast
from sentence_transformers import SentenceTransformer


def predict_socialFactors(utterances, model, backbone_model, tokenzier, device):
    """
    A function to predict social factors of a given dialogue.
    :param utterances: list of string e.g., ["王哥，这是我预定的会议室。", "这会议室还要预定才能用吗？", "是的，需要提前一天预定哦。"]
    :param model: prediction model
    :param tokenzier:
    :param device: cuda or cpu
    :return: dict e.g.,
    {'formality': 'informal', 'location': 'office', 'topic': 'food', 'social_distance': 'working', 'social_relation': 'chief-subordinate'}
    """
    text = " ".join(utterances)
    inputs = tokenzier(text, return_tensors="pt").to(device)
    formality_id2label = {"0": "formal", "1": "informal"}
    location_id2label= {"0": "home", "1": "hotel", "2": "office", "3": "online", "4": "open-area", "5": "police-station", "6": "refugee-camp",
    "7": "restaurant", "8": "school", "9": "store"}
    topic_id2label = {"0": "child-missing", "1": "counter-terrorism", "2": "farming", "3": "food", "4": "life-trivial",
    "5": "office-affairs", "6": "police-corruption", "7": "poverty-assistance", "8": "refugee", "9": "sale", "10": "school-life",
    "11": "tourism" }
    social_distance_id2label = {"0": "family", "1": "friend", "2": "neighborhood", "3": "romantic", "4": "stranger", "5": "working"}
    social_relation_id2label = {"0": "chief-subordinate", "1": "commander-soldier", "2": "customer-server", "3": "elder-junior",
    "4": "mentor-mentee", "5": "partner-partner", "6": "peer-peer", "7": "student-professor" }

    head_id2label = {"formality": formality_id2label, "location": location_id2label, "topic": topic_id2label,
                     "social_distance": social_distance_id2label, "social_relation": social_relation_id2label}

    predict_socialFactors = {}
    for head, id2label in head_id2label.items():
        outputs = model(**inputs, backbone_model=backbone_model, cls_head=head)
        prediction = outputs.logits.argmax(dim=-1).tolist()[0]
        predict_socialFactors[head] = id2label[str(prediction)]
    #print(predict_socialFactors)
    return predict_socialFactors

def predict_normType_normViolate(utterance, social_factors, model, backbone_model, tokenzier, device):
    """
    A function to predict social factors of a given dialogue.
    :param utterance: string e.g., "王哥，这是我预定的会议室。"
    :param model: prediction model
    :param tokenzier:
    :param device: cuda or cpu
    :return: dict e.g.,
    {'model': 'ChatYuan', 'violation': False, 'which_norm': 'Other'}
    """
    text = utterance
    inputs = tokenzier(text, return_tensors="pt").to(device)
    norm_type_id2label = {"0": "apology", "1": "criticism", "2": "greeting", "3": "persuasion", "4": "request"}
    norm_violate_id2label = {"0": "adhere", "1": "violate"}

    head_id2label = {"norm_type": norm_type_id2label, "norm_violate": norm_violate_id2label}

    predict_norm = {}
    for head, id2label in head_id2label.items():
        outputs = model(**inputs, backbone_model=backbone_model, cls_head=head)
        prediction = outputs.logits.argmax(dim=-1).tolist()[0]
        predict_norm[head] = id2label[str(prediction)]
    if predict_norm["norm_violate"] == "adhere":
        return {"model": "ChatYuan", "violation": False, "which_norm": predict_norm["norm_type"]}
    else:
        return {"model": "ChatYuan", "violation": True, "which_norm": predict_norm["norm_type"]}

def predict_impact(utterance, model, backbone_model, tokenzier, device):
    """
    A function to predict social factors of a given dialogue.
    :param utterance: string e.g., "王哥，这是我预定的会议室。"
    :param model: prediction model
    :param tokenzier:
    :param device: cuda or cpu
    :return: dict e.g., {"model": "ChatYuan", "impact": "高"}
    """
    text = utterance
    inputs = tokenzier(text, return_tensors="pt").to(device)
    norm_impact_id2label = {"0": "低", "1": "高"}
    head_id2label = {"impact": norm_impact_id2label}

    predict_norm = {}
    for head, id2label in head_id2label.items():
        outputs = model(**inputs, backbone_model=backbone_model, cls_head=head)
        prediction = outputs.logits.argmax(dim=-1).tolist()[0]
        predict_norm[head] = id2label[str(prediction)]
    return {"model": "ChatYuan", "impact": predict_norm["impact"]}

def predict_emotion(utterance, model, backbone_model, tokenzier, device):
    """
    A function to predict social factors of a given dialogue.
    :param utterance: string e.g., "王哥，这是我预定的会议室。"
    :param model: prediction model
    :param tokenzier:
    :param device: cuda or cpu
    :return: dict e.g., {"model": "ChatYuan", "emotion": "中性"}
    """
    text = utterance
    inputs = tokenzier(text, return_tensors="pt").to(device)
    norm_emotion_id2label = {'0': '中性', '1': '伤心', '2': '反感', '3': '害怕', '4': '开心', '5': '惊讶', '6': '感谢', '7': '愤怒', '8': '担忧', '9': '放松', '10': '正面', '11': '消极', '12': '负面'}
    head_id2label = {"emotion": norm_emotion_id2label}

    predict_norm = {}
    for head, id2label in head_id2label.items():
        outputs = model(**inputs, backbone_model=backbone_model, cls_head=head)
        prediction = outputs.logits.argmax(dim=-1).tolist()[0]
        predict_norm[head] = id2label[str(prediction)]
    return {"model": "ChatYuan", "emotion": predict_norm["emotion"]}


# load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone_model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v1").to(device)
model = T5ForTokenAttentionCLS_Multihead.from_pretrained("checkpoints_socialDial/Fo_Lo_To_SD_SR_NT_NV_Im_Em_CLS_attnMultihead_ChatYuan_merge", cls_head="formality").to(device)
tokenizer = AutoTokenizer.from_pretrained("checkpoints_socialDial/Fo_Lo_To_SD_SR_NT_NV_Im_Em_CLS_attnMultihead_ChatYuan_merge")

# predict social factors, 'utterances' is dialogue history.
# return example: {'formality': 'informal', 'location': 'office', 'topic': 'food', 'social_distance': 'working', 'social_relation': 'chief-subordinate'}
predicted_factors = predict_socialFactors(utterances=["王哥，这是我预定的会议室。", "这会议室还要预定才能用吗？", "是的，需要提前一天预定哦。"],
                      model=model, backbone_model=backbone_model, tokenzier=tokenizer, device=device)
print("predicted social factors: ", predicted_factors)

# predict norm type and violation using ChatGPT
# return example: {"model": "chatgpt", "violation": True, "which_norm": "criticism"}
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
chatgpt_result = chatgpt_norm_violation_detection_fast(utterance="我想知道你们国家如何解决卫星通信系统里的一些问题。难道不行吗?", embedder=embedder, social_factors=predicted_factors)
print("chatgpt norm violation: ", chatgpt_result)

# predict norm type and violation using ChatYuan
# return example: {'model': 'ChatYuan', 'violation': False, 'which_norm': 'Other'}
chatYuan_result = predict_normType_normViolate(utterance="我想知道你们国家如何解决卫星通信系统里的一些问题。难道不行吗?", social_factors=predicted_factors, model=model, backbone_model=backbone_model, tokenzier=tokenizer, device=device)
print("chatYuan norm violation: ", chatYuan_result)

# predict impact
# return example: {"model": "ChatYuan", "impact": "高"}
chatYuan_impact_result = predict_impact(utterance="我想知道你们国家如何解决卫星通信系统里的一些问题。难道不行吗?", model=model, backbone_model=backbone_model, tokenzier=tokenizer, device=device)
print("chatYuan norm impact: ", chatYuan_impact_result)

# predict emotion
# return example: {"model": "ChatYuan", "emotion": "中性"}
chatYuan_emotion_result = predict_emotion(utterance="我想知道你们国家如何解决卫星通信系统里的一些问题。难道不行吗?", model=model, backbone_model=backbone_model, tokenzier=tokenizer, device=device)
print("chatYuan norm emotion: ", chatYuan_emotion_result)

# df = pd.read_csv("indonisia_dataset/hold_out.csv")
# predicted_norm_violate = []
# predicted_norm_type = []
# for index, row in tqdm(df.iterrows(), total=len(df)):
#     text = row['utterance']
#     predicted_factors = predict_socialFactors(utterances=[text],
#                                               model=model, backbone_model=backbone_model, tokenzier=tokenizer,
#                                               device=device)
#     predict = chatgpt_norm_violation_detection_fast(utterance=text, embedder=embedder, social_factors=predicted_factors)
#     print(predict)
#     if predict["violation"] == True:
#         predicted_norm_violate.append("violate")
#     else:
#         predicted_norm_violate.append("adhere")
#     predicted_norm_type.append(predict["which_norm"])
#
# df["chatgpt_predicted_status"] = predicted_norm_violate
# df["chatgpt_predicted_type"] = predicted_norm_type
# df.to_csv("indonisia_dataset/hold_out.csv", index=False)