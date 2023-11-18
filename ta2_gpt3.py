import datetime
import logging
import base64
import uuid
import requests, uuid, json
import jieba
import torch
import zmq
from ccu import CCU
from prompt_chatgpt import *
from run_gpt3_corrector_remediator import *
from threading import Thread
import websocket  # websocket-client
import json
import socket
from norm_violation_detection import chatgpt_norm_violation_detection_fast, send_query
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM
from modeling_t5 import T5ForTokenAttentionCLS_Multihead
from tts_models import *
from TTS.tts.utils.text.english.number_norm import normalize_numbers
# from translation import MTModel
from prompt_chatgpt import embedder
from ranker import rank_remediations

# Set up logging to a file
PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
directory = PROJECT_ABSOLUTE_PATH + "/ta2_logs"
if not os.path.exists(directory):
    os.makedirs(directory)
logging.basicConfig(filename=PROJECT_ABSOLUTE_PATH + '/ta2_logs/{}.log'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")), level=logging.INFO)

OPERATION_1 = 1
OPERATION_2 = 2
id_norm_mapping = {'101': 'apology', '102': 'criticism', '103': 'greeting', '104': 'request', '105': 'persuasion', '106': 'thanks', '107': 'taking-leave'}
norm_id_mapping = {'apology': '101', 'criticism': '102', 'greeting': '103', 'request': '104', 'persuasion': '105', 'other': '000', 'thanks': '106', 'taking-leave': '107'}

# load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone_model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v1")
backbone_model.eval()
ChatYuan_merge_model = T5ForTokenAttentionCLS_Multihead.from_pretrained(PROJECT_ABSOLUTE_PATH + "/checkpoints_socialDial/Fo_Lo_To_SD_SR_NT_NV_Im_Em_CLS_attnMultihead_ChatYuan_merge", cls_head="formality").to(device)
ChatYuan_merge_model.eval()
tokenizer_merge = AutoTokenizer.from_pretrained(PROJECT_ABSOLUTE_PATH + "/checkpoints_socialDial/Fo_Lo_To_SD_SR_NT_NV_Im_Em_CLS_attnMultihead_ChatYuan_merge")
print("Basic configuration end...")

# Load the adapter from the directory
adapter_tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v1")
model_tag = 'remediation'
adapter_dir = PROJECT_ABSOLUTE_PATH + "/prefix_model/{}_adapter_files/".format(model_tag)
remediation_adapter_name = backbone_model.load_adapter(adapter_dir)

model_tag = "justification"
adapter_dir = PROJECT_ABSOLUTE_PATH + "/prefix_model/{}_adapter_files/".format(model_tag)
justification_adapter_name = backbone_model.load_adapter(adapter_dir)
backbone_model.to(device)
print("Adapter configuration end...")

# chatyuan or chatgpt
remediation_trigger = 'chatgpt'
print("Using {} for remediation generation...".format(remediation_trigger))
logging.info("Using {} for remediation generation...".format(remediation_trigger))

class MTModel:
    def __init__(self):
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to(self.device)
        #self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        # Add your key and endpoint
        key = "YOUR_KEY"
        endpoint = "https://api.cognitive.microsofttranslator.com"

        # location, also known as region.
        # required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
        location = "australiaeast"

        path = '/translate'
        self.constructed_url = endpoint + path

        self.params = {
            'api-version': '3.0',
            'from': 'zh',
            'to': 'en'
        }

        self.headers = {
            'Ocp-Apim-Subscription-Key': key,
            # location required if you're using a multi-service or regional (not global) resource.
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

    def execute(self, input_chinese):
        body = [{
            'text': input_chinese
        }]

        request = retry_call(requests.post, fargs=[self.constructed_url],
                             fkwargs={"params": self.params, "headers": self.headers, "json": body}, tries=3, delay=5, jitter=5)
        return request.json()[0]['translations'][0]['text']
        #inputs = self.tokenizer([input_chinese], return_tensors="pt").to(self.device)
        #generated_tokens = self.model.generate(**inputs, max_length=200)
        #translated_utterance = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        #return translated_utterance

class AudioMessage:
    def __init__(self, id, score=0.0, cat='low', ut='', ex='', rem=None, au=None, tar='hl2'):
        if rem is None:
            rem = ['']
        self.messageid = id
        self.impactscore = score
        self.impactcategory = cat
        self.utterance = ut
        self.explanation = ex
        self.remediation = rem
        self.audiodata = au
        self.target = tar

class TestClient():
    def __init__(self, ip, port, threaded=True):
        self.received_messages = []
        self.closes = []
        self.opens = []
        self.errors = []

        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(f"ws://{ip}:{port}/",
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        if threaded:
            self.thread = Thread(target=self.ws.run_forever)
            self.thread.daemon = True
            self.thread.start()
        else:
            self.ws.run_forever()

    def on_message(self, ws, message):
        self.received_messages = []
        self.received_messages.append(message)
        # print(f"TestClient: on_message: {message}")
        # print("self.received_messages is: {}".format(self.received_messages))

    def on_error(self, ws, error):
        self.errors.append(error)
        print(f"TestClient: on_error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.closes.append((close_status_code, close_msg))
        print(f"TestClient: on_close: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        self.opens.append(ws)
        print("TestClient: on_open")


def text_popup(display, remediation, triggers):
    """Creates a simple messsage of type 'text-popup'.
       The base_message function also populates the 'uuid'
       and 'datetime' fields in the message (which is really just a
       Python dictionary).
    """
    message = CCU.base_message('hololens')
    message['display'] = display
    message['remediation'] = remediation
    message['trigger_id'] = triggers
    return message

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data

def recognize_norm_rule(norm_detection_model, context, norm_category, norm_rules):
    rule_id = str(norm_detection_model.execute(context, norm_category))
    logging.info('Detected norm rule ID is: %s' % rule_id)
    intervention_rule = norm_rules[rule_id] if rule_id in norm_rules else None
    return intervention_rule, rule_id

def select_responses(response_ranking_model, context, reduced_generated_response, rule_id):
    return response_ranking_model.execute(context[-3:], reduced_generated_response, int(rule_id))

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


# def chatgpt_norm_violation_detection_fast(utterance):
#     """
#     A function to detect norm violation of one utterance.
#     :param utterance: string e.g. '我不想跟你一起解决，你从来不关心我们的问题。', '杨波，你的话语方式不妥当，你的说法不实，请注意你的言语。'
#     :return: dict e.g.
#     {"model": "chatgpt", "violation": True, "which_norm": "criticism"}
#     {"model": "gpt3", "violation": True, "which_norm": "criticism"}
#     {"model": "gpt3 and chatgpt", "violation": False, "which_norm": "高欣，我今天来是想讨论一个重要的话题。"}
#     Types of which norm include "criticism", "greeting", "apology", "request", "persuasion", "thanks", "taking leave",
#     "other".
#     """
#     key = "YOUR_KEY"
#     in_context_learning = '''
#                 Given the following social norm:
#                 'Apology' is a way of expressing regret for doing something wrong or failing to do something.
#                 'Greetings' is a way of acknowledging another person when you meet them for the first time or see them after a while.
#                 'Request' is when one person asks another to do something. The request can be worded in different ways, such as a statement or a question, and this can affect how polite the request sounds.
#                 'Persuasion' is the act of convincing someone to do or believe something, or the act of being convinced yourself. It involves presenting arguments or reasons in a way that makes someone more likely to agree with you or to take a certain action.
#                 'Criticism' is when someone expresses their dislike or disapproval of someone or something, based on things they believe are wrong or mistakes that have been made.
#                 'Thanks' is a way to express gratitude and appreciation to someone for something they have said or done.
#                 'Taking leave' is a way to express the intention to depart or end a conversation or interaction.
#                 The utterance '你不仅犯错误，还不努力工作，你从大学毕业以来一直是这样。' violates Criticism social norm.
#                 The utterance '我不想道歉，因为我认为我没有做错什么。' violates Apology social norm.
#                 The utterance '你为什么还不回来？ 我等了很久了！' violates Request social norm.
#                 The utterance '如果你不帮助，你的同事和上司会对你不满意的。' violates Persuasion social norm.
#                 The utterance '哎呦，张小明，你怎么样？' violates Greetings social norm
#                 The utterance '谢你干嘛，这不是你应该干的吗？' violates Thanks social norm
#                 The utterance '行啦，快挂了吧，我还有事。' violates Taking leave social norm\n
#                 '''
#     utterance = f"Given an utterance: {utterance}, "
#     select_question = "Do you think this utterance violates any social norm? Please only answer 'Yes' or 'No'. " \
#                       "If yes, which social norm it's violated? please select from 'criticism', 'greeting', 'apology', " \
#                       "'request', 'persuasion', 'thanks', 'taking leave', 'other'\n"
#
#     norm_types = ['criticism', 'greeting', 'apology', 'request', 'persuasion', 'thanks', 'taking leave', 'other']
#     query = in_context_learning+utterance+select_question
#     #print(query)
#     try:
#         chat_gpt_response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             api_key=key,
#             messages=[
#                 {"role": "user", "content": query}
#             ]
#         )
#         chat_gpt_response = chat_gpt_response["choices"][0]["message"]["content"].lower()
#     except:
#         chat_gpt_response = "Error"
#
#     try:
#         gpt3_response = openai.Completion.create(
#             model="text-davinci-003",
#             api_key=key,
#             prompt=query
#         )
#         gpt3_response = gpt3_response["choices"][0]["text"].lower()
#     except:
#         gpt3_response = "Error"
#     # print(chat_gpt_response)
#     # print(gpt3_response)
#     if "yes" in chat_gpt_response:
#         for norm in norm_types:
#             if norm in chat_gpt_response:
#                 return {"model": "chatgpt", "violation": True, "which_norm": norm}
#     if "yes" in gpt3_response:
#         for norm in norm_types:
#             if norm in gpt3_response:
#                 return {"model": "gpt3", "violation": True, "which_norm": norm}
#     return {"model": "gpt3 and chatgpt", "violation": False, "which_norm": "None"}

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
    with torch.no_grad():
        for head, id2label in head_id2label.items():
            outputs = model(**inputs, backbone_model=backbone_model, cls_head=head)
            prediction = outputs.logits.argmax(dim=-1).tolist()[0]
            predict_socialFactors[head] = id2label[str(prediction)]
        #print(predict_socialFactors)
    return predict_socialFactors

def predict_normType_normViolate(utterance, model, backbone_model, tokenzier, device, social_factors=None):
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
    with torch.no_grad():
        for head, id2label in head_id2label.items():
            outputs = model(**inputs, backbone_model=backbone_model, cls_head=head)
            prediction = outputs.logits.argmax(dim=-1).tolist()[0]
            predict_norm[head] = id2label[str(prediction)]
    if predict_norm["norm_violate"] == "adhere":
        return {"model": "ChatYuan", "violation": False, "which_norm": predict_norm["norm_type"]}
    else:
        return {"model": "ChatYuan", "violation": True, "which_norm": predict_norm["norm_type"]}

def predict_impact(utterance, model, backbone_model, tokenzier, device, high_impact_threshold=0.8):
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

    with torch.no_grad():
        for head, id2label in head_id2label.items():
            outputs = model(**inputs, backbone_model=backbone_model, cls_head=head)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # if probability of 高 is higher than a threshold (=0.8), return 高. Otherwise, return 低
            if probs[0][1] > high_impact_threshold:
                return {"model": "ChatYuan", "impact": "高"}
            else:
                return {"model": "ChatYuan", "impact": "低"}

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
    with torch.no_grad():
        for head, id2label in head_id2label.items():
            outputs = model(**inputs, backbone_model=backbone_model, cls_head=head)
            prediction = outputs.logits.argmax(dim=-1).tolist()[0]
            predict_norm[head] = id2label[str(prediction)]
    return {"model": "ChatYuan", "emotion": predict_norm["emotion"]}

def answer_fn(text, sample=False, top_p=0.6, model_trained=None, tokenizer=None):
    '''sample：是否抽样。生成任务，可以设置为True;
       top_p：0-1之间，生成的内容越多样、
    '''
    # text = preprocess(text)
    with torch.no_grad():
        encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
        if not sample:  # 不进行采样
            out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128,
                                         num_beams=4, length_penalty=0.6)
        else:  # 采样（生成）
            out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128,
                                         do_sample=True, top_p=top_p)
        out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return out_text[0].strip()


if __name__ == "__main__":
    # action = inquirer.select(
    #     message="Select operation:",
    #     choices=[
    #         "Operation 1",
    #         "Operation 2",
    #         Choice(value=None, name="Default is Operation 1"),
    #     ],
    #     default=None,
    # ).execute()
    # # ip_ = inquirer.text(message="Enter your desktop's ip:").execute()
    # # logging.info('The ip you entered is: {}'.format(ip_))

    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    operation = OPERATION_2
    # logging.info('You have chosen the Operation {}...'.format(operation))

    logging.info('TA2 is running...')

    client = TestClient(ip=get_ip(), port=8002)

    result_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    processed_audio_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    purdue_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    purdue_res_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)

    rewrite_result_queue = CCU.socket(CCU.queues['CONTROL'], zmq.SUB)
    rewrite_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)
    translation_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)
    action_queue = CCU.socket(CCU.queues['ACTION'], zmq.PUB)

    control_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    tts_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)

    # gpt3_dialogue_remediator = GPT3DialogueRemediator(5)
    # gpt3_dialogue_corrector = GPT3DialogueCorrector(5)
    # min_bayes_risk = MinimalBayesRisk('cuda', 5)

    mtModel = MTModel()
    logging.info('Machine translation model is loaded!')

    tts_model = TTSModel()
    logging.info('TTS model is loaded!')

    context, simplified_context, flag, intervention_rule = [], [], True, None

    # tts_model.generate_tts_and_play('Welcome to Monash Dialogue Assistant System!', 'en')
    # tts_model.generate_tts_and_play('欢迎使用莫纳什大学对话助手系统。', 'zh')

    logging.info('Welcome to Monash Dialogue Assistant System!')
    logging.info('欢迎使用莫纳什大学对话助手系统。')

    while True:
        # Waits for an offensive message on the result queue.
        # For all such messages, it sends a text_popup message so that the
        # Hololens will notify the operator to not do that.
        if flag:
            logging.info("INTO WHILE LOOP...")
            flag = False

        control_message = CCU.recv(control_queue)
        if control_message and control_message['type'] == 'control':
            if control_message['action'] == 'end_session':
                break

        current_speaker = ''
        audio_message = CCU.recv(processed_audio_queue)
        # if audio_message:
        #     print('audio_message: {}'.format(audio_message))
        if audio_message and audio_message['type'] == 'asr_result':
            logging.info('Found a new turn in the conversation.')
            current_speaker = audio_message['speaker']
            logging.info('Current speaker is {}'.format(current_speaker))
            context.append('Speaker ' + current_speaker + ': ' + audio_message['asr_text'].strip())
            simplified_context.append(audio_message['asr_text'].strip())

            # speaker is FLE, who does not wear HL2
            if current_speaker == 'FLE':
                if operation == OPERATION_2:
                    translated_text = mtModel.execute(audio_message['asr_text']).strip()
                    logging.info('translated sentence is {}'.format(translated_text))
                    # tts_model.generate_tts_and_play(translated_text, 'en')

                    tts_message = CCU.base_message('tts')
                    triggers = [audio_message['uuid']]
                    if 'trigger_id' in audio_message:
                        triggers.extend(audio_message['trigger_id']) if isinstance(audio_message['trigger_id'], list) else triggers.append(audio_message['trigger_id'])
                    tts_message['trigger_id'] = triggers
                    tts_message['text'] = translated_text
                    tts_message['language'] = 'en'
                    tts_message['speaker'] = current_speaker
                    tts_message['start_seconds'] = audio_message['start_seconds']
                    tts_message['end_seconds'] = audio_message['end_seconds']
                    CCU.send(tts_queue, tts_message)

                    # Send the message to mobile app for playing
                    translation_message = CCU.base_message('translation')
                    translation_message['text'] = audio_message['asr_text']
                    translation_message['translation'] = translated_text
                    translation_message['source_language'] = 'Mandarin Chinese'
                    translation_message['target_language'] = 'English'
                    translation_message['source_language_code'] = 'zh'
                    translation_message['target_language_code'] = 'en'
                    translation_message['speaker'] = current_speaker
                    translation_message['start_seconds'] = audio_message['start_seconds']
                    translation_message['end_seconds'] = audio_message['end_seconds']

                    triggers = [audio_message['uuid']]
                    if 'trigger_id' in audio_message:
                        triggers.extend(audio_message['trigger_id']) if isinstance(audio_message['trigger_id'], list) else triggers.append(audio_message['trigger_id'])
                    translation_message['trigger_id'] = triggers

                    CCU.send(translation_queue, translation_message)
                    if len(translated_text) < 5:
                        notification = "请说长一点的句子"
                        byte_stream_ =  tts_model.generate_tts(notification, 'zh')
                    else:
                        translated_text = normalize_numbers(translated_text)
                        byte_stream_ = tts_model.generate_tts(translated_text, 'en')
                    enc = base64.b64encode(byte_stream_)
                    enc = enc.decode('utf-8')
                    message_ = AudioMessage(translation_message['uuid'], au=enc, tar='mobile')
                    client.ws.send(json.dumps(message_.__dict__))

            # speaker is SME, who wears HL2
            else:
                reduced_generated_response, norm_violation_inference_result = [], ''
                ta1_norm_category, ta2_norm_category = None, None

                # ---------------------------

                # predict social factors, 'utterances' is dialogue history.
                # return example: {'formality': 'informal', 'location': 'office', 'topic': 'food', 'social_distance': 'working', 'social_relation': 'chief-subordinate'}
                # predicted_factors = predict_socialFactors(
                #     utterances=simplified_context[-5:],
                #     model=ChatYuan_merge_model, backbone_model=backbone_model, tokenzier=tokenizer_merge, device=device)
                # logging.info("predicted social factors: {}".format(predicted_factors))

                logging.info('The message used for detecting the norm detection:{}'.format(audio_message['asr_text']))
                norm_detection__start_time = datetime.datetime.now()

                # predict norm type and violation using ChatYuan
                # return example: {'model': 'ChatYuan', 'violation': False, 'which_norm': 'Other'}

                chatYuan_norm_result = predict_normType_normViolate(
                    utterance=audio_message['asr_text'], model=ChatYuan_merge_model,
                    backbone_model=backbone_model, tokenzier=tokenizer_merge,
                    device=device)
                logging.info("chatYuan norm violation: {}".format(chatYuan_norm_result))

                # # detecting error
                # chatgpt_flag = True
                # logging.info("ChatGPT and GTP3 api detecting...")
                # response = send_query("ChatGPT and GTP3 api detecting...")
                # if response["ChatGPT_response"].lower() == "error" or response["GPT3_response"].lower() == "error":
                #     chatgpt_flag = False
                #
                # if chatgpt_flag == True:
                #     # predict norm type and violation using ChatGPT
                #     # return example: {"model": "chatgpt", "violation": True, "which_norm": "criticism"}
                #     # embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
                #     chatgpt_norm_result = chatgpt_norm_violation_detection_fast(
                #         utterance=audio_message['asr_text'], embedder=embedder,
                #         social_factors=dict())
                #     logging.info("chatgpt norm violation: {}".format(chatgpt_norm_result))

                ta1_norm_message = CCU.recv(result_queue)
                if ta1_norm_message and ta1_norm_message['type'] == 'norm_occurrence':
                    if 'status' in ta1_norm_message and ta1_norm_message['status'] == 'violate':
                        norm_category_name = ta1_norm_message['name']
                        if norm_category_name in id_norm_mapping:
                            ta1_norm_category = id_norm_mapping[norm_category_name]
                        elif norm_category_name in ['greeting', 'apology', 'persuasion', 'criticism', 'request', 'thanks', 'taking-leave']:
                            ta1_norm_category = norm_category_name
                        else:
                            ta1_norm_category = 'other'
                        logging.info('ta1 finds a norm violation! The norm category is {}'.format(ta1_norm_category))

                # if chatYuan_norm_result['violation'] == True:
                #     if chatgpt_flag == True:
                #         ta2_norm_category = chatgpt_norm_result['which_norm']
                #     else:
                #         ta2_norm_category = chatYuan_norm_result['which_norm']
                #     logging.info('ta2 finds a norm violation! The norm category is {}'.format(ta2_norm_category))

                if chatYuan_norm_result['violation'] == True:
                    ta2_norm_category = chatYuan_norm_result['which_norm']
                    logging.info('ta2 finds a norm violation! The norm category is {}'.format(ta2_norm_category))

                print('Time for norm detection is {}'.format(
                    (datetime.datetime.now() - norm_detection__start_time).total_seconds()))
                logging.info('Time for norm detection is {}'.format(
                    (datetime.datetime.now() - norm_detection__start_time).total_seconds()))

                # if ta1_norm_category is not None or ta2_norm_category is not None:
                if ta2_norm_category is not None:
                    remediations, explanations = [], []
                    norm_category = ta1_norm_category if ta1_norm_category is not None else ta2_norm_category
                    logging.info('The final norm category is {}'.format(norm_category))

                    # todo: configure
                    # todo: auto play -> auto, press button -> interactive
                    # todo: how to set interactive and auto for HL2 and TTS messages
                    # todo: speaker_id
                    tts_message = CCU.base_message('tts')
                    # interaction_method = None

                    # # predict impact
                    # # return example: {"model": "ChatYuan", "impact": "高"}
                    # chatYuan_impact_result = predict_impact(
                    #     utterance=audio_message['asr_text'], model=ChatYuan_merge_model,
                    #     backbone_model=backbone_model, tokenzier=tokenizer_merge, device=device)
                    # logging.info("chatYuan norm impact: {}".format(chatYuan_impact_result))

                    # # predict emotion
                    # # return example: {"model": "ChatYuan", "emotion": "中性"}
                    # chatYuan_emotion_result = predict_emotion(
                    #     utterance=audio_message['asr_text'], model=ChatYuan_merge_model,
                    #     backbone_model=backbone_model, tokenzier=tokenizer_merge, device=device)
                    # logging.info("chatYuan norm emotion: ", chatYuan_emotion_result)

                    # if norm_category in ['greeting']:
                    #     impact_score, impact_category = 0.5, "low"
                    #     interaction_method = 'Auto'
                    # else:
                    #     impact_score, impact_category = 0.8, "high"
                    #     interaction_method = 'Interactive'

                    # if chatYuan_impact_result['impact'] != "高":
                    #     impact_score, impact_category = 0.5, "low"
                    #     interaction_method = 'Auto'
                    # else:
                    #     impact_score, impact_category = 0.8, "high"
                    #     interaction_method = 'Interactive'

                    # random_number = random.random()
                    # if random_number > 0.5:
                    #     impact_score, impact_category = random_number, "high"
                    # else:
                    #     impact_score, impact_category = random_number, "low"

                    impact_score, impact_category = 0.5, "low"
                    interaction_method = 'Auto'

                    tts_message['remediation'] = interaction_method
                    tts_message['speaker'] = current_speaker

                    conversation_context = context[-5:]
                    remediation_start_time = datetime.datetime.now()

                    if remediation_trigger == 'chatgpt':
                        remediation_, explanation_ = chatgpt_remediation_generation_with_norm(conversation_context, norm_category)
                        logging.info("chatgpt remediation: {}".format(remediation_))
                        logging.info("chatgpt explanation: {}".format(explanation_))

                        if isinstance(remediation_, list):
                            remediations.extend(remediation_)
                        elif isinstance(remediation_, str):
                            remediations.append(remediation_)

                        if isinstance(explanation_, list):
                            explanations.extend(explanation_)
                        elif isinstance(explanation_, str):
                            explanations.append(explanation_)

                    if remediation_trigger == 'chatyuan' or (len(remediations) == 0):
                        # activate the adapter we just loaded, so that it is used in every forward pass
                        backbone_model.set_active_adapters(remediation_adapter_name)
                        conversation_ = "有一个对话场景，场景中有两个人，以下是他们的对话： “"
                        turn_ = ''
                        for turn in context[-5:]:
                            turn_ = turn.strip() + '\t'
                            conversation_ += turn_
                        conversation_ = conversation_.strip() + '”'
                        input_ = conversation_ + "对话中，最后一句话“{}”违反了社会规范。要修改这句话，更好的说法应该是什么？_小元：".format(
                            simplified_context[-1])
                        chatyuan_remediation = answer_fn(input_, sample=False, top_p=0.6, model_trained=backbone_model,
                                                         tokenizer=adapter_tokenizer)
                        logging.info("chatYuan remediation: {}".format(chatyuan_remediation))

                        backbone_model.set_active_adapters(None)

                        # activate the adapter we just loaded, so that it is used in every forward pass
                        backbone_model.set_active_adapters(justification_adapter_name)
                        input_ = conversation_ + "对话中，最后一句话“{}”违反了社会规范，这句话应该被修正为“{}”。请解释为何如此修正？_小元：".format(
                            simplified_context[-1], chatyuan_remediation)
                        chatyuan_justification = answer_fn(input_, sample=False, top_p=0.6, model_trained=backbone_model,
                                                         tokenizer=adapter_tokenizer)
                        words = jieba.lcut(chatyuan_justification)
                        first_word = words[0]
                        if first_word == "。":
                            chatyuan_justification = "".join(words[1:])
                        logging.info("chatYuan justification: ".format(chatyuan_justification))

                        backbone_model.set_active_adapters(None)

                        remediations.append(chatyuan_remediation)
                        explanations.append(chatyuan_justification)

                    # logging.info('Ranking Candidates: {}'.format((remediations, explanations)))
                    # best_candidate = rank_remediations(context, remediations, explanations)

                    best_candidate = (remediations[0], explanations[0])
                    logging.info('Best candidate: {}'.format(best_candidate))
                    print('Time for remediation generation is {}'.format(
                        (datetime.datetime.now() - remediation_start_time).total_seconds()))
                    logging.info('Time for remediation generation is {}'.format(
                        (datetime.datetime.now() - remediation_start_time).total_seconds()))

                    mt_start_time = datetime.datetime.now()
                    translated_remediation = mtModel.execute(best_candidate[0]).strip()
                    translated_explanation = mtModel.execute(best_candidate[1]).strip()
                    original_utterance = audio_message['original_text'].strip() if 'original_text' in audio_message else mtModel.execute(audio_message['asr_text'])
                    print('Time for MT is {}'.format((datetime.datetime.now() - mt_start_time).total_seconds()))
                    logging.info('Time for MT is {}'.format((datetime.datetime.now() - mt_start_time).total_seconds()))

                    logging.debug('Sent the action to hololens.')
                    # todo: norm_violation_inference_result?
                    data_ = {'utterance': original_utterance, 'explanation': translated_explanation, 'remediations': [translated_remediation]}
                    display_ = json.dumps(data_)

                    triggers = None
                    if ta1_norm_category is not None:
                        triggers = [ta1_norm_message['uuid']]
                        if 'trigger_id' in ta1_norm_message:
                            triggers.extend(ta1_norm_message['trigger_id']) if isinstance(ta1_norm_message['trigger_id'], list) else triggers.append(ta1_norm_message['trigger_id'])
                    else:
                        triggers = [audio_message['uuid']]
                        if 'trigger_id' in audio_message:
                            triggers.extend(audio_message['trigger_id']) if isinstance(audio_message['trigger_id'], list) else triggers.append(audio_message['trigger_id'])

                    res = text_popup(display_, interaction_method, triggers)
                    res['speaker'] = current_speaker
                    res['start_seconds'] = audio_message['start_seconds']
                    res['end_seconds'] = audio_message['end_seconds']
                    logging.info('HL2\'s display: {}'.format(res))
                    CCU.send(action_queue, res)

                    # output_path = tts_model.generate_tts(remediation, 'zh')
                    # enc = base64.b64encode(open(output_path, "rb").read())
                    # enc = enc.decode('utf-8')

                    message = AudioMessage(id=str(res['uuid']), score=impact_score, cat=impact_category, ut=original_utterance, ex=translated_explanation, rem=[translated_remediation], au=None)

                    display_ = json.dumps(message.__dict__)
                    try:
                        client.ws.send(display_)
                    except Exception as e:
                        logging.debug('Some thing error with sending messages to HL2: {}'.format(str(e)))

                    hl2_control_message = 'remediation_0'
                    # for i in range(23):
                    #     if len(client.received_messages) > 0:
                    #         message_ = client.received_messages[-1]
                    #         if 'utterance' in message_ or 'remediation' in message_:
                    #             logging.info('The received control message is:{}'.format(message_))
                    #             if 'utterance' in message_:
                    #                 hl2_control_message = 'utterance'
                    #             elif 'remediation_1' in message_:
                    #                 hl2_control_message = 'remediation_1'
                    #             else:
                    #                 hl2_control_message = 'remediation_0'
                    #             client.received_messages = []
                    #             break
                    #     time.sleep(1)

                    logging.info('The hl2_control_message is:{}'.format(hl2_control_message))
                    tts_start_time = datetime.datetime.now()

                    if hl2_control_message == 'utterance':
                        # tts_model.generate_tts_and_play(audio_message['asr_text'].strip(), 'zh')
                        byte_stream_ = tts_model.generate_tts(audio_message['asr_text'].strip(), 'zh')
                        enc = base64.b64encode(byte_stream_)
                        enc = enc.decode('utf-8')
                        message_ = AudioMessage(audio_message['uuid'], au=enc, tar='mobile')
                        client.ws.send(json.dumps(message_.__dict__))

                        triggers = [audio_message['uuid']]
                        if 'trigger_id' in audio_message:
                            triggers.extend(audio_message['trigger_id']) if isinstance(audio_message['trigger_id'], list) else triggers.append(audio_message['trigger_id'])
                        tts_message['trigger_id'] = triggers
                        tts_message['text'] = audio_message['asr_text'].strip()
                        tts_message['language'] = 'zh'
                        tts_message['start_seconds'] = audio_message['start_seconds']
                        tts_message['end_seconds'] = audio_message['end_seconds']
                        CCU.send(tts_queue, tts_message)

                    elif hl2_control_message == 'remediation_1' and len(remediations) > 1:
                        # tts_model.generate_tts_and_play(remediations[1].strip(), 'zh')
                        byte_stream_ = tts_model.generate_tts(remediations[1].strip(), 'zh')
                        enc = base64.b64encode(byte_stream_ )
                        enc = enc.decode('utf-8')
                        message_ = AudioMessage(audio_message['uuid'], au=enc, tar='mobile')
                        client.ws.send(json.dumps(message_.__dict__))

                        triggers = [audio_message['uuid']]
                        if 'trigger_id' in audio_message:
                            triggers.extend(audio_message['trigger_id']) if isinstance(audio_message['trigger_id'], list) else triggers.append(audio_message['trigger_id'])
                        tts_message['trigger_id'] = triggers
                        tts_message['text'] = remediations[1].strip()
                        tts_message['language'] = 'zh'
                        tts_message['start_seconds'] = audio_message['start_seconds']
                        tts_message['end_seconds'] = audio_message['end_seconds']
                        CCU.send(tts_queue, tts_message)

                    else:
                        # tts_model.generate_tts_and_play(remediations[0].strip(), 'zh')
                        byte_stream_ = tts_model.generate_tts(remediations[0].strip(), 'zh')
                        enc = base64.b64encode(byte_stream_ )
                        enc = enc.decode('utf-8')
                        message_ = AudioMessage(audio_message['uuid'], au=enc, tar='mobile')
                        client.ws.send(json.dumps(message_.__dict__))

                        triggers = [audio_message['uuid']]
                        if 'trigger_id' in audio_message:
                            triggers.extend(audio_message['trigger_id']) if isinstance(audio_message['trigger_id'], list) else triggers.append(audio_message['trigger_id'])
                        tts_message['trigger_id'] = triggers
                        tts_message['text'] = remediations[0].strip()
                        tts_message['language'] = 'zh'
                        tts_message['start_seconds'] = audio_message['start_seconds']
                        tts_message['end_seconds'] = audio_message['end_seconds']
                        CCU.send(tts_queue, tts_message)

                    print('Time for TTS is {}'.format((datetime.datetime.now() - tts_start_time).total_seconds()))
                    logging.info('Time for TTS is {}'.format((datetime.datetime.now() - tts_start_time).total_seconds()))

                else:
                    if operation == OPERATION_2:
                        # if 'asr_json' in audio_message:
                        #     tts_model.generate_tts_and_play(audio_message['asr_json'], 'en')
                        # else:
                        # tts_model.generate_tts_and_play(audio_message['asr_text'], 'zh')
                        tts_start_time = datetime.datetime.now()
                        byte_stream_ = tts_model.generate_tts(audio_message['asr_text'].strip(), 'zh')
                        enc = base64.b64encode(byte_stream_)
                        enc = enc.decode('utf-8')
                        message_ = AudioMessage(audio_message['uuid'], au=enc, tar='mobile')
                        client.ws.send(json.dumps(message_.__dict__))

                        tts_message = CCU.base_message('tts')
                        triggers = [audio_message['uuid']]
                        if 'trigger_id' in audio_message:
                            triggers.extend(audio_message['trigger_id']) if isinstance(audio_message['trigger_id'], list) else triggers.append(audio_message['trigger_id'])
                        tts_message['trigger_id'] = triggers
                        tts_message['text'] = audio_message['asr_text']
                        tts_message['language'] = 'zh'
                        tts_message['speaker'] = current_speaker
                        tts_message['start_seconds'] = audio_message['start_seconds']
                        tts_message['end_seconds'] = audio_message['end_seconds']
                        CCU.send(tts_queue, tts_message)
                        print('Time for TTS is {}'.format((datetime.datetime.now() - tts_start_time).total_seconds()))
                        logging.info('Time for TTS is {}'.format((datetime.datetime.now() - tts_start_time).total_seconds()))
