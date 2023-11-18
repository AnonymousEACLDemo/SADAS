import logging
from functools import reduce

import torch
import zmq
from ccu import CCU
from generate_model import PhraseModel
from run_norm_rule_classification_model import norm_rule_model
from run_ta1_norm_classification_model import norm_classification_model
from response_ranking_model import res_ranking_model
from constant import *
from threading import Thread
import websocket  # websocket-client
import json
import socket
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import subprocess
from pydub import AudioSegment
from pydub.playback import play

OPERATION_1 = 1
OPERATION_2 = 2
id_norm_mapping = {'101': 'apology', '102': 'criticism', '103': 'greeting', '104': 'request', '105': 'persuasion'}

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
        self.received_messages.append(message)
        print(f"TestClient: on_message: {message}")

    def on_error(self, ws, error):
        self.errors.append(error)
        print(f"TestClient: on_error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.closes.append((close_status_code, close_msg))
        print(f"TestClient: on_close: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        self.opens.append(ws)
        print("TestClient: on_open")

class MTModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

    def execute(self, input_chinese):
        inputs = self.tokenizer([input_chinese], return_tensors="pt").to(self.device)
        generated_tokens = self.model.generate(**inputs, max_length=200)
        translated_utterance = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return translated_utterance

class TTSModel:
    def __init__(self):
        self.model_name_dict = {
            "en": "tts_models/en/ljspeech/tacotron2-DCA",
            "zh": "tts_models/zh-CN/baker/tacotron2-DDC-GST"
        }
        self.vocoder_name_dict = {
            "en": "vocoder_models/en/ek1/wavegrad",
            "zh": "vocoder_models/universal/libri-tts/fullband-melgan"
        }

    def generate_tts_and_play(self, text, language, output_path="out_tts.wav"):
        if language == 'zh':
            text = ''.join([ch if not ch.isascii() else ' ' for ch in text])
            text = text.strip()
            if not text.endswith('。'):
                text += "。"

        subprocess.run(["tts", "--text", text, "--model_name", self.model_name_dict[language], "--vocoder_name",
                        self.vocoder_name_dict[language], "--out_path", output_path])

        song = AudioSegment.from_wav(output_path)
        print(song.duration_seconds)
        play(song)

def text_popup(display_, contents, uuid):
    """Creates a simple messsage of type 'text-popup'.
       The base_message function also populates the 'uuid'
       and 'datetime' fields in the message (which is really just a
       Python dictionary).
    """
    message = CCU.base_message('hololens')
    message['display'] = display_
    message['say'] = contents
    message['tts'] = contents
    message['trigger_id'] = [uuid]
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

if __name__ == "__main__2":
    action = inquirer.select(
        message="Select operation:",
        choices=[
            "Operation 1",
            "Operation 2",
            Choice(value=None, name="Default is Operation 1"),
        ],
        default=None,
    ).execute()
    # ip_ = inquirer.text(message="Enter your desktop's ip:").execute()
    # logging.info('The ip you entered is: {}'.format(ip_))

    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    operation = OPERATION_2 if action == 'Operation 2' else OPERATION_1
    logging.info('You have chosen the Operation {}...'.format(operation))

    logging.info('TA2 is running...')

    # client = TestClient(ip=ip_, port=8002)
    client = TestClient(ip=get_ip(), port=8002)

    result_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    processed_audio_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    purdue_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    purdue_res_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)

    rewrite_result_queue = CCU.socket(CCU.queues['CONTROL'], zmq.SUB)
    rewrite_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)
    action_queue = CCU.socket(CCU.queues['ACTION'], zmq.PUB)

    control_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)

    norm_rule_path = PROJECT_ABSOLUTE_PATH + "/models/dict/norm.json"
    norm_rules = read_json(norm_rule_path)
    logging.info('Norm rules are loaded!')
    # print(norm_rules)

    generate_model = PhraseModel()
    logging.info('Generation model is loaded!')

    bert_norm_rule_classification_path = PROJECT_ABSOLUTE_PATH + "/models/bert_norm_rule_classification"
    norm_detection_model = norm_rule_model('bert-base-chinese', bert_norm_rule_classification_path)
    logging.info('Norm detection model is loaded!')

    assistant_norm_classification_path = PROJECT_ABSOLUTE_PATH + "/models/bert_ta1_norm_classification"
    assistant_norm_classification_model = norm_classification_model('bert-base-chinese', assistant_norm_classification_path)
    logging.info('Assistant norm classification model is loaded!')

    response_ranking_model_path = PROJECT_ABSOLUTE_PATH + "/models/bert_response_ranking"
    response_ranking_model = res_ranking_model('bert-base-chinese', response_ranking_model_path)
    logging.info('Response ranking model is loaded!')

    mtModel = MTModel()
    logging.info('Machine translation model is loaded!')

    tts_model = TTSModel()
    logging.info('TTS model is loaded!')

    context, flag, intervention_rule = [], True, None

    tts_model.generate_tts_and_play('Welcome to Monash Dialogue Assistant System!', 'en')
    tts_model.generate_tts_and_play('欢迎使用莫纳什大学对话助手系统。', 'zh')

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
            context.append(audio_message['asr_text'])
            current_speaker = audio_message['speaker']
            logging.info('Current speaker is {}'.format(current_speaker))

            # speaker is FLE, who does not wear HL2
            if current_speaker == 'FLE':
                if operation == OPERATION_2:
                    translated_text = mtModel.execute(audio_message['asr_text'])
                    logging.info('translated sentence is {}'.format(translated_text))
                    tts_model.generate_tts_and_play(translated_text, 'en')

            # speaker is SME, who wears HL2
            else:
                reduced_generated_response, norm_violation_inference_result = [], ''
                norm_message = CCU.recv(result_queue)
                if norm_message and norm_message['type'] == 'norm_occurrence':
                    norm_context = context[-3:]
                    norm_category = id_norm_mapping[norm_message['name']] if norm_message['name'] in id_norm_mapping else 'No norm'
                    logging.info('norm_category is {}'.format(norm_category))
                    norm_status = 'Yes' if norm_message['status'] =='violate' else 'No'
                    norm_probability = norm_message['llr']
                    norm_violation_inference_result = assistant_norm_classification_model.execute(norm_context, norm_category, norm_status, norm_probability)

                    if norm_violation_inference_result != 'OK':
                        logging.info('Found a norm violation whose category is {}.'.format(norm_violation_inference_result))
                        rewrite_message = CCU.base_message('offensive_speech')
                        rewrite_message['input_text'] = context[-1]
                        CCU.send(rewrite_queue, rewrite_message)
                        logging.info('Found an offensive speech: {}'.format(rewrite_message))

                        intervention_rule, rule_id = recognize_norm_rule(norm_detection_model, context, norm_violation_inference_result,
                                                                norm_rules)
                        generated_response = generate_model.seperate_gpt2_causal_text_generation(context[-10:])
                        # logging.info('Context: {}'.format(context))
                        logging.info('The generation model composes the response: {}'.format(generated_response))
                        repls = ('[SEP]', ''), ('[CLS]', '')
                        reduced_generated_response = [reduce(lambda a, kv: a.replace(*kv), repls, x) for x in
                                                      generated_response]

                        rewrite_result = CCU.recv_block(rewrite_result_queue)
                        if rewrite_result['type'] == 'translation':
                            rewrite_responses = rewrite_result['translation']
                            logging.info('The rewrite model generates the responses: {}'.format(rewrite_responses))
                            reduced_generated_response.extend(rewrite_responses)

                        logging.info('All the generated responses: {}'.format(reduced_generated_response))
                        selected_response = select_responses(response_ranking_model, context, reduced_generated_response, rule_id)
                        logging.info('The selected response: {}'.format(selected_response))

                        tts_model.generate_tts_and_play(selected_response, 'zh')

                        # translate into English
                        translated_selected_response = mtModel.execute(selected_response)
                        if intervention_rule is not None:
                            explanation = "Since {0}, {1}.".format(intervention_rule['context'], intervention_rule['action'])
                        else:
                            explanation = "Violating norm."
                        data_ = {'title': norm_violation_inference_result, 'message': explanation, 'remediation': translated_selected_response}
                        # data_ = {'title': norm_violation_inference_result, 'message': 'message', 'remediation': 'remediation'}

                        display_ = json.dumps(data_)
                        client.ws.send(display_)
                        logging.debug('Sent the action to hololens.')

                        res = text_popup(display_, selected_response, norm_message['uuid'])
                        logging.info(res)
                        CCU.send(action_queue, res)

                    else:
                        if operation == OPERATION_2:
                            # if 'asr_json' in audio_message:
                            #     tts_model.generate_tts_and_play(audio_message['asr_json'], 'en')
                            tts_model.generate_tts_and_play(audio_message['asr_text'], 'zh')


                else:
                    if operation == OPERATION_2:
                        # if 'asr_json' in audio_message:
                        #     tts_model.generate_tts_and_play(audio_message['asr_json'], 'en')
                        # else:
                        tts_model.generate_tts_and_play(audio_message['asr_text'], 'zh')

        purdue_message = CCU.recv(purdue_queue)
        if purdue_message:
            if purdue_message['type'] == 'purdue_emotion':
                rewrite_message = CCU.base_message('emotion')
                rewrite_message['name'] = purdue_message['result']
                rewrite_message['trigger_id'] = [purdue_message['uuid']]
                rewrite_message['llr'] = purdue_message['llr']
                logging.info(rewrite_message)
                CCU.send(purdue_res_queue, rewrite_message)

            elif purdue_message['type'] == 'purdue_valence':
                rewrite_message = CCU.base_message('valence')
                rewrite_message['level'] = int(purdue_message['result'])
                rewrite_message['trigger_id'] = [purdue_message['uuid']]
                logging.info(rewrite_message)
                CCU.send(purdue_res_queue, rewrite_message)

            elif purdue_message['type'] == 'purdue_arousal':
                rewrite_message = CCU.base_message('arousal')
                rewrite_message['level'] = int(purdue_message['result'])
                rewrite_message['trigger_id'] = [purdue_message['uuid']]
                logging.info(rewrite_message)
                CCU.send(purdue_res_queue, rewrite_message)







