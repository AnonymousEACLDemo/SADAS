# ASR process.
#
import base64
import io
import logging
import os
import time
import signal
import uuid

import requests
import torch
import zmq
import whisper
import openai

from ccu import CCU
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from translation import MTModel

import constant
from pydub import AudioSegment
from langdetect import detect
from collections import OrderedDict

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)
from retry.api import retry_call

model = whisper.load_model("medium")
logging.info('ASR example is running...')

audio_queue = CCU.socket(CCU.queues['PROCESSED_AUDIO'], zmq.SUB)
output_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)
park_output_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)

recorded_audio_name_index = 0
audio_path = constant.PROJECT_ABSOLUTE_PATH + '/{}.wav'.format(recorded_audio_name_index)


"""
class MTModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

    def execute(self, input_chinese):
        inputs = self.tokenizer([input_chinese], return_tensors="pt").to(self.device)
        generated_tokens = self.model.generate(**inputs, max_length=200)
        translated_utterance = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return translated_utterance
"""

class MTModel:
    def __init__(self):
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to(self.device)
        #self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        # Add your key and endpoint
        #
        key = "YOUR_KEY"
        endpoint = "https://api.cognitive.microsofttranslator.com"

        # location, also known as region.
        # required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
        location = "australiaeast"

        path = '/translate'
        self.constructed_url = endpoint + path

        self.params = {
            'api-version': '3.0',
            'from': 'en',
            'to': 'zh'
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


def timeout_handler(num, stack):
    raise Exception("API-timeout")

def main():
    mtModel = MTModel()
    logging.info('Machine translation model is loaded!')
    flag, first_audio_time = False, 0

    def removeDupWithOrder(s):
        return "".join(OrderedDict.fromkeys(s))

    while True:
        if not flag:
            logging.info('INTO WHILE LOOP')
            flag = True
            first_audio_time = time.monotonic()

        # Start out by waiting for a message on the gesture queue.
        in_bound_message = CCU.recv_block(audio_queue)
        # print(in_bound_message)

        # Once we get that message, see if it is a audio message, since
        # that is the only thing we care about.
        if in_bound_message['type'] == 'audio_turn':
            # In a real analytic we would now run the analysis to see if this
            # is an offensive action, but for this sample, we just assume that
            # all pointing is offensive.
            logging.info('Found an audio.')

            # Creates a simple messsage of type 'offensive' with the detail
            # 'pointing'.  The base_message function also populates the 'uuid'
            # and 'datetime' fields in the message (which is really just a
            # Python dictionary).
            out_bound_message = CCU.base_message('asr_result')
            # parc_out_bound_message = CCU.base_message('asr_type')
            if 'audio' in in_bound_message:
                logging.info('Get a base64-type audio.')
                dec = in_bound_message['audio'].encode('utf-8')
                audio = base64.b64decode(dec)
                # print('Got a audio with {} data.  Writing to {}.flac.'.format(len(audio), recorded_audio_name_index))
                # recorded_audio_name_index += 1
                with open(audio_path, 'wb') as file:
                    file.write(audio)

                # try:
                #     # api
                #     # openai.api_key = "YOUR_KEY"
                #     audio_file = open(audio_path, "rb")
                #     AudioSegment.from_wav(audio_file).export(audio_path.replace("wav", "mp3"), format="mp3")
                #     mp3_audio_file = open(audio_path.replace("wav", "mp3"), "rb")
                #     # signal.signal(signal.SIGALRM, timeout_handler)
                #     # signal.alarm(15)
                #     # result = retry_call(openai.Audio.transcribe, fkwargs={ "model": "whisper-1", "file": mp3_audio_file, "request_timeout": 15}, tries=3, delay=5)
                # except:
                #     logging.info("There is an issue with Whisper API key")
                #     logging.info("Use the local whisper-medium instead")
                #     result = {"text":"Sorry can you repeat?"}
                # model
                result = model.transcribe(audio_path)
                translated_text = result["text"]
                out_bound_message['original_text'] = translated_text
                logging.info('The ASR text is :{}'.format(result["text"]))

                try:
                    if in_bound_message['speaker'] == 'SME':
                        if translated_text:
                            language_ = detect(translated_text)
                            logging.info('Current speaking language is: {}'.format(language_))
                            # if language_ in ['en', 'sw', 'fi', 'no', 'et', 'so', 'nl', 'cy', 'de', 'tl', 'id', 'fr', 'sv', 'ko', 'pt']:
                            if language_ not in ['zh-cn', 'ko', 'zh-tw', 'ja', 'vi', 'th']:
                                logging.info('SME is speaking English sentence: {}'.format(translated_text))
                                translated_text = mtModel.execute(translated_text)
                                reduced_text = removeDupWithOrder(translated_text)
                                if len(reduced_text) / len(translated_text) <= 0.3:
                                    translated_text = reduced_text
                                # out_bound_message['asr_json'] = 'Sorry, we found an ASR error, please continue the conversation without waiting.'
                                logging.info('The translated text is :{}'.format(translated_text))
                except:
                    logging.info('Error with langdetect')

                # # get time length of the audio
                # enc = base64.b64encode(open(audio_path, "rb").read())
                # r = base64.decodebytes(enc)
                # s = io.BytesIO(r)
                # duration_seconds = AudioSegment.from_file(s).duration_seconds
                out_bound_message['start_seconds'] = in_bound_message['start_seconds']
                out_bound_message['end_seconds'] = in_bound_message['end_seconds']
                out_bound_message['asr_text'] = translated_text
                # parc_out_bound_message['asr_text'] = translated_text
                # delete audio
                os.remove(audio_path)
                # os.remove(audio_path.replace("wav", "mp3"))

            # elif 'dict' in in_bound_message:
            #     logging.info('Get a textual audio.')
            #     out_bound_message['asr_text'] = in_bound_message['dict']
            #     parc_out_bound_message['asr_text'] = in_bound_message['dict']
            #     out_bound_message['start_seconds'] = time.monotonic() - first_audio_time
            #     out_bound_message['end_seconds'] = float(out_bound_message['start_seconds'] + 4.0)

            triggers = [in_bound_message['uuid']]
            if 'trigger_id' in in_bound_message:
                triggers.append(in_bound_message['trigger_id'])
            out_bound_message['trigger_id'] = triggers
            out_bound_message['audio_id'] = in_bound_message['uuid']
            # parc_out_bound_message['trigger_id'] = in_bound_message['uuid']
            # distinguish speak identification
            out_bound_message['speaker'] = in_bound_message['speaker']
            # parc_out_bound_message['speaker'] = in_bound_message['speaker']
            # parc_out_bound_message['asr_type'] = 'CONSOLIDATED_RESULT'
            logging.info(out_bound_message)
            # logging.info(parc_out_bound_message)
            # Next we send it.
            CCU.send(output_queue, out_bound_message)
            # CCU.send(park_output_queue, parc_out_bound_message)

def test_whisper_api(audio_path):
    openai.api_key = "YOUR_KEY"
    audio_file = open(audio_path, "rb")
    result = openai.Audio.translate("whisper-1", audio_file)
    return result

if __name__ == "__main__":
    main()
    # audio_path = "/home/monashnlp/audio.wav"
    # result = model.transcribe(audio_path)
    # print(result)
    # text = test_whisper_api(audio_path)
    # print(text)