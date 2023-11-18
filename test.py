import base64
import datetime
import json
import logging
import re

import ffmpeg
import numpy as np
import pytz
import torch
import whisper
from ccu import CCU
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import spacy
from spacy.language import Language

from constant import *
import io
from pydub import AudioSegment
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from enchant.checker import SpellChecker
from langdetect import detect
import asyncio

def main():
    # logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
    #                     format='%(asctime)s %(levelname)s %(message)s',
    #                     level=logging.DEBUG)
    # # model = whisper.load_model("medium")
    # #
    # # # logging.info('Whisper example is running...')
    # # # list_ = ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
    # # # print(list_[-3:])
    # #
    # # audio = CCU.base64string_to_binary("iv9h/4H/x/8nAHMAiwBpAFMASABnAI4A4QBMAa4B/gEnAisC9AGCAeQAPgC8/z//uP5W/kD+W/5s/jL+y/1P/c/8Ovyi+0v7afvt+9X84P3h/sz/vgDXAd0CbgMrA1oC3AGOAmgESQbUBvUFuwQ3BCQEoAMJAsn/i/3w+yz7Kfux+1b80vw5/eL9z/65/wcAsP/4/r7+U/9TAEkB0QEiAmUCvALwAngCYQHI/xL+0fxF/Iv8LP2s/dD99/2X/mr/4v+8/xX/pP7i/tD/wQCBARIClAIsA40DbAO6As4B5wANAF//8f7s/ir/bv+W/9//UgDFANEAYwDs/8P/7v9HAJ4A6AA0AaABFgJdAk0CxwEEAVgA1v9a/9X+fv5Y/kz+Gv7Q/XH9Af2L/OD7Zvs++3778/ut/LP91f72/+sA8AG7AigD8wKCAnkCZAMXBWYGpAbYBd0EQQS9A9QCQgFT/2j9/PtR+3/7G/yz/DH9jv3q/Vf+uv4A/+r+sv6m/iH/LQBnAWcC9gJcA6IDpwMwAwACeQC6/iv9NPwW/Lr8fv0U/kj+pv43/37/af/2/qX+n/70/tj/CAE+AgIDWQNfAxoDsALqAf8ABQBT/wb/G/9f/7D/7v8kAF4AgAB2ADEA1f+N/27/mv8YAMMAZwHIAQ4CPAI0At8BOQF3AM7/UP/x/rz+u/6+/q/+df4F/lT9nfzz+3L7IftA+737uvzh/fP+0/97ACIBlAGiAVIB/ACOATIDQQWmBs8GXgYJBvUFXQXwA/wB+v9L/jL9o/x3/IH8vvwo/ZH92v24/WH9Cv3E/Lj8Ff3z/TL/dgCsAd0CCASmBHMEjgNtAlsBQAAl/1n+Ev5E/o3+zv7Y/rD+Wf7c/Yn9bP2b/fj9i/5k/3YAoQGWAugCrQIsAsUBbgEBAYQAEwAHAEQApgDyAAUB5gCgAEkA6/+A/x7/5/7u/k7/+v/GAFoBtwHqAdsBdwH2AIQAIQCl/zX/IP9C/4z/pf+X/2r/9/5P/mj9i/zW+2P7efsQ/Ab9/v2d/h7/qf8uAHIAagA/AEgAIAGhAl4EWQWCBXgFlgXdBYcFdgTsAmcBTACq/zT/5v5+/iz+AP7+/fL9ff3h/Ef89fsC/Fj8Cv3l/b7+u//fAPABlQKqAlUCxgE3AZMAHgD4/xAAkAD5ACIB7wB8AL7/+P5k/iP++v3o/TH+4v7F/40A8QD5ALUARADm/5f/W/8a//b+Lf/D/24A7AAlASEBBgHWALYAbwASAMr/vv82AMsATAGRAbMB0gHGAXQB9wBvAPD/f/9T/3v/oP/T/wAAQABhAEAA0P86/8H+c/4d/rj9k/3O/TP+iP6x/rr+vP6z/qL+jv6o/g==")
    # # print(f'Got a audio with {len(audio)} data.  Writing to audio.mp3.')
    # # with open('audio.mp3', 'wb') as file:
    # #     file.write(audio)
    # # result = model.transcribe(audio)
    # # print(result["text"])
    #
    # # path = PROJECT_ABSOLUTE_PATH + '/M01000550_10s.flac'
    # # enc = base64.b64encode(open(path, "rb").read())
    # # enc = enc.decode('utf-8')
    # # message = {'audio': enc}
    # #
    # # dec = message['audio'].encode('utf-8')
    # # image_64_decode = base64.b64decode(dec)
    # # image_result = open('1.wmv', 'wb')  # create a writable image and write the decoding result
    # # image_result.write(image_64_decode)
    # #
    # # print(enc)
    #
    # import socket
    # def get_ip():
    #     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     s.settimeout(0)
    #     try:
    #         # doesn't even have to be reachable
    #         s.connect(('10.254.254.254', 1))
    #         IP = s.getsockname()[0]
    #     except Exception:
    #         IP = '127.0.0.1'
    #     finally:
    #         s.close()
    #     return IP
    # print(get_ip())

    # filename = PROJECT_ABSOLUTE_PATH + '/recorded_videos/video_chunk_{}.mp4'.format(100)
    # ffmpeg.input(PROJECT_ABSOLUTE_PATH + '/recorded_videos/frame_%d.jpg', pattern_type='glob',
    #              framerate=15).output(filename).run()
    # print('save chunk: {}'.format(filename))


    # import time
    # start = time.monotonic()
    #
    # context = ['c', 'd']
    # norm_context = context[-3:]
    # print(norm_context)
    #
    # list_ = ['a', 'b']
    # context.extend(list_)
    # print(context)
    #
    # stop = time.monotonic()
    # print("The time of the run: {:03f}".format(stop - start))

    # operation = 1
    # print('Before, operation is {}'.format(operation))
    # action = inquirer.select(
    #     message="Select operation:",
    #     choices=[
    #         "Operation 1",
    #         "Operation 2",
    #         Choice(value=None, name="Exit"),
    #     ],
    #     default=None,
    # ).execute()
    # print(action)
    # operation = 2 if action == 'Operation 2' else 1
    # print('After, operation is {}'.format(operation))
    #
    #
    filename = PROJECT_ABSOLUTE_PATH + '/recorded_audios_11_1_3/audio_01_11_2022_19_59_53_FLE.wav'
    enc = base64.b64encode(open(filename, "rb").read())
    # enc = enc.decode('utf-8')
    r = base64.decodebytes(enc)
    # q = np.frombuffer(r, dtype=np.float32)


    # audio = AudioSegment.from_file(filename)
    # print(audio.duration_seconds)


    s = io.BytesIO(r)
    print(AudioSegment.from_file(s).duration_seconds)

    max_error_count = 4
    min_text_length = 2

    # def is_in_english(quote):
    #     d = SpellChecker("en_US")
    #     d.set_text(quote)
    #     errors = [err.word for err in d]
    #     return False if (float(len(errors) / len(quote.split())) >= 0.67) else True
    #
    # print(is_in_english('中文'))
    # print(is_in_english(
    #     'Two things are 宇宙 infinite: the universe and human stupidity; 确定宇 and I\'m not sure about the universe.'))
    # print(is_in_english(
    #     '除了有时候whisper不自动翻译导致TTS不发声'))
    # print(is_in_english(
    #     'hello'))



    # class MTModel:
    #     def __init__(self):
    #         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #         self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh").to(self.device)
    #         self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    #
    #     def execute(self, input_chinese):
    #         inputs = self.tokenizer([input_chinese], return_tensors="pt").to(self.device)
    #         generated_tokens = self.model.generate(**inputs, max_length=200)
    #         translated_utterance = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    #
    #         return translated_utterance
    #
    # mtModel = MTModel()
    # logging.info('Machine translation model is loaded!')
    #
    # translated_text = mtModel.execute('hi, this is yuncheng hua')
    # print(translated_text)


    lst = ["What's", 'english1234!', 'Engl1sh', 'not english 行中ワ', '行中ワ',
           "War doesn't show who's right, just who's left.", 'hi', 'hello', 'bye', 'byebye', 'haha', '查看防火墙状态',
           '簡直胡說八道,你從哪裡聽來的', 'metabolism', 'cheers', 'good night', 'good day', 'BYE']

    allowed_strings = [detect(string) for string in lst]
    print(allowed_strings)
    print(detect('hi'))



    # path = PROJECT_ABSOLUTE_PATH + '/recorded_audios_11_07_03/ta1.jsonl'
    #
    # f = open(path, "r")
    # lines = f.readlines()
    #
    # count = 0
    # timestamp = 0.0
    # res_lines = []
    # for line in lines:
    #     x_ = json.loads(line)
    #     x_['message']['type'] = 'frame'
    #     x_['message']['count'] = count
    #     count += 1
    #     x_['message']['timestamp'] = timestamp
    #     timestamp += 0.066666667
    #     x_['message']['width'] = 896
    #     x_['message']['height'] = 504
    #     x_['message']['depth'] = 3
    #     x_['message']['num_samples'] = 3281
    #     string_ = json.dumps(x_)
    #     res_lines.append(string_)
    #
    # json_file = PROJECT_ABSOLUTE_PATH + '/recorded_audios_11_07_03/ta1_new.jsonl'
    # f = open(json_file, "w")
    # f.writelines(res_lines)
    # f.close()


if __name__ == "__main__":

    # global count_
    #
    #
    # async def my_coroutine():
    #     print("Before awaiting coroutine")
    #     await asyncio.sleep(1)
    #     print("After awaiting coroutine")
    #     print("Coroutine finished")
    #
    #
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(my_coroutine())

    # Set the timezone for Washington, D.C.
    timezone = pytz.timezone('America/New_York')
    # Get the current time in Washington
    washington_time = datetime.datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z%z')
    print(washington_time)

    clients = []
    client = {'id': 1, 'handler': '<websocket_server.websocket_server.WebSocketHandler object at 0x7f921446a530>', 'address': '(\'49.127.38.35\', 45946)'}
    clients.append(client)

    client = {'id': 2, 'handler': '<websocket_server.websocket_server.WebSocketHandler object at 0x7f921446a860>', 'address': '(\'130.194.73.144\', 55900)'}

    clients.append(client)

    if client in clients:
        clients.remove(client)
    print("Client(%d) disconnected" % client['id'])


