from TTS.api import TTS
import pydub
from pydub.playback import play
import numpy as np
import constant
import time
import signal
import sounddevice as sd
import io
import base64
from scipy.io import wavfile 

class TTSModel:
    def __init__(self):
        self.en_tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
        self.zh_tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=False, gpu=True)

    def signal_handler(self, signum, frame):
        raise Exception("Timed out!")

    def play_with_timer(self, song):
        play(song)
        return True

    def try_timer_play(self, song):
        signal.signal(signal.SIGALRM, self.signal_handler)
        signal.alarm(5)  # Ten seconds
        try:
            self.play_with_timer(song)
        except Exception:
            print ("Timed out!")

    def generate_tts_and_play(self, text, language, output_path="out_tts.wav"):
        try:
            o_path = constant.PROJECT_ABSOLUTE_PATH+'/'+output_path
            if language == 'zh':
                text = ''.join([ch if not ch.isascii() else ' ' for ch in text])
                text = text.strip()
                if not text.endswith('。'):
                    text += "。"
                # self.zh_tts.tts_to_file(text, file_path=o_path)
                wav = self.zh_tts.tts(text)
                sd.play(wav, 22050)
                sd.wait()
            else:
                wav = self.en_tts.tts(text, language='en', speaker=self.en_tts.speakers[3])
                sd.play(wav, 16000)
                sd.wait()

        except Exception as e:
            print('Some thing error with TTS: {}'.format(str(e)))

    def generate_tts(self, text, language, output_path="out_tts.wav"):
        if language == 'zh':
            text = ''.join([ch if not ch.isascii() else ' ' for ch in text])
            text = text.strip()
            if not text.endswith('。'):
                text += "。"
            # self.zh_tts.tts_to_file(text, file_path= 'audo.wav')
            wav = self.zh_tts.tts(text)
            # print(type(wav))
            wav = np.array(wav)
            wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
            # print("*"*90)
            # print(type(wav))
            wavfile.write('audio.wav', 22050, wav_norm.astype(np.int16))
        else:
            # self.en_tts.tts_to_file(text, language='en',  speaker=self.en_tts.speakers[3], file_path="audio.wav")
            wav = self.en_tts.tts(text, language='en', speaker=self.en_tts.speakers[3])
            wav = np.array(wav)
            wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
            wavfile.write('audio.wav', 16000, wav_norm.astype(np.int16))
        with open('audio.wav',"rb") as f:
            byte_stream = f.read() 
        return byte_stream

    def test_tts(self, text):
        model = self.zh_tts
        wav = self.zh_tts.tts(text)
        sd.play(wav, 16000)
        status = sd.wait()
        # print("-"*90)
        # print(wav)
        

if __name__ == "__main__":

    text = "周日的公告是在这两家银行与该国金融监管机构在瑞士进行了"
    tts_model = TTSModel()
    # tts_model.generate_tts_and_play2('欢迎使用莫纳什大学对话助手系统。', language='zh')
    stream = tts_model.generate_tts("欢迎使用莫纳什大学对话助手系统。", 'zh')
    b = base64.b64encode(stream)
    tts_model.generate_tts_and_play(text, "zh")
    # tts_model.generate_tts_and_play("Welcome to use monash dialogue system", 'en')
    tts_model.generate_tts("Welcome to use monash dialogue system", 'en')

    ## install two linux libs for sounddevice package: libportaudio2 and libasound-dev
