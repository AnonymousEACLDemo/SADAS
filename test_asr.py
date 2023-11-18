
import openai
import os
import glob
from pydub import AudioSegment
import whisper
old_key = "YOUR_KEY"
openai.api_key = old_key
model = whisper.load_model("medium")

def test_whisper_api(audio_path):
    
    audio_file = open(audio_path, "rb")
    try:
        result = openai.Audio.translate("whisper-1", audio_file)
    except:
        print("there is some error")
        print("using local model")
        result = model.transcribe(audio_path)

    return result['text']

# def audio_segment(audio):


if __name__ =="__main__":

    audio_dir = "/home/monashnlp/Desktop/wav-test"
    for audio_path in glob.glob(os.path.join(audio_dir, "*.wav")):

        # audio_path = "/home/monashnlp/Desktop/over-25mb-test.wav"
        text = test_whisper_api(audio_path)
        print(text)