from pydub import AudioSegment
from pydub.playback import play
import subprocess

def speak_out(text):
	subprocess.call(["tts", "--text", text, "--model_name", "tts_models/en/ljspeech/glow-tts", "--vocoder_name", "vocoder_models/en/ljspeech/univnet", "--out_path", "out"])
	song = AudioSegment.from_wav("out")
	play(song)


speak_out("Mission Complete Again")