import subprocess
from pydub import AudioSegment
from pydub.playback import play


class TTSModel:

    def __init__(self):
        self.model_name_dict = {
            "en": "tts_models/en/ljspeech/glow-tts",
            "zh": "tts_models/zh-CN/baker/tacotron2-DDC-GST"
        }
        self.vocoder_name_dict = {
            "en": "vocoder_models/en/ljspeech/univnet",
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


if __name__ == "__main__":
    tts_model = TTSModel()
    tts_model.generate_tts_and_play('他們應該用大型頻道,不傷害小孩.', 'zh')
    tts_model.generate_tts_and_play('Could you check if the annotations of TA1 data use the same IDs as in the email below.', 'en')
