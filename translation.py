import re

from ccu import CCU
import zmq
from langdetect import detect

output_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)
sub_queue = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
MT_ENGLISH_NAME = "English"
MT_MANDARIN_NAME = "Mandarin Chinese"

class MTModel:

    def __init__(self):
        return
    
    def detect_lanugage(self, text):
        language_code = None
        try:
            language_code = detect(text)
        except Exception as e:
            print("Detect language failed: ", e)
            print(f"Use: {MT_ENGLISH_NAME} as default")

        print("Detected language code is {}".format(language_code))
        return MT_ENGLISH_NAME if language_code not in ['zh-cn', 'ko', 'zh-tw', 'ja', 'vi', 'th'] else MT_MANDARIN_NAME

        # if language_code in ['zh-cn', 'zh-tw']:
        #     return MT_MANDARIN_NAME
        # elif language_code in ['en']:
        #     return MT_ENGLISH_NAME
        # else:
        #     print(f"Language: {language_code} is not supported, use {MT_ENGLISH_NAME} as default")
        #     return MT_ENGLISH_NAME
        
    def execute(self, input):
        msg = CCU.base_message("translation_request")
        lanugage_name = self.detect_lanugage(input)
        # cleaned_string = input
        punctuations = r"，。！？；：“”‘’【】《》（）［］｛｝〔〕【】〖〗「」『』〈〉"
        cleaned_string = re.sub("[%s]+" % punctuations, " ", input)
        msg["asr_text"] = cleaned_string
        msg["asr_language"] = lanugage_name
        CCU.send(output_queue, msg)
        # Wait results
        while True:
            # RESULT queue
            message = CCU.recv_block(sub_queue)

            # Look for translation messages
            if message is not None and 'type' in message and message['type'] == 'translation':
                
                text = message['text']
                translation = message['translation']
                source_language = message['source_language']
                target_language = message['target_language']
                audio_source = message['audio_source']

                return translation

if __name__ == "__main__":
    mtModel = MTModel()
    #print(mtModel.execute('他們應該用大型頻道,不傷害小孩、小孩、地區商業,你同意嗎?'))
    print(mtModel.execute('how old are you'))
