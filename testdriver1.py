
# This script simulates events coming out of the Hololens.
# It simulates several operator move events and one operator pointing event.
# There is commented out code to simulate a frame from a video feed.
# (But this is not used because it is so large it makes it hard to follow the logs.)


import base64
import datetime
import json
import logging
import time

import zmq

from ccu import CCU
from constant import *

seconds = 0
def increment_time(interval=1):
    global seconds
    time.sleep(interval)
    seconds = seconds + 1
    print(f'time: {seconds}')
    
def get_image(file_name):
    with open(file_name, mode='rb') as file: 
       fileContent = file.read()
    base64EncodedStr = base64.b64encode(fileContent).decode('utf-8')
    print(base64EncodedStr)
    return base64EncodedStr

    # logging.debug(f'decoded {base64.b64decode(base64EncodedStr)}')
    
logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)    
    
logging.info('Test driver is Running...')
    
gesture = CCU.socket(CCU.queues['GESTURE'], zmq.PUB)
imu = CCU.socket(CCU.queues['IMU'], zmq.PUB)
video = CCU.socket(CCU.queues['VIDEO_MAIN'], zmq.PUB)
audio = CCU.socket(CCU.queues['AUDIO_SELF'], zmq.PUB)
processed_audio = CCU.socket(CCU.queues['PROCESSED_AUDIO'], zmq.PUB)


# Simple script of actions.  10 seconds long

increment_time(5)
message = CCU.base_message('move')
message['direction'] = 0
message['distance'] = 12
print(message)
CCU.send(imu, message)

# increment_time(5)
# # now = datetime.datetime.now()
# # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# message = CCU.base_message('audio_status')
# message['status'] = 'started'
# message['sample_rate'] = 8000
# message['audio_id'] = '1'
# print(message)
# CCU.send(audio, message)
#
# increment_time(5)
# now = datetime.datetime.now()
# # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# message = CCU.base_message('audio')
# path = PROJECT_ABSOLUTE_PATH + '/M01000550_10s.flac'
# enc = base64.b64encode(open(path, "rb").read())
# enc = enc.decode('utf-8')
# message['audio'] = enc
# message['timestamp'] = datetime.datetime.timestamp(now)
# #print(message)
# CCU.send(audio, message)
# #
# increment_time(5)
# # # now = datetime.datetime.now()
# # # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# message = CCU.base_message('audio_status')
# message['status'] = 'ended'
# message['sample_rate'] = 8000
# message['audio_id'] = '1'
# print(message)
# CCU.send(audio, message)


increment_time(5)
message = CCU.base_message('audio')
path = PROJECT_ABSOLUTE_PATH + '/M01000550_10s.flac'
enc = base64.b64encode(open(path, "rb").read())
enc = enc.decode('utf-8')
message['audio'] = enc
message['speaker'] = 'FLE'
print(message)
CCU.send(audio, message)

# increment_time(5)
# message = CCU.base_message('audio')
# message['dict'] = "爸：剛纔聽興龍伯伯說，是向你彙報思想改造情況是不是？是呀！地主分子每個月都要向我彙報一次思想改造情況，這已經形成了慣例。爸，“四人幫”已經打到快一年了，黨中央已經召開了十一屆三中全會。以鄧小平同志爲核心的黨中央在全党進行了平凡“冤假錯案”，在全國各條戰線上進行撥亂反正。地主、富農也要摘掉帽子。今後地主的子弟同樣可以參軍保衛祖國，同樣可以考大學，同樣可以招工、提幹。今天你跟左伯伯還在管制呀！不管制能行嗎？爸，這是中央的精神，不是我說你，部隊早就貫徹了。你要加強學習呀，不然的話很危險！你是要犯錯誤的呀！難怪你跟正鵬送了一套軍裝，想和他們打成一片是不是？爸，我和他們打成一片又怎樣呢，我們要團結廣大羣衆，搞革命工作就是要人多一點嘛。再說，我入伍前和正鵬不是玩得很好嗎，我到了部隊後，他自知之明，怕連累我，就主動不跟我聯繫了。爸，剛纔我聽興龍伯伯講，正鵬哥不在學校教書了，這是爲什麼？現在貧下中農子弟多得很還要他幹什麼？爸，照你這樣說，正鵬哥是你跟他撤了！是我跟他撤了又怎樣呢？你如果真的這樣做了，就是你不對了，你想到沒有，你要“菩薩”就請“菩薩”，不要“菩薩”就撤“菩薩”，你覺得對得住正鵬哥嗎？再說他也是在爲我們國家培養人才呀。你對正鵬哥這樣不許他去做，那樣也不讓他去幹，你這不是在埋沒別人的前途，你是在犯罪，你知道嗎？你說夠了嗎？你好像看到我跟左正鵬做了什麼“缺德”事似的！我告訴你，我是你父親，是一位老革命者，用不着你來指教我！"
# message['speaker'] = 'SME'
# print(message)
# CCU.send(audio, message)

increment_time(10)
message = CCU.base_message('audio')
message['dict'] = '你干嘛？'
message['speaker'] = 'SME'
print(message)
CCU.send(audio, message)

increment_time(10)
message = CCU.base_message('audio')
message['dict'] = '我怎么感觉你很紧张'
message['speaker'] = 'FLE'
print(message)
CCU.send(audio, message)

increment_time(10)
message = CCU.base_message('point')
message['direction'] = 30
message['height'] = 30
print(message)
CCU.send(gesture, message)

increment_time(5)
message = CCU.base_message('audio')
message['dict'] = '你怎么一直瞪着我，也不说话，吓死我了！'
message['speaker'] = 'SME'
print(message)
CCU.send(audio, message)

increment_time(10)
message = CCU.base_message('head')
message['direction'] = 90
print(message)
CCU.send(imu, message)

increment_time(5)
message = CCU.base_message('audio')
message['dict'] = '我又没盯着你看。'
message['speaker'] = 'FLE'
print(message)
CCU.send(audio, message)

increment_time(10)
#base64str = get_image('garage1.jpg')
#message = CCU.base_message('frame')
#message['frame'] = base64str
#CCU.send(video, message)

logging.info('Script 1 Done') 