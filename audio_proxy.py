import base64
import socket

import pytz
from simple_websocket_server import WebSocketServer, WebSocket
import io
import numpy as np
import struct
from scipy.io.wavfile import write
import zmq
from ccu import CCU
from constant import *
from datetime import datetime


# only supports one connection
wavarr = np.array([])
filename = ''
# Do this just once when starting up your analytic.
audio_queue = CCU.socket(CCU.queues['PROCESSED_AUDIO'], zmq.PUB)
output_queue = CCU.socket(CCU.queues['RESULT'], zmq.PUB)
speaker_id = ''
session_start_time = -1
start_seconds = 0.0
end_seconds = 0.0

class Server(WebSocket):
    def handle(self):
        global wavarr, filename, speaker_id, session_start_time, start_seconds, end_seconds
        if (self.data == "START-FLE" or self.data == "START-SME"):
            if session_start_time == -1:
                session_start_time = datetime.now()
                start_seconds = 0.0
            else:
                start_seconds = (datetime.now() - session_start_time).total_seconds()
            print(self.data)
            speaker_id = self.data.split('-')[1]
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            # filename = 'D:\\audio_' + str(self.address) + '_' + dt_string + '.wav'
            filename = PROJECT_ABSOLUTE_PATH + '/recorded_audios/audio_{}_{}.wav'.format(dt_string, speaker_id)
            dir_path = os.path.dirname(filename)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        elif (self.data == "STOP-FLE" or self.data == "STOP-SME"):
            print(self.data)
            end_seconds = (datetime.now() - session_start_time).total_seconds()
            wavarr = np.array([])
            self.forwad_data()
        else:
            self.write_audio(self.data)

    def write_audio(self, data):
        global wavarr
        audio_input = data
        s = np.frombuffer(audio_input, dtype=np.float32)
        wavarr = np.append(wavarr, s)
        # test write the file, works so far
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
        write(filename, 16000, wavarr.astype(np.float32))

    def forwad_data(self):
        global speaker_id
        message = CCU.base_message('audio_turn')
        message['audio_id'] = message['uuid']
        message['speaker'] = speaker_id
        message['timestamp'] = start_seconds
        message['start_seconds'] = start_seconds
        message['end_seconds'] = end_seconds
        message['sample_rate'] = 16000
        message['bit_depth'] = 16
        # Set the timezone for Washington, D.C.
        timezone = pytz.timezone('America/New_York')
        # Get the current time in Washington
        washington_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z%z')
        message['current_time'] = washington_time
        enc = base64.b64encode(open(filename, "rb").read())
        enc = enc.decode('utf-8')
        message['audio'] = enc
        print('send audio')
        # print(message)
        # Do this every time you need to send a message (created as described above).
        CCU.send(audio_queue, message)
        
    # When other device is connected to the proxy server, this function is called.
    def connected(self):
        global wavarr, filename
        wavarr = np.array([])
        print(self.address, 'connected')

    def handle_close(self):
        global filename
        filename = ''
        print(self.address, 'closed')


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
        print("IP", IP)
    except Exception:
        print("Exception")
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

server = WebSocketServer(get_ip(), 8001, Server)
print("Audio proxy server is running!")
server.serve_forever()

