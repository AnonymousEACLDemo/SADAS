# This is the script used for testing Ssri-asr-mt pipeline.

import base64
import socket
import time

from simple_websocket_server import WebSocketServer, WebSocket
import io
import numpy as np
import struct
from scipy.io.wavfile import write
import zmq
from ccu import CCU
import datetime
from constant import *
from pydub import AudioSegment

# Do this just once when starting up your analytic.
audio_queue = CCU.socket(CCU.queues['AUDIO_SELF'], zmq.PUB)
processed_audio_queue = CCU.socket(CCU.queues['PROCESSED_AUDIO'], zmq.PUB)
speaker_id = ''
arrive_time = time.monotonic()
start_sample = 0
first_audio_time = time.monotonic()
audio_id = None
audio_fragments = []
audio_fragments_counter = 0
filename = ''
sample_rate = 16000

class Server(WebSocket):
    def handle(self):
        global wavarr, speaker_id, arrive_time, audio_id, audio_fragments, sample_rate
        if (self.data == "START-FLE" or self.data == "START-SME"):
            print(self.data)
            now = datetime.datetime.now()
            dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
            audio_id = 'CCU_ccu_{}'.format(dt_string)
            arrive_time = time.monotonic()
            message = CCU.base_message('audio_status')
            message['sample_rate'] = sample_rate
            message['status'] = 'started'
            message['audio_id'] = audio_id
            print(message)
            # Do this every time you need to send a message (created as described above).
            CCU.send(audio_queue, message)
            # speaker_id = self.data.split('-')[1]

            # message = CCU.base_message('endpointer_event')
            # message['time_seconds'] = time.monotonic() - arrive_time
            # message['num_samples_added'] = 0
            # message['ep_type'] = 'OLIVE'
            # message['sample_index'] = 0
            # message['audio_id'] = audio_id
            # message['num_samples_added_seconds'] = 0
            # message['sample_rate'] = sample_rate
            # message['engine'] = 'OLIVE'
            # message['ep_event'] = 'START_OF_SAMPLES'
            # message['vendor'] = 'sri'
            # message['sample_seconds'] = 0
            # message['num_samples_added_seconds_latency'] = 0
            # message['segment_id'] = "0000"
            # message['audio_source'] = 'AUDIO_SELF'
            # message['timestamp'] = time.monotonic() - arrive_time
            # print(message)
            # # Do this every time you need to send a message (created as described above).
            # CCU.send(processed_audio_queue, message)

        elif (self.data == "STOP-FLE" or self.data == "STOP-SME"):
            # message = CCU.base_message('endpointer_event')
            # message['time_seconds'] = time.monotonic() - arrive_time
            # message['num_samples_added'] = 0
            # message['ep_type'] = 'OLIVE'
            # message['sample_index'] = 0
            # message['audio_id'] = audio_id
            # message['num_samples_added_seconds'] = 0
            # message['sample_rate'] = sample_rate
            # message['engine'] = 'OLIVE'
            # message['ep_event'] = 'END_OF_SAMPLES'
            # message['vendor'] = 'sri'
            # message['sample_seconds'] = 0
            # message['num_samples_added_seconds_latency'] = 0
            # message['segment_id'] = "0000"
            # message['audio_source'] = 'AUDIO_SELF'
            # message['timestamp'] = time.monotonic() - arrive_time
            # print(message)
            # # Do this every time you need to send a message (created as described above).
            # CCU.send(processed_audio_queue, message)

            print(self.data)
            message = CCU.base_message('audio_status')
            message['sample_rate'] = sample_rate
            message['status'] = 'ended'
            message['audio_id'] = audio_id
            print(message)
            # Do this every time you need to send a message (created as described above).
            CCU.send(audio_queue, message)

            # # delete all used images
            # for image in audio_fragments:
            #     os.remove(image)
            # audio_fragments = []


        else:
            global start_sample, first_audio_time
            if start_sample == 0:
                first_audio_time = time.monotonic()
            self.write_audio(self.data)
            self.forwad_data()

    def write_audio(self, data):
        wavarr = np.array([])
        global audio_fragments_name, audio_fragments_counter, audio_fragments, filename, sample_rate
        filename = PROJECT_ABSOLUTE_PATH + '/recorded_audios/sri_test/audio_{}.wav'.format(audio_fragments_counter)
        dir_path = os.path.dirname(filename)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        audio_fragments.append(filename)
        audio_fragments_counter += 1

        audio_input = data
        s = np.frombuffer(audio_input, dtype=np.float32)
        wavarr = np.append(wavarr, s)
        write(filename, sample_rate, wavarr.astype(np.float32))

    def forwad_data(self):
        global start_sample, first_audio_time, start_seconds, filename, audio_id, arrive_time, sample_rate
        message = CCU.base_message('audio')
        enc = base64.b64encode(open(filename, "rb").read())
        r = base64.decodebytes(enc)
        s = io.BytesIO(r)
        audio_length = AudioSegment.from_file(s).duration_seconds
        delta_sample = int(audio_length * sample_rate)
        enc = enc.decode('utf-8')

        message['audio'] = CCU.binary_to_base64string(open(filename, "rb").read())
        message['start_sample'] = start_sample
        start_sample += delta_sample
        message['sample_rate'] = sample_rate
        message['start_seconds'] = time.monotonic() - first_audio_time
        message['timestamp'] = time.monotonic() - arrive_time
        message['num_samples'] = delta_sample
        message['audio_id'] = audio_id
        message['bit_depth'] = 16
        # print(message)
        # Do this every time you need to send a message (created as described above).
        # CCU.send(processed_audio_queue, message)
        CCU.send(audio_queue, message)
        print(enc)
        
    # When other device is connected to the proxy server, this function is called.
    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')


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

server = WebSocketServer(get_ip(), 8001, Server)
print("Audio server running")
server.serve_forever()

