import base64
import datetime
import os

import numpy as np
from simple_websocket_server import WebSocketServer, WebSocket
from PIL import Image
import io
import zmq
from ccu import CCU
import socket
import ffmpeg

from constant import PROJECT_ABSOLUTE_PATH

# Define a 2D array to contain video frames with a fix dimension. Say, we call is Arr.
# 1 dimension will store a single frame data, the length is 896x504x3
# Another dimension will store the frames chunk. The FPS is 15, so 1 minute video will require 900 elements
video_arr = list()
video_queue = CCU.socket(CCU.queues['VIDEO_MAIN'], zmq.PUB)
chunk_counter = 0
frame_counter = 0
CHUNK_FRAME_SIZE = 450
frames_per_second = 15
dir_path = ''

class Server(WebSocket):
    def handle(self):
        global video_arr, chunk_counter, frame_counter, dir_path
        if(self.data == "START"):
            print("START MESSAGE")
            now = datetime.datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            dir_path = PROJECT_ABSOLUTE_PATH + "/recorded_videos/{}".format(dt_string)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        elif(self.data == "STOP"):
            print("STOP MESSAGE")
            self.store_chunk()
        else:
            # print('len(video_arr): {}'.format(len(video_arr)))
            if frame_counter == CHUNK_FRAME_SIZE:
                self.store_chunk()
            self.write_data()

    def write_data(self):
        global video_arr, frame_counter
        data = self.data
        pilFrame = Image.open(io.BytesIO(data))
        video_arr.append(pilFrame)
        frame_counter += 1

    def store_chunk(self):
        print('Chunk writing...')
        global video_arr, chunk_counter, frame_counter, dir_path
        images = list()
        # now = datetime.datetime.now()
        # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        for i, pilFrame in enumerate(video_arr):
            filename = dir_path + "/{:04d}.jpg".format(i)
            pilFrame.save(filename)
            # print('write image: {}'.format(filename))
            images.append(filename)

        # clean cache
        video_arr = list()
        frame_counter = 0

        # save chunk
        filename = dir_path + '/video_chunk_{}.mp4'.format(chunk_counter)
        stream = ffmpeg.input(dir_path + '/*.jpg', pattern_type='glob', framerate=frames_per_second)
        stream = ffmpeg.output(stream, filename)
        ffmpeg.run(stream, overwrite_output=True)

        print('save chunk: {}'.format(filename))
        chunk_counter += 1
        self.forward_latest_chunk(images)

        # delete all used images
        for image in images:
            os.remove(image)

    # def show_video(self):
    #     data = self.data
    #     pilFrame = Image.open(io.BytesIO(data))
    #     frame = np.asarray(pilFrame)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     title = 'webcam' + str(frame.shape)
    #     cv2.imshow(title, frame)
    #     if cv2.waitKey(5) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()

    def get_image(self, file_name):
        with open(file_name, mode='rb') as file:
            fileContent = file.read()
        base64EncodedStr = base64.b64encode(fileContent).decode('utf-8')
        # print(base64EncodedStr)
        return base64EncodedStr

    # This function should forward the latest video chunk to CCU server
    def forward_latest_chunk(self, images):
        print('sending messages into ZMQ...')
        count = 0
        timestamp = 0.0
        # num_samples = CHUNK_FRAME_SIZE + 1
        width = 896
        height = 504
        depth = 3
        for image in images:
            message = CCU.base_message('webcam')
            # columbia_message = CCU.base_message('frame')
            enc = self.get_image(image)
            message['image'] = enc
            # columbia_message['image'] = enc
            # columbia_message['count'] = count
            count += 1
            # columbia_message['timestamp'] = timestamp
            message['timestamp'] = timestamp
            timestamp += 1.0 / frames_per_second
            message['width'] = width
            message['height'] = height
            message['depth'] = depth
            message['format'] = 'jpg'
            # columbia_message['width'] = width
            # columbia_message['height'] = height
            # columbia_message['depth'] = depth
            # columbia_message['num_samples'] = num_samples
            # columbia_message['container_name'] = 'columbia-vision'
            # print(message)
            CCU.send(video_queue, message)
            # CCU.send(video_queue, columbia_message)

    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')
        # cv2.destroyAllWindows()

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

server = WebSocketServer(get_ip(), 8000, Server)
print("Webcam proxy server is running!")
server.serve_forever()
