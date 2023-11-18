import logging
from time import sleep
from threading import Thread

import pytest
import websocket  # websocket-client

# Add path to source code
import sys, os
import zmq
from ccu import CCU

if os.getcwd().endswith('tests'):
    sys.path.insert(0, '../..')
elif os.path.exists('websocket_server'):
    sys.path.insert(0, '..')
from websocket_server import WebsocketServer


class TestClient():
    def __init__(self, port, threaded=True):
        self.received_messages = []
        self.closes = []
        self.opens = []
        self.errors = []

        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(f"ws://130.194.73.211:{port}/",
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        if threaded:
            self.thread = Thread(target=self.ws.run_forever)
            self.thread.daemon = True
            self.thread.start()
        else:
            self.ws.run_forever()

    def on_message(self, ws, message):
        self.received_messages.append(message)
        print(f"TestClient: on_message: {message}")

    def on_error(self, ws, error):
        self.errors.append(error)
        print(f"TestClient: on_error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.closes.append((close_status_code, close_msg))
        print(f"TestClient: on_close: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        self.opens.append(ws)
        print("TestClient: on_open")


if __name__ == '__main__':
    client = TestClient(port=9001)
    # assert client.ws.sock and client.ws.sock.connected
    action = CCU.socket(CCU.queues['ACTION'], zmq.SUB)
    flag = False
    while True:
        # Waits for an offensive message on the result queue.
        # For all such messages, it sends a text_popup message so that the
        # Hololens will notify the operator to not do that.
        if not flag:
            print("INTO WHILE LOOP")
            flag = True
        message = CCU.recv(action)
        if message and message['type'] == 'text-popup':
            logging.info('Found a remediation in the conversation.')
            client.ws.send(message['remediation'])
    # client.ws.close()
