# The status server that will answer the app's requests about whether the T2 system is busy or not.
from simple_websocket_server import WebSocketServer, WebSocket

import sched
import time
import zmq
from ccu import CCU
import socket
import random


class Server(WebSocket):
    def handle(self):
        # self.data is corresponding to the requests sent from the mobile app periodically to detect the TA2 status.
        # print(self.data)

        # Always available.
        if (self.data == "status"):
            # r = random.randint(0, 100)
            # status = "busy"
            # if (r < 10):
            #     status = "available"

            status = "available"
            data = '{"status":"Status", "message":"' + str(status) + '"}'
            self.send_message(data)

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
        print("IP", IP)
    except Exception:
        print("Exception")
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# hostname = socket.gethostname()
# IPAddr = socket.gethostbyname(hostname)
# print("Running on " + IPAddr)
server = WebSocketServer(get_ip(), 8008, Server)
print("Status server running")
server.serve_forever()

