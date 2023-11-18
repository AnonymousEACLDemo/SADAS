import socket

from simple_websocket_server import WebSocketServer, WebSocket


class SimpleEcho(WebSocket):
    # handle: gets called when there is an incoming message from the client endpoint
    # self.address: TCP address port tuple of the endpoint
    # self.opcode: the WebSocket frame type (STREAM, TEXT, BINARY)
    # self.data: bytearray (BINARY frame) or unicode string payload (TEXT frame)
    # self.request: HTTP details from the WebSocket handshake (refer to BaseHTTPRequestHandler)
    def handle(self):
        # echo message back to client
        self.send_message(self.data)

    # connected: called when handshake is complete
    def connected(self):
        print(self.address, 'connected')

    # handle_close: called when the endpoint is closed or there is an error
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

server = WebSocketServer(get_ip(), 8000, SimpleEcho)
server.serve_forever()
