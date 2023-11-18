import logging
import socket

from websocket_server import WebsocketServer


class TestServer(WebsocketServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_fn_new_client(self.new_client)
        self.set_fn_client_left(self.client_left)
        self.set_fn_message_received(self.message_received)

    # Called for every client connecting (after handshake)
    def new_client(self, client, server):
        print("New client connected and was given id %d" % client['id'])
        print(client)
        server.send_message(client, "hello, client {}".format(client['id']))
        server.send_message_to_all("Hey all, a new client {} has joined us".format(client['id']))
        self.clients.append(client)
        # print("The current message proxy clients are {}".format(self.clients))

    # Called for every client disconnecting
    def client_left(self, client, server):
        if client in self.clients:
            self.clients.remove(client)
            self.clients.remove(client)
            # print("The current message proxy clients are {}".format(self.clients))
        print("Client(%d) disconnected" % client['id'])


    # Called when a client sends a message
    def message_received(self, client, server, message):
        # if len(message) > 200:
        #     message = message[:200] + '..'
        # print("Client(%d) said: %s" % (client['id'], message))
        # if client['id'] == self.clients[-1]['id']:
        for client_ in self.clients:
            if client_['id'] != client['id']:
                server.send_message(client_, message)



if __name__ == "__main__":
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
        print('IP is {}'.format(IP))
        return IP

    # change to 8002
    PORT = 8002
    server = TestServer(host=get_ip(), port=PORT, loglevel=logging.DEBUG)
    print("Message proxy server is running!")
    server.run_forever()
    server.server_close()

