import socket

HOST = 'localhost'
PORT = 17785

def raise_error_windows(e):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(str(e).encode())

if __name__ == "__main__":
    raise_error_windows("Error messages")