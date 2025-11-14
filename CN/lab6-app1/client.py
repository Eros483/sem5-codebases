#!/usr/bin/env python3
import socket

def main():
    HOST = "127.0.0.1"
    PORT = 5050

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    for i in range(5):
        msg = f"hello {i+1}"
        client_socket.sendall(msg.encode())
        data = client_socket.recv(1024)
        print(f"Server replied: {data.decode()}")

    client_socket.close()

if __name__ == "__main__":
    main()
