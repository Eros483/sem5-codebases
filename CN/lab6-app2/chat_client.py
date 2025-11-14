#!/usr/bin/env python3
import socket
import threading

def receive_messages(client_socket):
    while True:
        data = client_socket.recv(1024)
        if not data:
            print("Server disconnected.")
            break
        print(f"\nServer: {data.decode()}")

def send_messages(client_socket):
    while True:
        msg = input("You: ")
        if msg.lower() == "exit":
            print("Chat ended.")
            client_socket.close()
            break
        client_socket.sendall(msg.encode())

def main():
    HOST = "127.0.0.1"
    PORT = 5050

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print("Connected to the server. Type messages below:")

    threading.Thread(target=receive_messages, args=(client_socket,), daemon=True).start()
    send_messages(client_socket)

if __name__ == "__main__":
    main()
