#!/usr/bin/env python3
import socket
import threading

def receive_messages(conn):
    while True:
        data = conn.recv(1024)
        if not data:
            print("Client disconnected.")
            break
        print(f"\nClient: {data.decode()}")
    conn.close()

def send_messages(conn):
    while True:
        msg = input("You: ")
        if msg.lower() == "exit":
            print("Chat ended.")
            conn.close()
            break
        conn.sendall(msg.encode())

def main():
    HOST = "127.0.0.1"
    PORT = 5050

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Server listening on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"Connected with {addr}")

    threading.Thread(target=receive_messages, args=(conn,), daemon=True).start()
    send_messages(conn)

if __name__ == "__main__":
    main()
