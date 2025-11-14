#!/usr/bin/env python3
import socket
import threading

# Handle each client connection
def handle_client(conn, addr):
    print(f"[+] Connected by {addr}")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        print(f"Received from {addr}: {data.decode()}")
        conn.sendall(b"ACK")
    conn.close()
    print(f"[-] Connection closed: {addr}")

def main():
    HOST = "127.0.0.1"  # localhost
    PORT = 5050         # arbitrary port

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()

if __name__ == "__main__":
    main()
