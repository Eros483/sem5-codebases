#!/usr/bin/env python3
import socket

SERVER_NAME = "Server of ARNAB MANDAL"
HOST = "127.0.0.1"  # localhost
PORT = 5050


def main():
    print(f"Starting server: {SERVER_NAME}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Server listening on {HOST}:{PORT}\n")

    while True:
        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")

            data = conn.recv(1024).decode()
            if not data:
                continue

            try:
                client_name, client_num = data.split("||")
                client_num = int(client_num)
            except:
                print("Invalid message format. Closing.")
                break

            print(f"Client name: {client_name}")
            print(f"Server name: {SERVER_NAME}")

          
            if not (1 <= client_num <= 100):
                print("Received out-of-range number. Shutting server down...")
                break

            server_num = 42  
            print(f"Client number: {client_num}, Server number: {server_num}, Sum: {client_num + server_num}\n")

            reply = f"{SERVER_NAME}||{server_num}"
            conn.sendall(reply.encode())

    server_socket.close()
    print("Server terminated.")


if __name__ == "__main__":
    main()
