#!/usr/bin/env python3
import socket

CLIENT_NAME = "Client of ARNAB MANDAL"
HOST = "127.0.0.1"
PORT = 5050


def main():
    num = int(input("Enter an integer (1-100): "))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        msg = f"{CLIENT_NAME}||{num}"
        s.sendall(msg.encode())

        data = s.recv(1024).decode()
        server_name, server_num = data.split("||")
        server_num = int(server_num)

        print("\n--- CLIENT OUTPUT ---")
        print(f"Client name: {CLIENT_NAME}")
        print(f"Server name: {server_name}")
        print(f"Client number: {num}, Server number: {server_num}")
        print(f"Sum: {num + server_num}")


if __name__ == "__main__":
    main()
