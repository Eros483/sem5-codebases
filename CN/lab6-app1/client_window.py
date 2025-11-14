#!/usr/bin/env python3
import socket
import time

def main():
    HOST = "127.0.0.1"
    PORT = 5050
    WINDOW_SIZE = 5
    TOTAL_MSGS = 15

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    next_msg = 0
    acked = 0

    # Send initial 5 messages
    while next_msg < WINDOW_SIZE and next_msg < TOTAL_MSGS:
        msg = f"hello {next_msg+1}"
        client_socket.sendall(msg.encode())
        print(f"Sent: {msg}")
        next_msg += 1

    while acked < TOTAL_MSGS:
        ack = client_socket.recv(1024)
        if not ack:
            break
        print(f"Received: {ack.decode()}")
        acked += 1

        # Send next message as window slides
        if next_msg < TOTAL_MSGS:
            msg = f"hello {next_msg+1}"
            client_socket.sendall(msg.encode())
            print(f"Sent: {msg}")
            next_msg += 1

        time.sleep(0.3)

    client_socket.close()

if __name__ == "__main__":
    main()
