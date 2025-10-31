import socket
import sys
import time

def traceroute(dest_name, max_hops=30, timeout=2.0):
    dest_addr = socket.gethostbyname(dest_name)
    print(f"traceroute to {dest_name} ({dest_addr}), max {max_hops} hops\n")

    port = 33434
    ttl = 1

    while True:
        # set up sockets
        recv_socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
        recv_socket.settimeout(timeout)
        send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        send_socket.setsockopt(socket.SOL_IP, socket.IP_TTL, ttl)

        send_socket.sendto(b'', (dest_addr, port))
        start_time = time.time()

        curr_addr = None
        try:
            _, addr = recv_socket.recvfrom(512)
            curr_addr = addr[0]
            elapsed = (time.time() - start_time) * 1000
            try:
                host = socket.gethostbyaddr(curr_addr)[0]
            except socket.herror:
                host = curr_addr
            print(f"{ttl}\t{host} ({curr_addr})\t{elapsed:.1f} ms")
        except socket.timeout:
            print(f"{ttl}\t*")
        finally:
            send_socket.close()
            recv_socket.close()

        ttl += 1
        if curr_addr == dest_addr or ttl > max_hops:
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: sudo python3 cna1.py <destination>")
        sys.exit(1)

    traceroute(sys.argv[1])