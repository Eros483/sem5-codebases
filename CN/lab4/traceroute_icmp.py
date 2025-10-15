import os
import sys
import struct
import time
import select
from socket import *

ICMP_ECHO_REQUEST = 8
MAX_HOPS = 30
TIMEOUT = 2.0
TRIES = 2

TARGETS=["amazon.com", "google.com", "mit.edu", "wikipedia.org"]

def checksum(string):
    """
    Pre-provided code for calculating the checksum of a packet.
    """
    csum= 0
    countTo= (len(string) // 2) * 2

    count= 0
    while count < countTo:
        thisVal= (string[count + 1]) * 256 + (string[count])
        csum= csum + thisVal
        csum= csum & 0xffffffff
        count= count + 2

    if countTo<len(string):
        csum=csum + (string[len(string) - 1])
        csum=csum & 0xffffffff

    csum=(csum >> 16) + (csum & 0xffff)
    csum=csum + (csum >> 16)
    answer=~csum
    answer=answer & 0xffff
    answer=answer >> 8 | (answer << 8 & 0xff00)
    return answer

def build_packet():
    """
    Builds an icmp packet.
    """
    myChecksum=0
    myID=os.getpid() & 0xFFFF

    header=struct.pack("bbHHh", ICMP_ECHO_REQUEST, 0, myChecksum, myID, 1)
    data=struct.pack("d", time.time())

    myChecksum=checksum(header + data)

    header=struct.pack("bbHHh", ICMP_ECHO_REQUEST, 0, myChecksum, myID, 1)
    packet=header + data
    return packet

def get_route(hostname):
    """
    Discovers each hop beween two machines.
    """
    overall_start=time.time()

    timeLeft=TIMEOUT
    for ttl in range(1, MAX_HOPS):
        if time.time()-overall_start >3:
            return

        for _ in range(TRIES):
            if time.time()-overall_start>3:
                return
            
            _ =gethostbyname(hostname)
            mySocket=socket(AF_INET, SOCK_RAW, getprotobyname("icmp"))

            mySocket.setsockopt(IPPROTO_IP, IP_TTL, struct.pack("I", ttl))
            mySocket.settimeout(TIMEOUT)

            try:
                d=build_packet()
                mySocket.sendto(d, (hostname, 0))
                t=time.time()
                startedSelect=time.time()

                whatReady=select.select([mySocket], [], [], timeLeft)
                howLongInSelect=time.time()-startedSelect

                if whatReady[0]==[]:
                    print(f"{ttl}------Request timed out.---------")
                    continue

                recvPacket, addr=mySocket.recvfrom(1024)
                timeReceived=time.time()
                timeLeft=timeLeft-howLongInSelect

                icmpHeader=recvPacket[20:28]
                types, _, _, _, _=struct.unpack("bbHHh", icmpHeader)

                if types==11:
                    print(f"{ttl}    rtt={(timeReceived - t) * 1000:.2f} ms    {addr[0]}")
                elif types==3:
                    print(f"{ttl}    rtt={(timeReceived - t) * 1000:.2f} ms    {addr[0]}")
                elif types==0:
                    print(f"{ttl}    rtt={(timeReceived - t) * 1000:.2f} ms    {addr[0]}")
                    return
                else:
                    print("ERROR: Unexpected ICMP type.")

            except timeout:
                continue
            finally:
                mySocket.close()

if __name__=="__main__":
    for target in TARGETS:
        print("--------------------------------")
        print(f"Traceroute to {target}")
        get_route(target)
        print("--------------------------------")
        print("\n")