#!/usr/bin/env python

import socket
import time

IPAddress="172.22.22.2" # Robot IP 
Port=30001

#Establish socket connection
try:
  sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  sock.settimeout(2)
  sock.connect((IPAddress,Port))
except Exception as e:
  print("Could not connect to robot: ",e)
  print("Socket connect failed!")
  exit()

#send script to robot
#sock.sendall(cmd)

#wait for a few seconds and close socket
time.sleep(3)
if(sock):
  sock.close()
sock=None

