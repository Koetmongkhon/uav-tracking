# sender

from dronekit import connect

import math
import socket
import random

vehicle = connect('/dev/ttyACM0', wait_ready=True)

ip = "192.168.1.28"
port = 5001

for i in range(10):
    lat = vehicle.location.global_frame.lat + random.randint(0,10)
    lon = vehicle.location.global_frame.lon + random.randint(0,10)
    msg = str(lat)+','+str(lon)
    print(msg)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(msg.encode('utf-8'), (ip, port))

sock.close()
