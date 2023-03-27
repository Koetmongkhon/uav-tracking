# receiver

import socket
import select

ip = "192.168.1.28"
port = 5001

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))

sock.setblocking(0)

print(f'Start listening.')

while True:
  # sock.settimeout(0.1)
  ready = select.select([sock], [], [], 0.1)
  if ready[0]:
    data, address = sock.recvfrom(1024)
    data = data.decode('utf-8')
    lat, lon, pitch, roll, yaw = data.split(',')
    print(f'Lat: {lat} || Lon: {lon} || Pitch: {pitch} Roll: {roll} || Yaw: {yaw}')
