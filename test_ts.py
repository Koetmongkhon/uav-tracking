from dronekit import connect

import math

vehicle = connect('COM3', wait_ready=True)

def get_roll():
  if type(math.degrees(vehicle.attitude.roll)) == 'float':
      return math.degrees(vehicle.attitude.roll)
  else:
      return 0

def get_pitch():
  if type(math.degrees(vehicle.attitude.pitch)) == 'float':
      return math.degrees(vehicle.attitude.pitch)
  else:
      return 0

def get_yaw():
  if type(math.degrees(vehicle.attitude.yaw)) == 'float':
      return math.degrees(vehicle.attitude.yaw)
  else:
      return 0

print(f"Lat: {vehicle.location.global_frame.lat}")
print(f"Lon: {vehicle.location.global_frame.lon}")
print(f"Pitch: {get_pitch()}")
print(f"Roll: {get_roll()}")
print(f"Yaw: {get_yaw()}")
