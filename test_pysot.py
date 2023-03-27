import torch
import time
import cv2
import argparse

import os

import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

# from ts import find_lat_lon, changetorad

from random import randint

ap = argparse.ArgumentParser()
ap.add_argument('-ct', '--cthreshold', default=0.10, help='confidence Threshold.')
ap.add_argument('-w', '--webcam', default='n', help='Use webcam. (y/n)')
ap.add_argument('-i', '--input', default=None, help='Source path.')
ap.add_argument('-c', '--config', type=str, help='config file', default='siamrpn_alex_dwxcorr_otb/config.yaml')
ap.add_argument('-sn', '--snapshot', type=str, help='model name', default='siamrpn_alex_dwxcorr_otb/model.pth')
args = ap.parse_args()

CONFIDENCE_THRESHOLD = args.cthreshold

class Status:
  def __init__(self, detect, clk, select, track, pause):
    self.DETECTING = detect
    self.CLICKED = clk
    self.SELECTED = select
    self.TRACKING = track
    self.PAUSE = pause

status = Status(True, False, False, False, False)
bbox = []
selected_bbox = []
cursor_x, cursor_y = 0, 0
writer = None

def detection(model, frame, confidence, loss=False):
  # start = time.time()

  # Prediction
  print('Frame:', frame)
  results = model(frame)
  print('Result:', results)

  # Extract predictions
  detected = results.pandas().xyxy[0]

  # Detect only human and only when confidence exceeds the threshold.
  detected = detected[(detected['confidence'] > confidence) & (detected['class'] == 0)]

  bbox = []

  for i in detected.iterrows():
    box = i[1][0:4].astype(int).values
    bbox.append(box)
    cv2.rectangle(frame, (box[0], box[1]),  # x1, y1
                  (box[2], box[3]),  # x2, y2 
                  (0, 255, 0), 2)  # color, border width
    
  if loss:
    return bbox

  # detection_end = time.time() - start
  # fps = None
  # if detection_end == 0 :
  #   fps = 999
  # else :
  #   fps = 1/detection_end

  # Show FPS
  # cv2.putText(frame, f'FPS: {fps:.2f}', (20, 40),
  #             cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 6, cv2.LINE_AA)
  # cv2.putText(frame, f'FPS: {fps:.2f}', (20, 40),
  #             cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

  cv2.imshow('output', frame)

  return bbox

def on_click(event, x, y, p1, p2):
  # Set these variables global
  global cursor_x, cursor_y, status

  # If left click, change global cursor_x, cursor_y
  # and set CLICKED to TRUE
  if event == cv2.EVENT_LBUTTONDOWN:
    cursor_x, cursor_y = x, y
    # print(cursor_x, cursor_y)
    if not status.TRACKING:
      status.CLICKED = True

def crop_image(img, x1, x2, y1, y2):
  x1 += 5
  y1 = 0 if y1 <= 5 else y1-5
  x2 += 5
  y2 = 0 if y2 <= 5 else y2-5
  if x1>=y1:
    if x2>=y2:
      cropped = img[y2:x2, y1:x1]
    else:
      cropped = img[x2:y2, y1:x1]
  else:
    if x2>=y2:
      cropped = img[y2:x2, x1:y1]
    else:
      cropped = img[x2:y2, x1:y1]
  return cropped

# Load model
yolo_model = torch.hub.load('D:/Senior-Project/Code/YOLO/v5', 'custom', path='./yolov5n.pt', source='local')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/Senior Project/Code/YOLO/v5/yolov5n', force_reload=True) 

# load config
cfg.merge_from_file(args.config)
cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
device = torch.device('cuda' if cfg.CUDA else 'cpu')

# Create tracker instance
load_model = ModelBuilder()
load_model.load_state_dict(torch.load(args.snapshot,
    map_location=lambda storage, loc: storage.cpu()))
load_model.eval().to(device)

tracker = build_tracker(load_model)

# gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! autovideosink

cap_str = ''

start_time = time.time()
if args.webcam == 'y':
  # cap = cv2.VideoCapture(0)
  cap_str = 'autovideosrc ! videoconvert ! appsink'
elif args.webcam == 'u':
  cap_str = "udpsrc port=5000 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! rtph264depay ! avdec_h264 ! decodebin ! videoconvert ! appsink"
else :
  video_input = args.input
  if video_input == None:
    cap_str = './video/v..special-เดินหลบต้นไม้ แล้วเปลี่ยนเสื้อ แล้วเดินออกมา.mp4'
    # video_input = 'production ID_3687560.mp4'
    # video_input = './video/Special-คนเยอะ.mp4'

print(cap_str)
cap = cv2.VideoCapture(cap_str)

file_name = './output/v..special-เดินหลบต้นไม้ แล้วเปลี่ยนเสื้อ แล้วเดินออกมา.mp4'

# gst_str = 'udpsrc port=5000 ! application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! appsink'
# gst_str = 'udpsrc port=5000 ! application/x-rtp,media=video, rtpjitterbuffer latency=300 ! encoding-name=H264 ! rtph264depay ! avdec_h264 ! appsink drop=1'


# cap = cv2.VideoCapture(gst_str)

# Initialize window
cv2.namedWindow('output')

# Set mouse click callback function
cv2.setMouseCallback('output', on_click)

frame_count = 0

while True:
  if not(status.PAUSE):
    ret, frame = cap.read()
    if ret == False:
      break

    frame = cv2.resize(frame, (1280, 720))

    if status.DETECTING:
      cv2.putText(frame, f'Detecting', (5, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
      bbox = detection(yolo_model, frame, CONFIDENCE_THRESHOLD)
      print("Detection :", bbox)

    if status.CLICKED:
      # print(bbox)
      for box in bbox:
        x1, y1, x2, y2 = box

        # Check if the cursor's position is in the range of the box
        if cursor_x > x1 and cursor_x < x2 and cursor_y > y1 and cursor_y < y2:
          print("CLICKED", x1, y1, x2, y2)

          # If bbox is valid, turn off detection
          status.DETECTING = False
          status.SELECTED = True
          selected_bbox = [x1, y1, x2, y2]
          break

      status.CLICKED = False

    if status.SELECTED:
      # Change bbox format to tracker's format
      bbox = [selected_bbox[0], # x1
        selected_bbox[1], # y1
        selected_bbox[2]-selected_bbox[0], # x2 - x1
        selected_bbox[3]-selected_bbox[1]] # y2 - y1
      print(bbox)
      tracker.init(frame, bbox)
      status.SELECTED = False
      status.TRACKING = True
      cropped = crop_image(frame, bbox[0], bbox[1], 
                bbox[0] + bbox[2], bbox[1] + bbox[3])

    if status.TRACKING:
      cv2.putText(frame, f'Tracking', (5, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
      t1 = time.time()
      outputs = tracker.track(frame)
      if 'polygon' in outputs:
        polygon = np.array(outputs['polygon']).astype(np.int32)
        cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                      True, (0, 255, 0), 3)
        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
        mask = mask.astype(np.uint8)
        mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
      else:
        bbox = list(map(int, outputs['bbox']))
        cv2.rectangle(frame, (bbox[0], bbox[1]),
                      (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                      (0, 255, 0), 3)
        # cv2.putText(frame, f'lat: 13.74', (5,690),
        #         cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 6, cv2.LINE_AA)
        # cv2.putText(frame, f'lat: 13.74', (5,690),
        #         cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, f'lon: 100.52', (5, 710),
        #         cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 6, cv2.LINE_AA)
        # cv2.putText(frame, f'lon: 100.52', (5, 710),
        #         cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, f'FPS: {fps:.2f}', (20, 40),
        #       cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 6, cv2.LINE_AA)
      cv2.imshow('output', frame)
      if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(file_name, fourcc, 30,
          (frame.shape[1], frame.shape[0]), True)
        # some information on processing single frame
        # if total > 0:
        #   elap = (end - start)
        #   print("[INFO] single frame took {:.4f} seconds".format(elap))
        #   print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
      writer.write(frame)
                      
  key = cv2.waitKey(1)

  if key == ord('q'):
    break
  elif key == 32:
    status.PAUSE = not(status.PAUSE)

# delete custom train data
# for i in range(0, max_train_num):
#   imTrainName = cus_im_train_path + str(i) + '.jpg'
#   labTrainName = cus_label_train_path + str(i) + '.txt'
#   if os.path.exists(imTrainName):
#     os.remove(imTrainName)
#   if os.path.exists(labTrainName):
#     os.remove(labTrainName)

# delete custom validate data
# for i in range(0, max_val_num):
#   imValName = cus_im_val_path + str(i) + '.jpg'
#   labValName = cus_label_val_path + str(i) + '.txt'
#   if os.path.exists(imValName):
#     os.remove(imValName)
#   if os.path.exists(labValName):
#     os.remove(labValName)

cap.release()
cv2.destroyAllWindows()
print(f'Computation Time: {time.time() - start_time}')
writer.release()
# c = 0
# for i in range(len(a)):
#   if i%3==0 and i:
#     c -= 1
#   print(a[i]+c)
#   c += 1
  