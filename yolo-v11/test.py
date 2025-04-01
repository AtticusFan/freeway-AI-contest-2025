import torch
import torch.nn as nn
from ultralytics import YOLO
from roboflow import Roboflow
print(torch.__version__)
print(torch.cuda.is_available())
import cv2 as cv
# load dataset
'''
rf = Roboflow(api_key="8kJjXwHkOtOUJOu41nCo")
project = rf.workspace("yolov8-l09qq").project("taiwan-cctv-s41bf")
version = project.version(1)
dataset = version.download("yolov11")
'''
