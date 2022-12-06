# imports 
import torch
from PIL import Image


def ml2_api(path):
    #instatiate model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    #detect person / set confidence interval
    model.classes = [0]
    model.conf = 0.4
    #open image
    im1 = Image.open(path)
    #results
    results = model(im1, size=640)
    count = len(results.xyxy[0]) 
    return count