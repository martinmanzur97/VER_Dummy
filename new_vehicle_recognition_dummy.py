import cv2
import openvino
import imutils
import numpy as np

video_path = "./video/in.avi"

def main():
    vidcap = cv2.VideoCapture(video_path)
    
