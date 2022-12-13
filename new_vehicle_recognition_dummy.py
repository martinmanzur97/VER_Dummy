import cv2
import openvino
import imutils
import numpy as np

video_path = "./video/in.avi"



def main():
    #paso 1 capturar frame
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        count += 1
