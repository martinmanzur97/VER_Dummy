import cv2
import openvino
from openvino.inference_engine import IECore
import imutils
import numpy as np
import time
from datetime import datetime

video_path = "./video/in.mp4"
frame_path = "./img/test.jpg"

initial_dt = datetime.now()
initial_ts = int(datetime.timestamp(initial_dt))    
fps = 0

def prueba_numpy():
    # lista = [1,2,3,4]
    #lista = [[1,2],2,[6,5,4],4,5]

    lista =[[2,1],[5,4],[1,2,3,4,5]]
    x = [2,1]

    listanp = np.array(lista,dtype=object)
    shape1 = np.asarray(lista).shape
    shapes = np.shape(lista)
    shapesnp = np.shape(listanp)

    nueva = np.expand_dims(x, axis=0)
    # print(listanp)
    print(shape1)

def prueba():
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    print(vidcap.read())

def crop_image(frame):
    window_name_roi = "Select Detection Area."
    roi = cv2.selectROI(window_name_roi, frame, False)
    #img = cv2.imread("lenna.png")
    crop_img = img[y:y+h, x:x+w]
