import cv2
import openvino
from openvino.inference_engine import IECore
import imutils
import numpy as np
import time

video_path = "./video/in.mp4"

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

prueba_numpy()

def fps_counter():
    fps = int(vidcap.get(cv2.CAP_PROP_FPS)) # Acces FPS property
    font = cv2.FONT_HERSHEY_SIMPLEX # Font to apply on text
    cv2.putText(img, str(fps), (50,50), font, 1, (0, 0, 255), 2) # Add text on frame

def prueba():
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    print(vidcap.read())

def frames2():
    prev_frame_time = 0
    new_frame_time = 0
    #FPS Counter
    ret = True
    if not ret:
        exit(0)
    gray = frame
    gray = cv2.resize(gray, (500, 300))
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(gray, fps, (7, 70), font, 2, (100, 255, 0), 3, cv2.LINE_AA)
