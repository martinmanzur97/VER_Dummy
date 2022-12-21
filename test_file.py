import cv2
import openvino
from openvino.inference_engine import IECore
import imutils
import numpy as np
import time

def prueba_numpy():
    # lista = [1,2,3,4]
    #lista = [[1,2],2,[6,5,4],4,5]

    lista =[[2,1],[5,4]]
    x = [2,1]

    listanp = np.array(lista)
    shapes = np.shape(lista)
    shapesnp = np.shape(listanp)

    nueva = np.expand_dims(x, axis=0)

def fps_counter():
    fps = int(vidcap.get(cv2.CAP_PROP_FPS)) # Acces FPS property
    font = cv2.FONT_HERSHEY_SIMPLEX # Font to apply on text
    cv2.putText(img, str(fps), (50,50), font, 1, (0, 0, 255), 2) # Add text on frame


prueba_numpy()