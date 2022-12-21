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

prueba_numpy