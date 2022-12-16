import cv2
import openvino
import imutils
import numpy as np
import time

#video_path = "./video/in.mp4"
video_path = "../BlindspotFront.mp4"
model_xml = "./model/person-detection-0303.xml"
model_bin = "./model/person-detection-0303.bin"


def crop_frame(x):
    #paso 3 recortar frame
    return x 

def define_area(x):
    return x

def caculate_fps():
    fps_start = 0
    fps_counter = 0

    capture = cv2.VideoCapture(video_path)

    while True:
        rec , frame = capture.read()

        fps_end = time.time()
        time_diff = fps_end - fps_start
        fps = 1/time_diff
        fps_start = fps_end

        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame,fps_text,(5,30),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 1)

        cv2.imshow("Vehicle Event Recognition", frame)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            break

    capture.release()


def vehicle_event_recognition(frame,neural_net,execution_net,input,output,detection_area):
    #B - batch size
    #C - number of channels
    #H - image height
    #W - image width
    #obtiene parametros del modelo
    B, C, H, W = neural_net.input_info[input].tensor_desc.dims 
    #paso 2 redimensionar el frame
    #redimensiona el frame de acuerdo a los parametros del modelo
    resized_frame = cv2.resize(frame, (W, H))
    #setea altura y ancho en base a las dimensiones de la matriz frame
    initial_h, initial_w, _ = frame.shape


def drawText(frame, scale, rectX, rectY, rectColor, text):
    #funcion para escribir texto en imagen
    rectThickness = 2
    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    top = max(rectY - rectThickness, textSize[0])
    cv2.putText(frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3)


def main():
    #paso 1 capturar frame
    vidcap = cv2.VideoCapture(video_path)
    #success, img = vidcap.read()
    #while success:
    while (vidcap.isOpened()):
        success, img = vidcap.read()
        if success == True:
            cv2.imshow('video', img)
            if cv2.waitKey(30) == 27:   
                break
        else:
            break
    vidcap.release()
    cv2.destroyAllWindows()


def prueba_numpy():
    lista = [1,2,3,4]
    listanp = np.array(lista)
    print(lista)
    print(listanp)

prueba_numpy()