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
    #paso 2 redimensionar el frame
    B, C, H, W = neural_net.input_info
    return frame




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

caculate_fps()