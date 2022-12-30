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

def crop_frame(frame):
    # paso 3 recortar el frame
    # By default, keep the original frame and select complete area
    frame_height, frame_width = frame.shape[:-1]
    detection_area = [[0, 0], [frame_width, frame_height]]
    top_left_crop = (0, 0)
    bottom_right_crop = (frame_width, frame_height)
    # Select detection area
    window_name_roi = "Select Detection Area."
    roi = cv2.selectROI(window_name_roi, frame, False)
    cv2.destroyAllWindows()
    if int(roi[2]) != 0 and int(roi[3]) != 0:
        x_tl, y_tl = int(roi[0]), int(roi[1])
        x_br, y_br = int(roi[0] + roi[2]), int(roi[1] + roi[3])
        detection_area = [
            (x_tl, y_tl),
            (x_br, y_br),
        ]
    else:
        detection_area = [(0, 0),(bottom_right_crop[0] - top_left_crop[0],bottom_right_crop[1] - top_left_crop[1],),]
    return detection_area

# path = "./img/picture2.jpeg"
# img = cv2.imread(path)
# # print(img)
# det = crop_frame(img)
# # print(det)

now = time.time() # time when we finish processing for this frame
fps = 1/(now-prev_frame_time) # Calculating the fps
prev_frame_time = now