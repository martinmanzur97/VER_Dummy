import cv2
import openvino
from openvino.inference_engine import IECore
import imutils
import numpy as np
import time
import datetime

#video de origen para detecciones
video_path = "./video/in.mp4"
#video_path = "../BlindspotFront.mp4"

#modelos de openvino
model_xml = "./model/person-detection-0303.xml"
model_bin = "./model/person-detection-0303.bin"

#dispositivo
device = "CPU"

#colores para el cuadro de la deteccion
BLUE = (255, 0, 0)
RED = (0, 0, 255)

#parametro para filtrar detecciones en base a la confianza
confidence = 0.6
MODEL_HEIGHT = 720
MODEL_WIDTH = 1280

#setear los frame time en 0 antes de empezar a contar
new_frame_time = 0 
prev_frame_time = 0


def crop_frame(frame):
    #consigue los datos de la imagen, alto y ancho
    frame_height, frame_width = frame.shape[:-1]
    #determina un area de recorte entre el principio y fin de los pixeles
    detection_area = [[0, 0], [frame_width, frame_height]]
    top_left_crop = (0, 0)
    bottom_right_crop = (frame_width, frame_height)
    #setea el nombre de la ventana e invoca a selectroi que selecciona una porcion del frame
    window_name_roi = "Select Detection Area."
    roi = cv2.selectROI(window_name_roi, frame, False)
    cv2.destroyAllWindows()
    #organiza los resultados de selectROI en una lista de 2 tuplas para sea procesada luego por check_detection area
    #si no se selecciono nada por defecto toma todo el frame y por ultimo retorna los valores
    if int(roi[2]) != 0 and int(roi[3]) != 0:
        x_tl, y_tl = int(roi[0]), int(roi[1])
        x_br, y_br = int(roi[0] + roi[2]), int(roi[1] + roi[3])
        detection_area = [(x_tl, y_tl),(x_br, y_br)]
    else:
        detection_area = [(0, 0),(bottom_right_crop[0] - top_left_crop[0],bottom_right_crop[1] - top_left_crop[1],)]
    return detection_area


def check_detection_area(x, y, detection_area):
    #verifica que el area de deteccion tenga tamaño 2 por los dos elementos establecidos en crop_frame
    if len(detection_area) != 2:
        raise ValueError("Invalid number of points in detection area")
    #establece limites del area en base a lo recibido al recorte resultado de crop_frame, donde top left son 0 en el eje, y bottom right son los maximos en pixeles
    top_left = detection_area[0]
    bottom_right = detection_area[1]
    xmin, ymin = top_left[0], top_left[1]
    xmax, ymax = bottom_right[0], bottom_right[1]
    # Retorna True si los parametros que pasan estan dentro del area de deteccion, False si no
    return xmin < x and x < xmax and ymin < y and y < ymax


def fps_counter(frame):
    global new_frame_time, prev_frame_time
    font = cv2.FONT_HERSHEY_COMPLEX
    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (7, 70), font, 2, BLUE, 3)


def vehicle_event_recognition(frame, neural_net, execution_net, ver_input, ver_output, detection_area):
    #obtiene parametros del modelo, B - batch size, C - number of channels, H - image height, W - image width
    B, C, H, W = neural_net.input_info[ver_input].tensor_desc.dims 
    #paso 2 redimensionar el frame - redimensiona el frame de acuerdo a los parametros del modelo
    resized_frame = cv2.resize(frame, (W, H))
    #setea altura y ancho en base a las dimensiones de la matriz frame
    initial_h, initial_w, _ = frame.shape
    #formatea la matriz para que tenga la forma especificada en el modelo dejando el elemento 3 primero, 1 segundo, y 2 tercero y luego agregando uno nuevo en posicion 1
    resized_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)
    
    #ver_results = execution_net.infer(inputs={ver_input: resized_image}).keys()
    #
    #
    #
    #COMENTAR
    #
    #
    #
    ver_results = execution_net.infer(inputs={ver_input: resized_image}).get(ver_output)
    for detection in ver_results:
        ver_confidence = detection[4]
        if ver_confidence < confidence:
            break 
        #de las detecciones
        xmin = int(detection[0] * initial_w / MODEL_WIDTH)
        ymin = int(detection[1] * initial_h / MODEL_HEIGHT)
        xmax = int(detection[2] * initial_w / MODEL_WIDTH)
        ymax = int(detection[3] * initial_h / MODEL_HEIGHT)
        xmin = max(0, xmin - 5)
        xmax = min(xmax + 5, initial_w - 1)
        ymax = min(ymax + 5, initial_h - 1)
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2

        # marca las detecciones con un rectangulo si cumplen con estar dentro del area de deteccion
        if check_detection_area(x, y, detection_area):
            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),RED,thickness=2)

def main():

    #se instancia un objeto IEcore para trabajar con openvino
    ie = IECore()

    ver_neural_net = ie.read_network(model=model_xml, weights=model_bin)
    ver_execution_net = ie.load_network(network=ver_neural_net, device_name=device.upper())
    ver_input_blob = next(iter(ver_execution_net.input_info))
    ver_output_blob = next(iter(ver_execution_net.outputs))
    ver_neural_net.batch_size = 1

    #paso 1 capturar frame utilizando un video como archivo de origen el video_path
    vidcap = cv2.VideoCapture(video_path)
    #devuelve tupla con booleano y los datos del frame en forma de matriz
    success, img = vidcap.read()
    #recorta el frame estableciendo el area de deteccion
    detection = crop_frame(img)
    #mientras haya obtenido el frame de manera correcta

    while success:
        success, img = vidcap.read()
        vehicle_event_recognition(img,ver_neural_net,ver_execution_net,ver_input_blob,ver_output_blob, detection)
        if cv2.waitKey(10) == 27:  
            break
        fps_counter(img)



        showImg = imutils.resize(img, height=500)
        cv2.imshow("showImg", showImg)



main()
