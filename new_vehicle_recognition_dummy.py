import cv2
import imutils
from openvino.inference_engine import IECore
import numpy as np
from datetime import datetime
import json

path = "./constants.json"
file=open(path)
data=json.load(file)

#video de origen para detecciones
video_path = data.get("video_path1")

#modelos de openvino
model_xml = data.get("model_xml")
model_bin = data.get("model_bin")

#dispositivo
device = data.get("device")

#colores para utilizar con opencv
BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

#parametro para filtrar detecciones en base a la confianza
confidence = data.get("confidence")

#setear los frame time en 0 antes de empezar a contar
initial_dt = datetime.now()
initial_ts = int(datetime.timestamp(initial_dt))
fps = 0
old_fps = 0


def crop_frame(frame, message):
    #consigue los datos de la imagen, alto y ancho
    frame_height, frame_width = frame.shape[:-1]
    #determina un area de recorte entre el principio y fin de los pixeles como default
    detection_area = [[0, 0], [frame_width, frame_height]]
    top_left_crop = (0, 0)
    bottom_right_crop = (frame_width, frame_height)
    #setea el nombre de la ventana e invoca a selectroi que selecciona una porcion del frame
    window_name_roi = message
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

def vehicle_event_recognition(frame, neural_net, execution_net, ver_input, ver_output, detection_area):
    #obtiene parametros del modelo, B - batch size, C - number of channels, H - image height, W - image width
    B, C, H, W = neural_net.input_info[ver_input].tensor_desc.dims 
    #paso 2 redimensionar el frame - redimensiona el frame de acuerdo a los parametros del modelo
    resized_frame = cv2.resize(frame, (W, H))
    #setea altura y ancho en base a las dimensiones de la matriz frame
    initial_h, initial_w, _ = frame.shape
    #formatea la matriz para que tenga la forma especificada en el modelo dejando el elemento 3 primero, 1 segundo, y 2 tercero y luego agregando uno nuevo en posicion 1
    resized_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)
    
    ver_results = execution_net.infer(inputs={ver_input: resized_image}).get(ver_output)
    for detection in ver_results:
        ver_confidence = detection[4]
        if ver_confidence < confidence:
            break 
        xmin = int(detection[0]*initial_w/W)
        ymin = int(detection[1]*initial_h/H)
        xmax = int(detection[2]*initial_w/W)
        ymax = int(detection[3]*initial_h/H)
        xmin = max(0, xmin-5)
        xmax = min(xmax+5, initial_w-1)
        ymax = min(ymax+5, initial_h-1)
        x = (xmin+xmax)/2
        y = (ymin+ymax)/2

        # marca las detecciones con un rectangulo si cumplen con estar dentro del area de deteccion
        if check_detection_area(x, y, detection_area):
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),RED, thickness=2)

def fps_counter(frame):
    global initial_dt, initial_ts, fps, old_fps
    dt = datetime.now()
    ts = int(datetime.timestamp(dt))

    if ts > initial_ts:
        print("FPS: ", fps)
        old_fps = fps
        fps = 0
        initial_ts = ts
    else:
        fps += 1
    cv2.putText(frame, "fps:"+str(int(old_fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN, 2)

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
    cropped_frame = crop_frame(img, "Crop Image")
    frame = img
    frame = frame[cropped_frame[0][1] : cropped_frame[1][1],cropped_frame[0][0] : cropped_frame[1][0]]
    frame = cv2.resize(frame,(cropped_frame[1][0] - cropped_frame[0][0],cropped_frame[1][1] - cropped_frame[0][1]))
    detection = crop_frame(frame, "Select Detection Area")
    #mientras haya obtenido el frame de manera correcta

    while success:
        success, img = vidcap.read()
        frame = img[cropped_frame[0][1] : cropped_frame[1][1],cropped_frame[0][0] : cropped_frame[1][0]]

        vehicle_event_recognition(frame,ver_neural_net,ver_execution_net,ver_input_blob,ver_output_blob, detection)
        if cv2.waitKey(10) == 27:  
            break
        
        fps_counter(frame)

        showImg = imutils.resize(frame,400)
        cv2.imshow("VER - Dummy Demo", showImg)

main()
