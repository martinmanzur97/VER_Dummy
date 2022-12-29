import cv2
import openvino
from openvino.inference_engine import IECore
import imutils
import numpy as np
import time
import datetime

video_path = "./video/in.mp4"
#video_path = "../BlindspotFront.mp4"
model_xml = "./model/person-detection-0303.xml"
model_bin = "./model/person-detection-0303.bin"
# model_xml = "./model/pedestrian-and-vehicle-detector-adas-0001.xml"
# model_bin = "./model/pedestrian-and-vehicle-detector-adas-0001.bin"
device = "CPU"
BLUE = (255, 0, 0)
RED = (0, 0, 255)
confidence = 0.6

def crop_frame(frame):
    #consigue los datos de la imagen, alto y ancho
    frame_height, frame_width = frame.shape[:-1]
    #determina un area de recorte entre el principio y fin de los pixeles
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

def check_detection_area(x, y, detection_area):
    #verifica que el area de deteccion tenga tamaño 2 si no tira error
    if len(detection_area) != 2:
        raise ValueError("Invalid number of points in detection area")
    #establece limites del area
    top_left = detection_area[0]
    bottom_right = detection_area[1]
    #en base a los 
    xmin, ymin = top_left[0], top_left[1]
    xmax, ymax = bottom_right[0], bottom_right[1]
    # Check if the point is inside a ROI
    return xmin < x and x < xmax and ymin < y and y < ymax


def caculate_fps(path):
    fps_start = 0
    fps_counter = 0

    capture = cv2.VideoCapture(path)

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


def vehicle_event_recognition(frame, neural_net, execution_net, ver_input, ver_output, detection_area):
    #obtiene parametros del modelo, B - batch size, C - number of channels, H - image height, W - image width
    B, C, H, W = neural_net.input_info[ver_input].tensor_desc.dims 
    #paso 2 redimensionar el frame - redimensiona el frame de acuerdo a los parametros del modelo
    resized_frame = cv2.resize(frame, (W, H))
    #setea altura y ancho en base a las dimensiones de la matriz frame
    initial_h, initial_w, _ = frame.shape
    #formatea la matriz para que tenga la forma especificada en el modelo 
    #dejando el elemento 3 primero, 1 segundo, y 2 tercero y 
    #luego agregando uno nuevo en posicion 1
    resized_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)
    
    #ver_results = execution_net.infer(inputs={ver_input: resized_image}).keys()
    ver_results = execution_net.infer(inputs={ver_input: resized_image}).get(ver_output)
    for detection in ver_results:
        ver_confidence = detection[4]
        # if ver_confidence < confidence:
        #     break 

        xmin = int(detection[0] * initial_w)
        ymin = int(detection[1] * initial_h)
        xmax = int(detection[2] * initial_w)
        ymax = int(detection[3] * initial_h)
        # xmin = max(0, xmin - 5)
        # xmax = min(xmax + 5, initial_w - 1)
        # ymax = min(ymax + 5, initial_h - 1)
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2

        # Check if central points fall inside the detection area
        if check_detection_area(x, y, detection_area):
            cv2.rectangle(
                frame,
                (xmin, ymin),
                (xmax, ymax),
                BLUE,
                thickness=2,
                )

    showImg = imutils.resize(frame, height=750)
    cv2.imshow("showImg", showImg)

def old_ver_detection(ver_results):

    for detection in ver_results[0]:
        label = int(detection[1])
        accuracy = float(detection[2])
        det_color = BLUE if label == 1 else RED
        # Draw only objects when accuracy is greater than configured threshold
        if accuracy > confidence_threshold:
            xmin = int(detection[3] * initial_w)
            ymin = int(detection[4] * initial_h)
            xmax = int(detection[5] * initial_w)
            ymax = int(detection[6] * initial_h)
            # Central points of detection
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2

            # Check if central points fall inside the detection area
            if check_detection_area(x, y, detection_area):
                cv2.rectangle(
                    frame,
                    (xmin, ymin),
                    (xmax, ymax),
                    det_color,
                    thickness=2,
                )

    showImg = imutils.resize(frame, height=600)
    cv2.imshow("showImg", showImg)


def drawText(frame, scale, rectX, rectY, rectColor, text):
    #funcion para escribir texto en imagen
    rectThickness = 2
    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    top = max(rectY - rectThickness, textSize[0])
    cv2.putText(frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3)


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
    print(detection)
    #mientras haya obtenido el frame de manera correcta
    while success:
        vehicle_event_recognition(img,ver_neural_net,ver_execution_net,ver_input_blob,ver_output_blob, detection)
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break
        success, img = vidcap.read()


#paso 3 recortar imagen con opencv
def crop(img_path):
    import cv2
    img = cv2.imread(img_path)
    crop_img = img[y:y+h, x:x+w]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)


main()
