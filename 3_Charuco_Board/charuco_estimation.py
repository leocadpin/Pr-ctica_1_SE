import numpy as np
import cv2 as cv
import glob
from cv2 import aruco

wight = 7
hight = 5

# Cargamos los parámetros de la camara obtenidos en la calibración
with np.load('Camera_parameters.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','distance','rvecs','tvecs')]


squareLength = 1.5
markerLength = 1.2
############# Definimos la función que dibujará nuestro objeto, en este caso la funcion dibuja un hombre de palo azul 😎😎😎😎###########

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    
    #Todos lo que dibujamos en este caso son puntos que corresponden a los axis que hemos definido 
    img = cv.line(img, tuple(imgpts[1]), tuple(imgpts[3]), (255), 3)
    img = cv.line(img, tuple(imgpts[2]), tuple(imgpts[5]), (255), 3)
    img = cv.line(img, tuple(imgpts[3]), tuple(imgpts[5]), (255), 3)
    img = cv.line(img, tuple(imgpts[4]), tuple(imgpts[7]), (255), 3)
    img = cv.line(img, tuple(imgpts[6]), tuple(imgpts[8]), (255), 3)
    img = cv.line(img, tuple(imgpts[6]), tuple(imgpts[9]), (255), 3)
    img = cv.line(img, tuple(imgpts[8]), tuple(imgpts[10]), (255), 3)
    img = cv.line(img, tuple(imgpts[7]), tuple(imgpts[11]), (255), 3)

    return img

# Creamos un criterio de terminacion
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Definimos la malla de puntos que colocaremos sobre el tablero
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_250 )
squareLength = 1.5
markerLength = 1.2
board = aruco.CharucoBoard_create(7, 5, squareLength, markerLength, aruco_dict)
arucoParams = aruco.DetectorParameters_create()


# En axis introducimos las coordenadas de los puntos a dibujar del stickman

axis = np.float32([ [0,1,0],
                    [0,2*markerLength,0], [0,4*markerLength,0],
                    [0,2*markerLength,2*markerLength], [0,3*markerLength,2*markerLength], [0,4*markerLength,2*markerLength],
                    [0,2*markerLength,4*markerLength], [0,3*markerLength,4*markerLength], [0,4*markerLength,4*markerLength],
                    [0,1,2*markerLength], [0,5*markerLength,2*markerLength],
                    [0,3*markerLength,6*markerLength]])


# Activamos la camara 
cam=cv.VideoCapture(-1)
out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (int(cam.get(3)),int(cam.get(4))))

# Mientras la camara este capturando imagenes, para cada frame:
# 1- Sacamos las esquinas del tablero
# 2- Si encontramos las esquinas obtenemos los subpixeles de estas 
# 3- Usamos la funcion estimatePoseCharucoBoard() para que resuelva el problema de calcular la posicion de la camara
# 4- Teniendo las matrices de transformacion entre sistemas, podemos proyectar los puntos definidos en 
# axis, y asi encontrar la correspondencia entre los puntos 3d y 2d
# 5- Usamos draw() para dibujar el stickman como hemos descrito más arriba

while True:
    hasframe,frame=cam.read()
    if hasframe==False:
        break
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)  # First, detect markers
    aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)


    im_with_charuco_board = frame.copy()

    if np.all(ids != None): # Dibujaremos si hayamos un id al menos
            
            ##Obtenemos los puntos del tablero
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
            
            #Dibujamos los corners que se hayan
            im_with_charuco_board = aruco.drawDetectedCornersCharuco(im_with_charuco_board, charucoCorners, charucoIds, (0,255,0))
            
            #A partir de los puntos del tablero obtenemos las coordenadas (parametros extrinsecos)
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, dist, rvecs, tvecs)  # posture estimation from a charuco board (retval - buena estimacion)
            
            if retval == True:

                # Obtenemos los puntos proyectados
                imgpts, jac = cv.projectPoints(axis, rvec, tvec, mtx, dist)
                # im_with_charuco_board = aruco.drawAxis(im_with_charuco_board, mtx, dist, rvec, tvec, 2)  #esto printearia los axis
                
                ## Dibujamos el stickman
                im_with_charuco_board = draw(im_with_charuco_board,charucoCorners, imgpts)

    cv.imshow('images', im_with_charuco_board)
    if cv.waitKey(1)==13:
        break
cv.destroyAllWindows()
cam.release()
