import numpy as np
import cv2 as cv
import glob
from cv2 import aruco

wight = 7
hight = 5

# Cargamos los parámetros de la camara obtenidos en la calibración
with np.load('ParamsCamera_charuco.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','distance','rvecs','tvecs')]


squareLength = 1.5
markerLength = 1.2
############# Definimos la función que dibujará nuestro objeto, en este caso la funcion dibuja un hombre de palo azul ###########

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
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
squareLength = 1.5
markerLength = 1.2
board = aruco.CharucoBoard_create(7, 5, squareLength, markerLength, aruco_dict)
arucoParams = aruco.DetectorParameters_create()


# En axis introducimos las coordenadas de los puntos a dibujar del stickman
axis = np.float32([ [0,0,0],
                    [0,1,0], [0,3,0],
                    [0,1,-3], [0,2,-3], [0,3,-3],
                    [0,1,-6], [0,2,-6], [0,3,-6],
                    [0,0,-3], [0,4,-3],
                    [0,2,-7]])

# axis = np.float32([ [0*72,0*72,0*72],
#                     [0,1*72,0], [0,3*72,0],
#                     [0,1*72,-3*72], [0,2*72,-3*72], [0,3*72,-3*72],
#                     [0,1*72,-6*72], [0,2*72,-6*72], [0,3*72,-6*72],
#                     [0,0,-3*72], [0,4*72,-3*72],
#                     [0,2*72,-7*72]])

# Activamos la camara 
cam=cv.VideoCapture(-1)
out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (int(cam.get(3)),int(cam.get(4))))

# Mientras la camara este capturando imagenes, para cada frame:
# 1- Sacamos las esquinas del tablero
# 2- Si encontramos las esquinas obtenemos los subpixeles de estas 
# 3- Usamos la funcion pnp para que resuelva el problema de calcular la posicion de la camara
# 4- Teniendo las matrices de transformacion entre sistemas, podemos proyectar los puntos definidos en 
# axis, y asi encontrar la correspondencia entre los puntos 3d y 2d
# 5- Usamos draw() para dibujar el cubo como hemos descrito más arriba
while True:
    hasframe,frame=cam.read()
    if hasframe==False:
        break
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)  # First, detect markers
    aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)

    if np.any(ids) == False: # if there is at least one marker detected
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
            im_with_charuco_board = aruco.drawDetectedCornersCharuco(gray, charucoCorners, charucoIds, (0,255,0))
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, dist)  # posture estimation from a charuco board
            if retval == True:
                im_with_charuco_board = aruco.drawAxis(im_with_charuco_board, mtx, dist, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
    else:
        im_with_charuco_board = gray
    cv.imshow('images',frame)
    if cv.waitKey(1)==13:
        break
cv.destroyAllWindows()
cam.release()
