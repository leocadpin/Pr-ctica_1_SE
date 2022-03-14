import numpy as np
import cv2 as cv
import glob

wight = 9
hight = 6

# Cargamos los parámetros de la camara obtenidos en la calibración
with np.load('Camera_parameters.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

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

##################################################################################################################################

# Creamos un criterio de terminacion, o bien terminamos de buscar el pixel suboptimo a las x iteraciones
# o bien terminamos cuando encontremos un subpixel de precision x
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# Definimos la malla de puntos que colocaremos sobre el tablero
objp = np.zeros((hight*wight,3), np.float32)
objp[:,:2] = np.mgrid[0:wight,0:hight].T.reshape(-1,2)

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
cam=cv.VideoCapture(0)
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
    ret,corners = cv.findChessboardCorners(gray,(9,6),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # _,rvec,tvec,_=cv.solvePnPRansac(objp,corners2,mtx,dist)
        _,rvec,tvec =cv.solvePnP(objp,corners2,mtx,dist)
        imgpts,_=cv.projectPoints(axis,rvec,tvec,mtx,dist)
        frame = draw(frame,corners,imgpts)
    cv.imshow('images',frame)
    out.write(frame)
    if cv.waitKey(1)==13:
        break
cv.destroyAllWindows()
cam.release()