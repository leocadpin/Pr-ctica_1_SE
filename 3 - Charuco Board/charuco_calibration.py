import cv2
import numpy as np
import pathlib
from cv2 import aruco as aruco
import glob

#Añadimos la ruta a la carpeta de imagenes del tablero para calibrar
images = glob.glob('pattern_p/*.jpg')

# Parameters


# Dimensions in cm de los marcadores y cuadrados de nuestro tablero Charuco
marker_length = 1.2
square_length = 1.5


#ChArUco-----------------------------------------------------------------------------------------
#def calibrate_charuco(dirpath, image_format, marker_length, square_length):

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) #250, 500, 1000
board = aruco.CharucoBoard_create(7, 5, square_length, marker_length, aruco_dict)
arucoParams = aruco.DetectorParameters_create()


counter, corners_list, id_list = [], [], []

first = 0
# Find the ArUco markers inside each image
for img in images:
    print(f'using image {img}')

    image = cv2.imread(str(img))
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    corners, ids, rejected = aruco.detectMarkers(
        img_gray,
        aruco_dict,
        parameters=arucoParams
    )

    resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=img_gray,
        board=board
    )

    # Resaltar los bordes de los marcadores aruco encontrados en la imagen
    img = aruco.drawDetectedMarkers(
            image=image, 
            corners=corners)

    # Si se encuentra el charuco, tomamos puntos
    # Requerimos al menos 20 cuadrados
    if resp > 20:
        # Add these corners and ids to our calibration arrays
        corners_list.append(charuco_corners)
        id_list.append(charuco_ids)
    
    cv2.imshow("img", img) # display
    cv2.waitKey(0)

# Finalmente usamos la calibraciond de la libreria aruco y obtenemos los parametros de  nuestra camara
ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
    charucoCorners=corners_list,
    charucoIds=id_list,
    board=board,
    imageSize=img_gray.shape,
    cameraMatrix=None,
    distCoeffs=None)
# 

np.savez('Camera_parameters', mtx= mtx, distance= dist, rvecs=rvecs, tvecs= tvecs)
original = cv2.imread('pattern_p/1.jpg')
dst = cv2.undistort(original, mtx, dist, None, mtx)
cv2.imwrite('undist_charuco.jpg', dst)


# Error de proyección posterior, para todos los puntos del tablero
# 1- sacamos la proyeccion de los puntos 3d de la imagen usando los parametros
# que hemos obtenido en la calibracion
# 2-calculamos la norma entre las proyecciones y los puntos bidimensionales 
# 3-dividimos entre el numero de puntos para obtener un resultado normalizado
# 4-vamos acumulando este error y finalmente sacamos la media aritmética
# 5- Cuanto mas pequeño sea el valor, mas exactos deberian ser nuestros parametros
# total_error = 0
# for i in range(len(objpoints)):
# 	imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
# 	error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
# 	total_error += error
# alt, anch = img.shape[:2]
# #diagonal = raiz de alt² y anch²
# print(("total error: "), total_error/len(objpoints))