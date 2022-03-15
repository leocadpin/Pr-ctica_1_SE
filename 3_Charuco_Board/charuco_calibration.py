import cv2
import numpy as np
import pathlib
from cv2 import aruco as aruco
import glob

#Añadimos la ruta a la carpeta de imagenes del tablero para calibrar
images = glob.glob('patron_p/*.jpg')

# Dimensions in cm de los marcadores y cuadrados de nuestro tablero Charuco
marker_length = 1.2
square_length = 1.5

#ChArUco-----------------------------------------------------------------------------------------
#def calibrate_charuco(dirpath, image_format, marker_length, square_length):

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) #250, 500, 1000
board = aruco.CharucoBoard_create(7, 5, square_length, marker_length, aruco_dict)
arucoParams = aruco.DetectorParameters_create()

objp = np.zeros((6*4, 3), np.float32)
objp[0]  = (1*square_length, 1*square_length, 0)
objp[1]  = (1*square_length, 2*square_length, 0)
objp[2]  = (1*square_length, 3*square_length, 0)
objp[3]  = (1*square_length, 4*square_length, 0)
objp[4]  = (2*square_length, 1*square_length, 0)
objp[5]  = (2*square_length, 2*square_length, 0)
objp[6]  = (2*square_length, 3*square_length, 0)
objp[7]  = (2*square_length, 4*square_length, 0)
objp[8]  = (3*square_length, 1*square_length, 0)
objp[9]  = (3*square_length, 2*square_length, 0)
objp[10] = (3*square_length, 3*square_length, 0)
objp[11] = (3*square_length, 4*square_length, 0)
objp[12] = (4*square_length, 1*square_length, 0)
objp[13] = (4*square_length, 2*square_length, 0)
objp[14] = (4*square_length, 3*square_length, 0)
objp[15] = (4*square_length, 4*square_length, 0)
objp[16] = (5*square_length, 1*square_length, 0)
objp[17] = (5*square_length, 2*square_length, 0)
objp[18] = (5*square_length, 3*square_length, 0)
objp[19] = (5*square_length, 4*square_length, 0)
objp[20] = (6*square_length, 1*square_length, 0)
objp[21] = (6*square_length, 2*square_length, 0)
objp[22] = (6*square_length, 3*square_length, 0)
objp[23] = (6*square_length, 4*square_length, 0)


objpoints, corners_list, id_list = [], [], []

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
        objpoints.append(objp)
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

np.savez('Camera_parameters', mtx= mtx, dist= dist, rvecs=rvecs, tvecs= tvecs)
original = cv2.imread('patron_p/1.jpg')
dst = cv2.undistort(original, mtx, dist, None, mtx)
cv2.imwrite('undist_charuco.jpg', dst)


total_error = 0
for i in range(len(objpoints)):
	imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	error = cv2.norm(corners_list[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	total_error += error

alt, anch = img.shape[:2]

alt2 = alt**2
anch2 = anch**2
alo = alt2+anch2
diagonal = np.sqrt(alo)
print(("total error: "), total_error/len(objpoints)/diagonal)
