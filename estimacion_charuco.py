import cv2
import numpy as np
import pathlib
from cv2 import aruco as aruco
import glob

# Cargamos los parámetros de la camara obtenidos en la calibración
with np.load('ParamsCamera.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'distance', 'rvecs', 'tvecs')]

aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_1000 )

squareLength = 1.5   # Here, our measurement unit is centimetre.
markerLength = 1.2   # Here, our measurement unit is centimetre.
board = aruco.CharucoBoard_create(5, 7, squareLength, markerLength, aruco_dict)

arucoParams = aruco.DetectorParameters_create()


# Activamos la camara 
cam=cv2.VideoCapture(0)
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(cam.get(3)),int(cam.get(4))))
while(True):
    ret, frame = cam.read() # Capture frame-by-frame
    if ret == True:
        # frame_remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)    # for fisheye remapping
        # frame_remapped_gray = cv2.cvtColor(frame_remapped, cv2.COLOR_BGR2GRAY)
        frame_remapped = cv2.undistort(frame, mtx, dist, None, mtx)
        frame_remapped_gray = cv2.cvtColor(frame_remapped, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)  # First, detect markers
        aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejectedImgPoints)

        if ids != None: # if there is at least one marker detected
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, frame_remapped_gray, board)
            im_with_charuco_board = aruco.drawDetectedCornersCharuco(frame_remapped, charucoCorners, charucoIds, (0,255,0))
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, dist)  # posture estimation from a charuco board
            if retval == True:
                im_with_charuco_board = aruco.drawAxis(im_with_charuco_board, mtx, dist, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
        else:
            im_with_charuco_left = frame_remapped

        cv2.imshow("charucoboard", im_with_charuco_board)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    else:
        break


cam.release()   # When everything done, release the capture
cv2.destroyAllWindows()