import numpy as np
import cv2 as cv
import glob

wight = 9
hight = 6

# Cargamos los datos guardados durante la calibración
with np.load('ParamsCamera.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'distance', 'rvecs', 'tvecs')]

# Definimos la función que dibuja el objeto
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((hight*wight, 3), np.float32)
objp[:, :2] = np.mgrid[0:wight, 0:hight].T.reshape(-1, 2)

axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3], [0,3,-3], [3,3,-3], [3,0,-3]])

for fname in glob.glob('imagenes/*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (wight, hight), None)
    
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        
        # Find the rotation and translation vectors
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        
        img = draw(img, corners2, imgpts)
        cv.imshow('img', img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:hight] + '.png', img)

cv.destroyAllWindows()