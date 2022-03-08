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
    imgpts = np.int32(imgpts).reshape(-1,2)
    
    img = cv.line(img, tuple(imgpts[1]), tuple(imgpts[3]), (255), 3)
    img = cv.line(img, tuple(imgpts[2]), tuple(imgpts[5]), (255), 3)
    img = cv.line(img, tuple(imgpts[3]), tuple(imgpts[5]), (255), 3)
    img = cv.line(img, tuple(imgpts[4]), tuple(imgpts[7]), (255), 3)
    img = cv.line(img, tuple(imgpts[6]), tuple(imgpts[8]), (255), 3)
    img = cv.line(img, tuple(imgpts[6]), tuple(imgpts[9]), (255), 3)
    img = cv.line(img, tuple(imgpts[8]), tuple(imgpts[10]), (255), 3)
    img = cv.line(img, tuple(imgpts[7]), tuple(imgpts[11]), (255), 3)

    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((hight*wight, 3), np.float32)
objp[:,:2] = np.mgrid[0:wight, 0:hight].T.reshape(-1,2)

axis = np.float32([ [0,0,0],
                    [0,1,0], [0,3,0],
                    [0,1,-3], [0,2,-3], [0,3,-3],
                    [0,1,-6], [0,2,-6], [0,3,-6],
                    [0,0,-3], [0,4,-3],
                    [0,2,-7]])

for fname in glob.glob('patron_1_movil/*.jpg'):
    img = cv.imread(fname)

    img_c = img.copy()
    img = cv.resize(img_c, None, fx=0.25, fy=0.25)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (wight,hight), None)
    
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