import numpy as np
import cv2 as cv
import glob

wight = 9
hight = 6

# Load previously saved data
with np.load('ParamsCamera_patroncirc.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','distance','rvecs','tvecs')]

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
objp = np.zeros((hight*wight,3), np.float32)
objp[:,:2] = np.mgrid[0:wight,0:hight].T.reshape(-1,2)

# axis = np.float32([ [0,0,0],
#                     [0,1,0], [0,3,0],
#                     [0,1,-3], [0,2,-3], [0,3,-3],
#                     [0,1,-6], [0,2,-6], [0,3,-6],
#                     [0,0,-3], [0,4,-3],
#                     [0,2,-7]])

axis = np.float32([ [0*72,0*72,0*72],
                    [0,1*72,0], [0,3*72,0],
                    [0,1*72,-3*72], [0,2*72,-3*72], [0,3*72,-3*72],
                    [0,1*72,-6*72], [0,2*72,-6*72], [0,3*72,-6*72],
                    [0,0,-3*72], [0,4*72,-3*72],
                    [0,2*72,-7*72]])

cam=cv.VideoCapture(0)
out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (int(cam.get(3)),int(cam.get(4))))
while True:
    hasframe,frame=cam.read()
    if hasframe==False:
        break
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret,corners = cv.findChessboardCorners(gray,(9,6),None)
    if ret == True:
        _,rvec,tvec,_=cv.solvePnPRansac(objp,corners,mtx,dist)
        imgpts,_=cv.projectPoints(axis,rvec,tvec,mtx,dist)
        frame = draw(frame,corners,imgpts)
    cv.imshow('images',frame)
    out.write(frame)
    if cv.waitKey(1)==13:
        break
cv.destroyAllWindows()
cam.release()