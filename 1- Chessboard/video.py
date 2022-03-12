import numpy as np
import cv2 as cv
import glob

wight = 4
hight = 11

# Load previously saved data
with np.load('ParamsCamera_patroncirc.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','distance','rvecs','tvecs')]

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((hight*wight,3), np.float32)
objp[:,:2] = np.mgrid[0:wight,0:hight].T.reshape(-1,2)

# axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
#                    [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])



axis = np.float32([[0,0,0], [0,3*72,0], [3*72,3*72,0], [3*72,0,0],
                   [0,0,-3*72],[0,3*72,-3*72],[3*72,3*72,-3*72],[3*72,0,-3*72] ]).reshape(-1,3)
cam=cv.VideoCapture(0)
out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (int(cam.get(3)),int(cam.get(4))))

while True:
    hasframe,frame=cam.read()
    if hasframe==False:
        break
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret,corners = cv.findCirclesGrid(gray,(4,11),None)
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