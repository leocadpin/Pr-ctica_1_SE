
import numpy as np
import cv2 
import glob


with np.load('Camera_parameters.npz') as file:
    mtx, dist, rvecs, tvecs = [file[i] for i in ('mtx','dist','rvecs','tvecs')]

objp = np.zeros((4*11,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:11].T.reshape(-1,2)


axis = np.float32([[2,0,0], [2,-3,0]])
                  

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    for i,j in zip(range(1),range(1,3)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255,255,0),3)
   
    return img




cam=cv2.VideoCapture(0)
out = cv2.VideoWriter('output3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(cam.get(3)),int(cam.get(4))))
while True:
    hasframe,frame=cam.read()
    if hasframe==False:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(gray, (4,11), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID) 
    if ret == True:
        _,rvec,tvec=cv2.solvePnP(objp,corners,mtx,dist)
        imgpts,_=cv2.projectPoints(axis,rvec,tvec,mtx,dist)
        frame = draw_cube(frame,corners,imgpts)

    cv2.imshow('images',frame)
    
    out.write(frame)
    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows()
cam.release()
