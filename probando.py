import cv2
import numpy as np
import os

with np.load('ParamsCamera.npz') as file:
    mtx,dist=[file[i] for i in ['mtx','distance']]

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)    #axis point for draw axis

axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),3)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img
    
    
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img                       

path='imagenes/'
root=os.getcwd()

l=os.listdir(path)
for i in l:
    path1=os.path.join(root,path,i)
    im=cv2.imread(path1)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(9,6),None)
    if ret == True:
        _,rvec,tvec,_=cv2.solvePnPRansac(objp,corners,mtx,dist)
        imgpts,_=cv2.projectPoints(axis,rvec,tvec,mtx,dist)
        img = draw(im,corners,imgpts)
        cv2.imshow('images',img)
        cv2.waitKey()
        cv2.destroyAllWindows()