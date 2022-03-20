import numpy as np
import cv2
import yaml
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################
#estos serán los parametros que comonen nuestro filtro blob, que es el que nos ayuda a encontrar los circulos en la imagen




# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filtro de área, habra que cambiarlo según nos interese y según el tamaño de los circulos de la imagen que queramos captar
blobParams.filterByArea = True
blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

# # Filter by Area.
# blobParams.filterByArea = True
# blobParams.minArea = 5000      # minArea may be adjusted to suit for your experiment
# blobParams.maxArea = 20000   # maxArea may be adjusted to suit for your experiment

# Filtro de circularidad
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filtro de convexidad
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87
# blobParams.minConvexity = 0.87

# Filtro de inercia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Creamos nuestro detector con los filtros que hemos establecido
blobDetector = cv2.SimpleBlobDetector_create(blobParams)
###################################################################################################


###################################################################################################
# Aqui introducimos las coordenadas de los centros de los circulos, las cuales deben ser conocidas
# La distancia entre los centros es de 72 centimetros
# 
# Aunque la distancia no es tan importante mientras estamos calculando los paramtros de la camara
objp = np.zeros((44, 3), np.float32)
objp[0]  = (0  , 0  , 0)
objp[1]  = (0  , 72 , 0)
objp[2]  = (0  , 144, 0)
objp[3]  = (0  , 216, 0)
objp[4]  = (36 , 36 , 0)
objp[5]  = (36 , 108, 0)
objp[6]  = (36 , 180, 0)
objp[7]  = (36 , 252, 0)
objp[8]  = (72 , 0  , 0)
objp[9]  = (72 , 72 , 0)
objp[10] = (72 , 144, 0)
objp[11] = (72 , 216, 0)
objp[12] = (108, 36,  0)
objp[13] = (108, 108, 0)
objp[14] = (108, 180, 0)
objp[15] = (108, 252, 0)
objp[16] = (144, 0  , 0)
objp[17] = (144, 72 , 0)
objp[18] = (144, 144, 0)
objp[19] = (144, 216, 0)
objp[20] = (180, 36 , 0)
objp[21] = (180, 108, 0)
objp[22] = (180, 180, 0)
objp[23] = (180, 252, 0)
objp[24] = (216, 0  , 0)
objp[25] = (216, 72 , 0)
objp[26] = (216, 144, 0)
objp[27] = (216, 216, 0)
objp[28] = (252, 36 , 0)
objp[29] = (252, 108, 0)
objp[30] = (252, 180, 0)
objp[31] = (252, 252, 0)
objp[32] = (288, 0  , 0)
objp[33] = (288, 72 , 0)
objp[34] = (288, 144, 0)
objp[35] = (288, 216, 0)
objp[36] = (324, 36 , 0)
objp[37] = (324, 108, 0)
objp[38] = (324, 180, 0)
objp[39] = (324, 252, 0)
objp[40] = (360, 0  , 0)
objp[41] = (360, 72 , 0)
objp[42] = (360, 144, 0)
objp[43] = (360, 216, 0)
###################################################################################################


img = glob.glob('pattern_p/*.jpg')


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



num = 10
found = 0
for image in img:
# while(found < 8):  # Here, 10 can be changed to whatever number you like to choose
    # ret, img = cap.read() # Capture frame-by-frame
    imagen = cv2.imread(image)
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    keypoints = blobDetector.detect(gray) # Detect blobs.

    # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() . 
    #Detectamos los keypoints
    im_with_keypoints = cv2.drawKeypoints(imagen, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4,11), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

    if ret == True:
        
        objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
        
        #ajustamos el pixel optimo
        corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (4,11), (-1,-1), criteria)    # Refines the corner locations.
        
        
        #añadimos los keypoints a la lista de  puntos de la imagen
        imgpoints.append(corners2)

        #Dibujamos los contornos encontrados
        im_with_keypoints = cv2.drawChessboardCorners(imagen, (4,11), corners2, ret)

        # Enable the following 2 lines if you want to save the calibration images.
        filename = str(found) +".jpg"
        cv2.imwrite(filename, im_with_keypoints)

        found += 1


    cv2.imshow("img", im_with_keypoints) # display
    cv2.waitKey(0)


# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez('Camera_parameters', mtx= mtx, dist= dist, rvecs=rvecs, tvecs= tvecs)

#Aqui calculamos el error de reproyeccion como hemo hecho en el patron del chessboard 
total_error = 0
for i in range(len(objpoints)):
	imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	total_error += error

alt, anch = im_with_keypoints.shape[:2]

alt2 = alt*alt
anch2 = anch*anch
alo = alt2+anch2
diagonal = np.sqrt(alo)
print(("total error: "), format(total_error/len(objpoints)/diagonal, '.8f'))
