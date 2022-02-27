from dis import dis
import cv2
import numpy as np
import glob

# Encuentra esquinas de tablero de ajedrez
# Límite
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Especificaciones de la plantilla del tablero
w = 9
h = 6

# Puntos de tablero de ajedrez en el sistema de coordenadas mundial, como (0,0,0), (1,0,0), (2,0,0) ...., (8,5,0), elimine la coordenada Z, grabado como una matriz bidimensional
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)

# Guarde las coordenadas de mundo y el par de coordenadas de imagen de los puntos de esquina del tablero
objpoints = [] # puntos 3D en el sistema mundial de coordenadas
imgpoints = [] # Puntos bidimensionales en el plano de la imagen

images = glob.glob('imagenes/*.jpg')

for fname in images:
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	# Encuentra la esquina del tablero de ajedrez
	ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
	
	# Si encuentra suficientes puntos, guárdelos
	if ret == True:
		cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		objpoints.append(objp)
		imgpoints.append(corners)
		
		# Mostrar los puntos de esquina en la imagen
		cv2.drawChessboardCorners(img, (w,h), corners, ret)

		img_res= img.copy()
		img_res= cv2.resize(img_res, None, fx=0.25, fy=.25)

		# cv2.imshow('findCorners',img_res)
		# cv2.waitKey(1)

cv2.destroyAllWindows()


# Calibración
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print (("ret:"),ret)
# print (("mtx: \ n"), mtx) # matriz de parámetros internos
# print (("dist: \ n"), dist) # coeficiente de distorsión distorsión cofficients = (k_1, k_2, p_1, p_2, k_3)
# print (("rvecs: \ n"), rvecs) # vector de rotación # parámetros externos
# print (("tvecs: \ n"), tvecs) # vector de traducción # parámetros externos

# Des-distorsión
img2 = cv2.imread('imagenes/1.jpg')
h,w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix (mtx, dist, (w, h), 0, (w, h)) # Parámetro de escala libre
# dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

# # Recorte la imagen de acuerdo con el área de ROI anterior
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.jpg',dst)



# Undistort with Remapping
mapx, mapy = cv2.initUndistortRectifyMap(newcameramtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('caliResult2.png', dst)

np.savez('ParamsCamera', mtx= mtx, distance= dist, rvecs=rvecs, tvecs= tvecs)

# Error de proyección posterior
total_error = 0
for i in range(len(objpoints)):
	imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	total_error += error
print(("total error: "), total_error/len(objpoints))
