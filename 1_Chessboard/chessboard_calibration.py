import cv2
from cv2 import sqrt
import numpy as np
import glob
from dis import dis

# Encuentra_esquinas de tablero de ajedrez
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
imgpoints = [] # Puntos 2D bidimensionales en el plano de la imagen

<<<<<<< HEAD

#Añadimos la ruta a la carpeta de imagenes del tablero para calibrar
images = glob.glob('pattern_m/*.jpg')

=======
# Añadimos la ruta a la carpeta de imagenes del tablero para calibrar
images = glob.glob('pattern_p/*.jpg')
>>>>>>> c490231418cd73dd0a31935e1ca45c0286ae6574

for fname in images: #Repetimos el siguiente proceso para cada imagen
	img = cv2.imread(fname)

	# img_c = img.copy()
	# img = cv2.resize(img_c, None, fx=0.75, fy=0.75)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Encuentra la esquina del tablero de ajedrez. Funcion de OpenCV
	ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
	
	# Si encuentra suficientes puntos, guárdelos
	if ret == True:
		objpoints.append(objp)

		# Buscamos la localizacion mas adecuada para los subpixeles
		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners)
		
		# Mostrar los puntos de esquina en la imagen
		cv2.drawChessboardCorners(img, (w,h), corners2, ret)

		# img_res= img.copy()
		# img_res= cv2.resize(img_res, None, fx=0.25, fy=.25)

		cv2.imshow('findCorners', img)
		cv2.waitKey(0)

cv2.destroyAllWindows()

# Calibración
# Usamos la funcion calibrate camera, que obtendrá los parámetros intrinsecos y extrinsecos de la
# camara. (Esta funcion aplicará el algoritmo de Levenberg Marquardt)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print (("ret:"),ret)
# print (("mtx: \ n"), mtx) # matriz de parámetros internos
# print (("dist: \ n"), dist) # coeficiente de distorsión distorsión cofficients = (k_1, k_2, p_1, p_2, k_3)
# print (("rvecs: \ n"), rvecs) # vector de rotación # parámetros externos
# print (("tvecs: \ n"), tvecs) # vector de traducción # parámetros externos

# Des-distorsión: Sabiendo los parametros de nuestra camara podemos corregir el error en distorsion

img = cv2.imread('pattern_p/1.jpg') #Tomamos una imagen de las que usamos en la calibracion

# img_c = img.copy()
# img = cv2.resize(img_c, None, fx=0.75, fy=0.75)

h,w = img.shape[:2] #obtenemos dimensiones de la imagen

# Obtenemos la nueva matriz de parametros intrinsecos de la camara 
newcameramtx, roi = cv2.getOptimalNewCameraMatrix (mtx, dist, (w,h), 1, (w,h)) # Parámetro de escala libre

# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# # Recorte la imagen de acuerdo con el área de ROI anterior
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.jpg',dst)

# Undistort usando un remapeado 

# Usamos la siguiente funcion para conseguir la rectificacion de la imagen dandonos como
# resultado mapas para remapear la imagen original
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# Recortamos la imagen
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibResult.jpg', dst)
np.savez('Camera_parameters', mtx= mtx, dist= dist, rvecs= rvecs, tvecs= tvecs)

# Error de proyección posterior, para todos los puntos del tablero
# 1- sacamos la proyeccion de los puntos 3d de la imagen usando los parametros
# que hemos obtenido en la calibracion
# 2-calculamos la norma entre las proyecciones y los puntos bidimensionales 
# 3-dividimos entre el numero de puntos para obtener un resultado normalizado
# 4-vamos acumulando este error y finalmente sacamos la media aritmética
# 5- Cuanto mas pequeño sea el valor, mas exactos deberian ser nuestros parametros
total_error = 0
for i in range(len(objpoints)):
	imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	total_error += error

alt, anch = img.shape[:2]
<<<<<<< HEAD

alt2 = h*h
anch2 = w*w
alo = alt2+anch2
diagonal = np.sqrt(alo)
print(("total error: "), total_error/len(objpoints)/diagonal)
=======
# diagonal = raiz de alt² y anch²
print(("total error: "), total_error/len(objpoints))
>>>>>>> c490231418cd73dd0a31935e1ca45c0286ae6574
