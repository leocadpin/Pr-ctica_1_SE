import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description = 'Programa para calcular esquinas con Harris')
parser.add_argument('--imagen', '-i', type=str, default='damas_corrected.jpg')
parser.add_argument('--salida', '-s', type=str, default='damasHarris.jpg')
args = parser.parse_args()

# Cargamos la imagen
img = cv.imread(args.imagen)

# Comprobamos que la imagen se ha podido leer
if img is None:
    print('Error al cargar la imagen')
    quit()

# Pasamos la imagen a escala de grises, y después a float32
img_gray = cv.imread(args.imagen, cv.IMREAD_GRAYSCALE)
img_gray = np.float32(img_gray)

# Detectar las esquinas con Harris. Parametros: blockSize=2, apertureSize=3, k=0.04.
dst = cv.cornerHarris(img_gray, 2, 3, 0.04)

# Sobre la imagen original, poner en color azul los píxeles detectados como borde.
# Son aquellos que en los que dst(i,j) tiene un valor mayor de 10000.

src = img
rows = src.shape[0]
cols = src.shape[1]
for i in range(rows): 
    for j in range(cols):
        if (dst[i,j] > 10000):
            src[i,j] = (255, 0, 0)


# Mostrar por pantalla la imagen src y además guardarla en el fichero que se pasa como segundo argumento al programa
cv.imwrite(args.salida,src)
cv.imshow('harris', src)
cv.waitKey(0)