import cv2
import numpy as np

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('oficina.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#image es la deteccion de rostros
#scaleFactor sirve para reducri la imagen - mientras mas altos se pierden rostros
#minNeigbord vecinos candidatos a los rostros limitados detectados
#minSize tamaño minimo posible del objeto
#maxSIze maximo posible del objeto
faces = faceClassif.detectMultiScale(gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30,30),
	maxSize=(200,200))

#rectangulos detectados
for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()