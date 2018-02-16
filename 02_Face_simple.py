import cv2
import numpy as np

#faceCascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('opencv/samples/data/lena.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('sukima.jpg', cv2.IMREAD_COLOR)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

face = faceCascade.detectMultiScale(gray, 1.1, 3)

if len(face) > 0:
        for rect in face:
                cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=2)
else:
        print( "no face" )

cv2.imwrite('detected.jpg', img)

#Add Comment

