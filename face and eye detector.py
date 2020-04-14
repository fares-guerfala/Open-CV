import cv2
import numpy as np

#load the required XML classifiers then load our input image (or video) in grayscale mode.
face_cascade = cv2.CascadeClassifier('C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')
img=cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#find the faces and eyes in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()