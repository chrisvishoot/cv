# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy
import urllib

# Download facial-recognition algorithm parameters
#urllib.urlretrieve('https://raw.githubusercontent.com/Itseez/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml', './faceparameters.xml')

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera();
rawCapture = PiRGBArray(camera);

# allow the camera to warmup
time.sleep(0.1);

# grab an image from the camera
camera.capture(rawCapture, format="bgr");
image = rawCapture.array;

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get facial recognition parameters
faceCascade = cv2.CascadeClassifier('faceparameters.xml')

# Scan grayscale image for faces
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

# Draw rectangle around faces
for (x,y,w,h) in faces:
	cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)

# Save image
cv2.imwrite('./in_facefound.png', image)

# display the image on screen and wait for a keypress
cv2.imshow("Image", image);
cv2.waitKey(0);
