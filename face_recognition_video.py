# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (480, 320)
camera.framerate = 25
rawCapture = PiRGBArray(camera, size=(480, 320))

# allow the camera to warmup
time.sleep(0.1);

# Get facial recognition parameters
faceCascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Scan grayscale image for faces
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

	# Draw rectangle around faces
	for (x,y,w,h) in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
	
	# show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
	
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	
	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
