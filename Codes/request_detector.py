# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="hand.mp4")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")


# if the video argument is None, then we are reading from webcam
'''
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
''' 
# otherwise, we are reading from a video file
#else:
camera = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None
# loop over the frames of the video
lastframe=0
currframe=0
text = "Human Working"
lasttext=text
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	text = "Human Working"
 
	# if the frame could not be grabbed, then we have reached the end
	# of e video
	if not grabbed:
		break
 
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)

	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)
 
	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
 
	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	#skin = cv2.bitwise_and(frame, frame, mask = skinMask)

	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = skinMask
		continue
		# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, skinMask)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(ab,cnts,cd) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_NONE)
	#print cnts
	if(cnts==[]):
		if(lasttext!=text):
			print datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
			lasttext=text

	if(cnts==[]):
		lastframe=currframe
		currframe=0
	
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		'''
		try:		
			if c!=np.zero() and cv2.contourArea(c) < 10:
				continue
 		except:
			pass
		'''
		if  cv2.contourArea(c) < 30000:
			continue	
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Human Waiting"
		if(lasttext!=text):
			print datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
		lasttext=text
		lastframe=currframe
		currframe=1
		# draw the text and timestamp on the frame
	cv2.putText(frame, "Request Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)

	key = cv2.waitKey(1) & 0xFF
 	
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
 	
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
