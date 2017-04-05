# import the necessary packages
import argparse
import datetime
from datetime import timedelta
import imutils
import time
import cv2
import numpy as np

#FOR ARIMA IMPORTS
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.arima_model import ARIMA

#For Plotting
import matplotlib.pyplot as plt


def get_sec(time_str):
	h, m, s = time_str.split(':')
	return int(h) * 3600 + int(m) * 60 + int(s)


plt.ion()
#plt.axis([0,20],[])
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

#ARIMAAAA
array = ["10:10","10:20","10:30","10:40","10:50","11:00","11:10","11:20","11:30","11:40","11:50","12:30","12:40","12:50","13:00","13:10","13:20","13:30","13:40","13:50","14:30","14:40","14:50","15:00","15:10","15:20","15:30","15:40","15:50","16:20","16:30","16:40","16:50","17:00","17:10","17:20","17:30","17:40"]
timeparse = lambda dates: pd.datetime.strptime(dates, '%H:%M')
data = pd.read_csv('data.csv', parse_dates=[0],date_parser=timeparse)
pre_time = datetime.datetime.now()
i = -1
curr_time = datetime.datetime.now()
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
			i = i+1
			time1 = pd.datetime.strptime(array[i],'%H:%M')
			time = datetime.datetime.now()		
			tdiff = (time-pre_time).total_seconds()
			df2 = pd.DataFrame([[time1,tdiff]], columns=['Local_Time','#Activity_Time'])
			data2 =  data.append(df2,ignore_index=True)
        		data2 = data2.set_index(['Local_Time'])
			pre_time = time
			lasttext=text
			if(lasttext=='Human Working'):
				ts = data2['#Activity_Time'] 
				ts_log = np.log(ts)
				ts_log_diff = ts_log - ts_log.shift()
				model = ARIMA(ts_log, order=(0, 1, 2))  
				results_ARIMA = model.fit(disp=-1) 
				results_ARIMA = model.fit(disp=-1)
        			predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
        			predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        			predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
        			predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
       				predictions_ARIMA_log.head()
        			predictions_ARIMA = np.exp(predictions_ARIMA_log)
				pred_time = time + timedelta(seconds = predictions_ARIMA[-1])
				pred_time1 = predictions_ARIMA[-1]
        			print "predicted", pred_time
				print pred_time1,act_time.total_seconds()
				plt.scatter(i,pred_time1,color='red')
				if(i>0):	
					plt.scatter(i-1,(act_time.total_seconds()),color='blue')
				plt.pause(1)
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
			act_time = (datetime.datetime.now() - curr_time)
			curr_time = datetime.datetime.now()
			print datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
		lasttext=text
		lastframe=currframe
		currframe=1
		# draw the text and timestamp on the frame
	cv2.putText(frame, "Request Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	#cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
	#	(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)

	key = cv2.waitKey(1) & 0xFF
 	
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
 	
# cleanup the camera and close any open windows
plt.savefig("act_pred_plot.jpg")
camera.release()
cv2.destroyAllWindows()
