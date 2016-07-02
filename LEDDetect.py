'''
LED Detect
file: detectLED.py
function: detect LED and return center points of the led
Written by Shayle Murray
Last Updated: June 22, 2016
GITHUB: https://github.com/LikeTheRock/PoseEstimation
'''

#IMPORTS
import cv2
import numpy as np
import glob
import os
from scipy.spatial import distance
import math

#CONSTANTS
H_MIN = 0
H_MAX = 255
S_MIN = 0
S_MAX = 255
V_MIN = 0
V_MAX = 255
PATH1='/home/shayle/Desktop/SR/JetsonSamples/WithWhiteBalance/GBIndoor'
PATH2='/home/shayle/Desktop/SR/JetsonSamples/WithWhiteBalance/GBOutdoor'
PATH3='/home/shayle/Desktop/SR/JetsonSamples/WithWhiteBalance/GIndoor'
PATH4='/home/shayle/Desktop/SR/JetsonSamples/WithWhiteBalance/GBBIndoor'
TRACKBARWINDOWNAME = "Trackbars"
WINDOWNAME  = "Original Image"
WINDOWNAME2 = "Thresholded Image"
WINDOWNAME1 = "Hough Circles"
WINDOWNAME3 = "Contour Image"
MAX_RADIUS = 1 
NUM_LEDS = 3 #The number of points(LEDs) to track
GREEN = (0,255,0)
RED = (0,0,255)

#COMMAND SETS
CSET1 = '''Current Command Options:
	'ESC' - Quit
	'p'   - Pause vid feed & select ROI's
	'r'   - Resume vid feed'''
	
CSET2 = '''Current Command Options:
	Click radius point - hold & drag to center - release
	'r'   - reset all ROI selections
	'c'   - ROI selection complete'''

#CLASS 
'''
A class to have presets for filters based on color and indoor
'''
class Filter:
	H_MIN = 0
	H_MAX = 255
	S_MIN = 0
	S_MAX = 255
	V_MIN = 253
	V_MAX = 255
	def __init__(self, color, outdoor):
		if (color == "red"):
			self.H_MIN = 0
			self.H_MAX = 50
			self.S_MAX = 12
		if (color == "green"):
			self.H_MIN = 104
			self.H_MAX = 200 
			self.S_MAX = 5
		if (color == "blue"):
			self.H_MIN = 52
			self.H_MAX = 117
			self.S_MIN = 30
			self.S_MAX = 255
			self.V_MIN = 221
			self.V_MAX = 255
		if (color == "value"):
			self.H_MIN = 0
			self.H_MAX = 255
			self.S_MIN = 0
			self.S_MAX = 255
			self.V_MIN = 254
			self.V_MAX = 255
		if outdoor : #(1) outdoor
			self.V_MIN -= 70
			

#VARIABLES
loop = 1 #(1) loop video till esc, (0) end at end of video
selectPath = 1 # 0 is loop all vids, 1-3 => PATH()
picFrame = 50 #Select the frame to view from one of the paths
pathTemp = PATH1
resize = 1
refPoints = [] #Capture points for the ROI will be set of (x,y))
circleROI = [] #set of circles which are the ROIs
#image = 0#a blank image to be used in ROI capture

#FUNCTIONS
def nothing(x):
	pass
	
'''
Returns the next path, given the current path
'''
def getNextPath(pastPath):
	if (pastPath==PATH1):
		return PATH2
	elif (pastPath == PATH2):
		return PATH3
	elif (pastPath == PATH3):
		return PATH4
	else: 
		return PATH1

'''
Creates trackbars to adjust the HSV filter
'''
def createBars():
	''' Create the sliders to filter out certain color in HSV image. 
	'''
	cv2.namedWindow(TRACKBARWINDOWNAME,0)
	cv2.createTrackbar("H_MIN",TRACKBARWINDOWNAME,colorFilter.H_MIN,H_MAX,nothing )
	cv2.createTrackbar("H_MAX",TRACKBARWINDOWNAME,colorFilter.H_MAX,H_MAX,nothing )
	cv2.createTrackbar("S_MIN",TRACKBARWINDOWNAME,colorFilter.S_MIN,S_MAX,nothing )
	cv2.createTrackbar("S_MAX",TRACKBARWINDOWNAME,colorFilter.S_MAX,S_MAX,nothing )
	cv2.createTrackbar("V_MIN",TRACKBARWINDOWNAME,colorFilter.V_MIN,V_MAX,nothing )
	cv2.createTrackbar("V_MAX",TRACKBARWINDOWNAME,colorFilter.V_MAX,V_MAX,nothing )

'''
	Gets the index'th bmp from the directory from path
'''
def getBmp(path, index):
	os.chdir(path)
	img = 0 
	counter =0
	l = len(glob.glob( os.path.join(path, "*.bmp")))-1
	if index >= l:
		return 0
	for infile in sorted(glob.glob( os.path.join(path, "*.bmp"))):
		if counter == index:
			img = infile
			break
		else:
			counter+=1
	return img
	
	
def morphOps(thresh):
	'''
	Get rid of white noise using dilation and erosion.
	Current kernal right now is used by 5px by 5px rectangle. 
	'''
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
	#modifiedImg = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	modImg2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
	return modImg2

'''
Detect and Draw circles
input: both the original image, and the thresholded image
Return: the frame with the circles drawn on it
'''
def detectCircles( cameraFeed,threshold ):
	#detect circles
	cThreshold = np.copy(threshold)
	cCameraFeed = np.copy(cameraFeed)
	#params : image, method, dp, minDist, circles, param1, param2, minRadius, maxRadius
	circles = cv2.HoughCircles(cThreshold,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=10,minRadius=1,maxRadius=10)
	if circles is None:
		return cCameraFeed
	circles2 = np.uint16(np.around(circles))
	for i in circles2[0,:]:
		# draw the outer circle
		cv2.circle(cCameraFeed,(i[0],i[1]),i[2],GREEN,2)
		# draw the center of the circle
		cv2.circle(cCameraFeed,(i[0],i[1]),2,RED,3)
	return cCameraFeed

'''
Displays the image passed, in a window titled 'name'.
 If rescalable is set to 1, then the window can be scalled
'''
def showWindow( name, img, rescalable):
	if rescalable:
		cv2.namedWindow( name, cv2.WINDOW_NORMAL)
	cv2.imshow(name, img)

'''
	Given the original image (cameraFeed) and the threshold'ed input
	shows a window of the minimim enclosing circles everlayed 
'''
def contourTrack(cameraFeed, threshold):
	#TODO check circles circumf ~~ area
	cThreshold=np.copy(threshold)
	contouredFeed = np.copy(cameraFeed)
	# findContours( source image, contour retrieval mode, contour approximation method)
	_, contours, hierarchy = cv2.findContours(cThreshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	#Draw All Contours
	#DrawContours(source image,the contours as a Python list, index of contours (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.
	#cv2.drawContours(contouredFeed, contours, -1, GREEN, 3)
	
	#Draw Contours Smaller than Max_Radius
	center = (0,0)
	for cnt in contours:
		#calculate minimim enclosing circle
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		if radius < MAX_RADIUS and radius > .5: #TODO Adjust this 
			center = (int(x),int(y))
    		radius = int(radius)
    		cv2.circle(contouredFeed,center,radius,GREEN,1)
	
	#Create & show window with result'''
	showWindow(WINDOWNAME3, contouredFeed, resize)

def compareFeed( feedOne, feedTwo, feedOneName, FeedTwoName):
	'''
		Takes two feeds and highlights the difference in a new window
	'''
		#TODO finish this, does not add to blue & red masks
	#get dimension of feeds
	f1x, f1y = feedOne.shape[:2]
	f1x, f1y = feedOne.shape[:2]
	
	#generate monochromatic images to bit_and
	blueMask =  np.zeros((f1x,f1y,3), np.uint8)
	redMask = np.zeros((f1x,f1y,3), np.uint8)
	blackMask = np.zeros((f1x,f1y,3), np.uint8)
	blueMask[:]=(255,0,0)
	redMask[:] = RED
	blackMask[:]=(0,0,0)
	
	#Subtract a from b, and vise versa (xor)
	amb = feedOne - feedTwo 
	bma = feedTwo - feedOne
	
	#add a-b to blue, b-a to a
	blackMask = cv2.bitwise_and(amb, amb, blueMask)
	#redMask  = cv2.bitwise_and(redMask, redMask, bma = bma)
	
	#add the above results to black
	#blackMask = cv2.bitwise_and(blueMask, BlackMask)
	#blackMask = cv2.bitwise_and(redMask, BlackMask)
	
	#mask = cv2.bitwise_and()
	title = feedOneName + FeedTwoName
	showWindow(title, blueMask ,resize)
	#showWindow(title, blackMask,resize)

def click_ROI(event, x, y, flags, param):
	'''
		Captures the mouse position, calculates & draws a circleROI on the original image
	'''
		#TODO try removing extra parameters
	radius=0
	#TODO make this select circles 
	#TODO remove dependancies on global var
	# grab references to the global variables
	global refPoints, circleROI

	# if the left mouse button was clicked, record the starting (x, y) coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		refPoints = [(x, y)]

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates 
		refPoints.append((x,y))
		
		#calculate radius and cast as interger
		radius = int(math.ceil( distance.euclidean(refPoints[0], refPoints[1]) ) )
		
		#Add circle to circleROI
		circleROI.append((x,y,radius))
		
		# draw a circle around the region of interest
		#cv2.circle(image, refPoints[0], radius, GREEN, 2)
		#cv2.imshow("image", image)

def identifyROI(circles):
	'''
		updates the ROIs and
		Draws shapes around the ROIs on the original image 
	'''
	pass #TODO
	
def userDefineROI(image, winName): 
	'''
		Has the user draw circles on the stil frame to initialize
		tracking of the ROI. 
	'''	
	#TODO How to do this on the original image
	#clone image, and setup the mouse callback function
	global circleROI
	clone = image.copy()
	winNameNew = winName + " ROI Select"
	cv2.namedWindow(winNameNew)
	cv2.setMouseCallback(winNameNew, click_ROI) 
	pastLenth = 0 #Past length of ROICircle

	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow(winNameNew, image) #Why is this image, and not clone
		key = cv2.waitKey(1) & 0xFF
		
		if pastLenth < len(circleROI):
			#draw circle
			cv2.circle(image, circleROI[pastLenth][0:2], circleROI[pastLenth][2], GREEN, 2)
			pastLenth +=1
		
		# if the 'r' key is pressed, reset the ROI
		if key == ord("r"):
			image = clone.copy()
			circleROI = 0 
			pastLen = 0

		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			break

	if len(circleROI) == NUM_LEDS:
		cv2.destroyWindow(winNameNew)

	
'''
Select the method of input. 
	-Video from bmp's in folder
		-Video on loop 
		-all videos on loop
	-live from jetson
		TODO
'''

#initialize filter
colorFilter = Filter('blue', 1)

# create slider bars for HSV filtering
createBars()

# start an infinite loop to loop through all images
c=0 #initialize counter
print CSET1
while(1):
	#TODO implement if block to select imput method
	c+=1
	nextBmp = getBmp(pathTemp, c)
	if nextBmp != 0:
		# capture the frame
		cameraFeed = cv2.imread( nextBmp ,1 )
		
		#TODO Put all of this in a method, that can be opened with an input
		#	Update command set to include option
		# Get trackbar positions
		colorFilter.H_MIN = cv2.getTrackbarPos("H_MIN",TRACKBARWINDOWNAME)
		colorFilter.H_MAX = cv2.getTrackbarPos("H_MAX",TRACKBARWINDOWNAME)
		colorFilter.S_MIN = cv2.getTrackbarPos("S_MIN",TRACKBARWINDOWNAME)
		colorFilter.S_MAX = cv2.getTrackbarPos("S_MAX",TRACKBARWINDOWNAME)
		colorFilter.V_MIN = cv2.getTrackbarPos("V_MIN",TRACKBARWINDOWNAME)
		colorFilter.V_MAX = cv2.getTrackbarPos("V_MAX",TRACKBARWINDOWNAME)
		
		# convert frame from BGR to HSV colorspace
		HSV = cv2.cvtColor(cameraFeed,cv2.COLOR_BGR2HSV)
		grey = cv2.cvtColor(cameraFeed,cv2.COLOR_BGR2GRAY)
		
		# filter HSV image between values and store filtered image to threshold
		lower_mask = np.array([colorFilter.H_MIN,colorFilter.S_MIN,colorFilter.V_MIN])
		higher_mask = np.array([colorFilter.H_MAX,colorFilter.S_MAX,colorFilter.V_MAX])
		mask = cv2.inRange(HSV,lower_mask,higher_mask)
		threshold = cv2.bitwise_and(grey, grey, mask = mask)
		
		
		#Morph operations
		threshold2 = morphOps(threshold)
		
		#detect Circles using Hough 
		feedAndCircles = detectCircles(cameraFeed, threshold)
		
		#Compare the effectivness of the two filter methods
		#compareFeed( threshold, threshold2, WINDOWNAME2, "afterMorph")
		
		#TODO Implement contuor detection 
		contourTrack(cameraFeed, threshold)

		#TODO implement a ROI (Reigon of Interest)
		
		# show frames 
		showWindow(WINDOWNAME,cameraFeed,resize)
		showWindow(WINDOWNAME1,feedAndCircles,resize)
		showWindow(WINDOWNAME2,threshold,resize)
		showWindow("afterMorph",threshold2,resize)
		

		# allow 30 ms for screen to refresh and wait for ESC
		k = cv2.waitKey(30) & 0xFF
		if k == 112 : #p pause
			#TODO click the LEDS
			print CSET2
			userDefineROI(cameraFeed, WINDOWNAME)
			while(k != 114):
				#wait till new key == r
				k = cv2.waitKey(30) & 0xFF
		if k == 27:
			break
	else:
		if loop: #restart video
			c=0
			if selectPath:
				pathTemp=PATH4
				#pathTemp=getNextPath(pathTemp)
		else:
			break #End of vid

cv2.destroyAllWindows()
