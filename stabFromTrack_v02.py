#!/usr/local/bin/python3

import sys, argparse, re, time
import cv2
import numpy as np


parser = argparse.ArgumentParser(description="Stabilize a movie file based on values in a data file in the following format:\n\tFRAME X Y.  \nFRAME is the frame number integer, X and Y are normalized floats. Blender\'s Y coordinate is flipped so if image has the same orientation in blender as it does in openCV view, yFlip argument is needed")

parser.add_argument('-fs', '--frameStart', default=-1, type=int, dest='frameStart', help='first frame of the stabilize')  
parser.add_argument('-fe', '--frameEnd',default=-1, type=int, dest='frameEnd', help='first frame of the stabilize')   
parser.add_argument('-td', '--trackData',default=1, type=str, dest='trackData', help='frame, X and Y values')   
parser.add_argument('-xf', '--xFlip', action='store_true', dest='xFlip', help='invert the x value in case image was tracked with horizontal flip')   
parser.add_argument('-xr', '--xReverse', action='store_true',dest='xReverse', help='')   
parser.add_argument('-yf', '--yFlip', action='store_true', dest='yFlip', help='invert the y value in case image was tracked with vertical flip')   
parser.add_argument('-yr', '--yReverse', action='store_true', dest='yReverse', help='frame, X and Y values')   
parser.add_argument('-db', '--debug', action='store_true', dest='debug', help='frame, X and Y values')   
parser.add_argument('-rf', '--refFrame',default=-1, type=int, dest='refFrame', help='frame to use as reference (default is "frameStart"')   
parser.add_argument('-tk', '--token',default="stab", type=str, dest='token', help='token filename extension')   
parser.add_argument('-wo', '--writeOutput', action='store_true', dest='writeOutput', help='first frame of the stabilize')
parser.add_argument('inputMovie', help='input movie')



args = parser.parse_args()

frameStart = args.frameStart
frameEnd = args.frameEnd
writeOutput = args.writeOutput
trackData = args.trackData
xFlip = args.xFlip
xReverse = args.xReverse
yFlip = args.yFlip
yReverse = args.yReverse
debug = args.debug
refFrame = args.refFrame
token = args.token
inputMovie = args.inputMovie 

patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')
outputMovieFile = re.sub(patt, '_%s.mp4' % token, inputMovie)


print("\n")
print("inputMovie: %s" % inputMovie)
print("frameStart: %s" % frameStart)
print("frameEnd: %s" % frameEnd)
print("xFlip: %s" % xFlip)
print("xReverse: %s" % xReverse)
print("yFlip: %s" % yFlip)
print("yReverse: %s" % yReverse)
print("refFrame: %s" % refFrame)
print("outputMovie: %s" % outputMovieFile)


def get_movie_range(movie):
	movie.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
	length = movie.get(cv2.CAP_PROP_FRAME_COUNT)
	movie.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
	return(length)

def jump_to_frame(frame, movie):
	movie.set(cv2.CAP_PROP_POS_FRAMES, frame)

def read_frame(movie):
	ret, frame = movie.read()
	return(ret, frame)
		
def readTrackerData(myFile):
	# read values from file
	with open(myFile) as file:
		tracker = file.readlines()

	frameFirst = int(tracker[0].split()[0])
	frameLast  = int(tracker[-1:][0].split()[0])
	clipRange = frameLast - frameFirst
	fileRange = len(tracker)
	# create dictionary of x y transforms per frame
	trackerDict = {}
	for v in tracker:
		k, x, y = v.rstrip().split()
		if xFlip:
			x = 1.0-float(x)
		else:
			x = float(x)
		if yFlip:
			y = 1.0-float(y)
		else:
			y = float(y)
		trackerDict[k]=(x,y)

	# fill in missing frames by copying last value
	n = frameFirst
	latest = 0
	while n < frameLast:
		if str(n) in trackerDict.keys():
			latest = n
		else:
			print("missing frame %s in track file, using frame %s" % (n, latest))
			trackerDict[str(n)] = trackerDict[str(latest)]
		n = n + 1


	return frameFirst, frameLast, trackerDict

def getMatrix(trackerDict, frame, frameRef, w, h):
	v = trackerDict[str(frame)]
	r = trackerDict[str(frameRef)]

	# swap x and y if image is portrait
	if portrait:
		vx = v[1]
		rx = r[1]
		vy = v[0]
		ry = r[0]
	else:
		vx = v[0]
		rx = r[0]
		vy = v[1]
		ry = r[1]

	if xReverse:
		X = (vx-rx)*w
	else:
		X = (rx-vx)*w
	if yReverse:
		Y = (vy-ry)*h
	else:
		Y = (ry-vy)*h
	M = np.float32([[1,0,X],[0,1,Y]])
	return M

def getXY(trackerDict, frame, w, h):
	v = trackerDict[str(frame)]
	if portrait:
		X = int(v[1]*w)
		Y = int(v[0]*h)
	else:
		X = int(v[0]*w)
		Y = int(v[1]*h)
	return [X,Y]


######################
##     START
######################


#--- Open input file and print metadata width and height
movie = cv2.VideoCapture(inputMovie)
movieHeight = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
movieWidth = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
print("\ninput movie height by width is: \t%s X %s"% (movieHeight, movieWidth))

#--- Load first frmae and print shape
ret, srcFrame = read_frame(movie)
if ret != True:
	print("Problem reading first frame")
	exit
print("loaded frame height by width is: \t%s X %s"% (srcFrame.shape[0], srcFrame.shape[1]))

if movieHeight > movieWidth:
	print("Portrait mode is True")
	portrait = True
else:
	portrait = False

#--- Open and setup output file
outputMovie = cv2.VideoWriter(
	outputMovieFile,
	cv2.VideoWriter_fourcc('M','P','4','V'), 120, 
	(movieWidth,movieHeight)
) if writeOutput else None


#read from trackign data file
ff, lf, trackingData = readTrackerData(trackData)

if frameStart == -1:
	frameStart = ff
if frameEnd == -1:
	frameEnd = lf
if refFrame == -1:
	refFrame = frameStart

#--- Go to first specified frame
jump_to_frame(frameStart, movie)

i = frameStart

startTime = time.perf_counter()
frameRange = frameEnd - frameStart
frameCurrent = frameStart
while frameCurrent < frameEnd:

	ret, srcFrame = read_frame(movie)
	if ret != True:
		print("Problem reading frame %s"%i)
		exit

	mtx = getMatrix(trackingData, frameCurrent+1, refFrame, movieWidth, movieHeight)
	x,y = getXY(trackingData, frameCurrent+1, movieWidth, movieHeight)
	if not debug:
		#cv2.circle(srcFrame,(x,y),30,(100,0,255),-1)
		
		dstStab = cv2.warpAffine(srcFrame, mtx, (movieWidth,movieHeight))
		dstStabShow = cv2.resize(dstStab,(int(movieWidth*.25),int(movieHeight*.25)))
		for x in range(50,dstStabShow.shape[0],50):
			for y in range(50,dstStabShow.shape[1],50):
				cv2.circle(dstStabShow,(y,x),3,(255,0,0),-1)
		cv2.imshow("stab",  dstStabShow)
	else:
		cv2.circle(srcFrame,(x,y),30,(100,0,255),-1)
		cv2.imshow("stab",  cv2.resize(srcFrame,(int(movieWidth*.25),int(movieHeight*.25))))
		dstStab = srcFrame

	outputMovie.write(dstStab) if writeOutput else None

	frameCurrent = frameCurrent + 1

	percentDone = 100.0*float(frameCurrent-frameStart)/float(frameRange)
	elapsedTime = time.perf_counter()-startTime
	remainingTime = int(.99+elapsedTime/percentDone * (100-percentDone) )
	remainingTime = "%02d:%02d" % (int(remainingTime/60), int(remainingTime%60))
	sys.stdout.write("\rFrame %4d of %s -- %.02f%% complete -- %s seconds remaining" % (frameCurrent-frameStart, frameRange, percentDone, remainingTime))
	sys.stdout.flush()



	# Exit if ESC pressed
	k = cv2.waitKey(1) & 0xff
	if k == 27 : break

movie.release()
outputMovie.release() 						if writeOutput else None
print("done")

