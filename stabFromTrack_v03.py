#!/usr/local/bin/python3

import sys, argparse, re, time
import cv2
import numpy as np
import math


parser = argparse.ArgumentParser(description="Stabilize a movie file based on values in a data file in the following format:\n\tFRAME X Y.  \nFRAME is the frame number integer, X and Y are normalized floats. Blender\'s Y coordinate is flipped so if image has the same orientation in blender as it does in openCV view, yFlip argument is needed")

parser.add_argument('-fs', '--frameStart', default=-1, type=int, dest='frameStart', help='first frame of the stabilize')  
parser.add_argument('-fe', '--frameEnd',default=-1, type=int, dest='frameEnd', help='first frame of the stabilize')   
parser.add_argument('-fo', '--frameOffset',default=0, type=int, dest='frameOffset', help='offset the frame lookup when playback rate is varying')   

inputStyle = parser.add_mutually_exclusive_group()
inputStyle.add_argument('-td', '--trackData',default=False, type=str, dest='trackData', help='frame, X and Y values')   
inputStyle.add_argument('-pd', '--perspData',default=False, type=str, dest='perspData', help='frame + multiple track points for perspective warp')   

parser.add_argument('-pt', '--posTrack',default=0, type=int, dest='posTrack', help='which track to use for position')   
parser.add_argument('-xf', '--xFlip', action='store_true', dest='xFlip', help='invert the x value in case image was tracked with horizontal flip')   
parser.add_argument('-xr', '--xReverse', action='store_true',dest='xReverse', help='')   
parser.add_argument('-yf', '--yFlip', action='store_true', dest='yFlip', help='invert the y value in case image was tracked with vertical flip')   
parser.add_argument('-yr', '--yReverse', action='store_true', dest='yReverse', help='frame, X and Y values')   
parser.add_argument('-db', '--debug', action='store_true', dest='debug', help='frame, X and Y values')   
parser.add_argument('-xo', '--xOffset',default=0, type=float, dest='xOffset', help='offset in x position')   
parser.add_argument('-yo', '--yOffset',default=0, type=float, dest='yOffset', help='offset in y position')
parser.add_argument('-xp', '--xPad',default=0, type=float, dest='xPad', help='pad in width, default = 0, 1 = half of width on both side')   
parser.add_argument('-yp', '--yPad',default=0, type=float, dest='yPad', help='pad in height, default = 0, 1 = half of heigth on both side')
parser.add_argument('-ro', '--rOffset',default=0, type=float, dest='rOffset', help='offset in rotation value')
parser.add_argument('-sm', '--scaleMult',default=1, type=float, dest='scaleMult', help='multiplication on scale')
parser.add_argument('-rf', '--refFrame',default=-1, type=int, dest='refFrame', help='frame to use as reference (default is "frameStart"')   
parser.add_argument('-tk', '--token',default="stab", type=str, dest='token', help='token filename extension')   
parser.add_argument('-wm', '--writeMask', action='store_true', dest='writeMask', help='first frame of the stabilize')
parser.add_argument('-wo', '--writeOutput', action='store_true', dest='writeOutput', help='first frame of the stabilize')
parser.add_argument('inputMovie', help='input movie')



args = parser.parse_args()

frameStart = args.frameStart
frameEnd = args.frameEnd
frameOffset = args.frameOffset
writeOutput = args.writeOutput
writeMask = args.writeMask
trackData = args.trackData
perspData = args.perspData
posTrack = args.posTrack
xFlip = args.xFlip
xReverse = args.xReverse
yFlip = args.yFlip
yReverse = args.yReverse
debug = args.debug
xOffset = args.xOffset
yOffset = args.yOffset
xPad = args.xPad
yPad = args.yPad
rOffset = args.rOffset
scaleMult = args.scaleMult
refFrame = args.refFrame
token = args.token
inputMovie = args.inputMovie 

patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')
outputMovieFile = re.sub(patt, '_%s.mp4' % token, inputMovie)

print("\n")
print("inputMovie: %s" % inputMovie)
print("frameStart: %s" % frameStart)
print("frameEnd: %s" % frameEnd)
print("frameOffset: %s" % frameOffset)
print("trackData: %s" % trackData)
print("perspData: %s" % perspData)
print("posTrack: %s" % posTrack)
print("xFlip: %s" % xFlip)
print("xReverse: %s" % xReverse)
print("yFlip: %s" % yFlip)
print("yReverse: %s" % yReverse)
print("xOffset: %s" % xOffset)
print("yOffset: %s" % yOffset)
print("rOffset: %s" % rOffset)
print("scaleMult: %s" % scaleMult)
print("refFrame: %s" % refFrame)
print("writeMask: %s" % writeMask)
print("outputMovie: %s" % outputMovieFile)



######################
##     Functions
######################

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
		
def createTestPoints():
	pt0 = {}
	pt1 = {}
	w = 1
	h = 1
	ref0 = [.5*w,.5*h]
	ref1 = [.8*w,.6*h]


	for n in range(601):
		if n < 2:
			pt0[str(n)] = ref0
			pt1[str(n)] = ref1
		else:
			pt0[str(n)] = [0, 0]
			pt1[str(n)] = [.5*w, (math.sin(n*.05)*.5+.5)*h]

	return {0:{'ff': 1, 'lf': 600, 'data': pt0}, 1: {'ff':1, 'lf': 600, 'data':pt1}}

def readTrackerData(myFile, fo = 0):
	# read values from file
	with open(myFile) as file:
		tracker = file.readlines()

	frameFirst = int(tracker[0].split()[0])+fo
	frameLast  = int(tracker[-1:][0].split()[0])+fo
	clipRange = frameLast - frameFirst
	fileRange = len(tracker)
	# create dictionary of x y transforms per frame
	trackerDict = {}
	for v in tracker:
		frame, x, y = v.rstrip().split()
		frame = str(int(frame) + fo)
		if xFlip:
			x = 1.0-float(x)
		else:
			x = float(x)
		if yFlip:
			y = 1.0-float(y)
		else:
			y = float(y)
		if portrait:
			trackerDict[frame]=(y,x)
		else:
			trackerDict[frame]=(x,y)
			
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

def readPerspData(myFile, fo = 0):
	# read values from file
	with open(myFile) as file:
		tracker = file.readlines()

	frameFirst = int(tracker[0].split()[0])+fo
	frameLast  = int(tracker[-1:][0].split()[0])+fo
	clipRange = frameLast - frameFirst
	fileRange = len(tracker)
	# create dictionary of x y transforms per frame
	trackerDict = {}
	for v in tracker:
		tmp = v.split()
		frame = tmp[0]
		data = eval(" ".join(tmp[1:]))
		trackerDict[frame]=data

	return frameFirst, frameLast, trackerDict



def getMatrix(trackerDict, frame, frameRef, w, h, posTrack = 0, rot0Track = 0, rot1Track=1):
	rot = 0
	scale = 1
	
	pnt0 = trackerDict[posTrack]['data'][str(frame)]
	ref0 = trackerDict[posTrack]['data'][str(frameRef)]
	pnt0x, pnt0y = tuple(pnt0)
	ref0x, ref0y = tuple(ref0)

	if len(trackersDict) >= 2:
		pntR0 = trackerDict[rot0Track]['data'][str(frame)]
		refR0 = trackerDict[rot0Track]['data'][str(frameRef)]
		pntR0x, pntR0y = tuple(pntR0)
		refR0x, refR0y = tuple(refR0)
		pntR1 = trackerDict[rot1Track]['data'][str(frame)]
		refR1 = trackerDict[rot1Track]['data'][str(frameRef)]
		pntR1x, pntR1y = tuple(pntR1)
		refR1x, refR1y = tuple(refR1)

		refVectorx = (refR1x - refR0x)
		refVectory = (refR1y - refR0y)
		pntVectorx = (pntR1x - pntR0x)
		pntVectory = (pntR1y - pntR0y)

		refAngle = (math.atan2(refVectory*h/w, refVectorx)+2*math.pi)
		pntAngle = (math.atan2(pntVectory*h/w, pntVectorx)+2*math.pi)

		rot = (refAngle - pntAngle ) + rOffset
		scale = (math.dist([refR0x,refR0y*h/w],[refR1x,refR1y*h/w]) / math.dist([pntR0x,pntR0y*h/w],[pntR1x,pntR1y*h/w]))
		scale = scale * scaleMult

	Moffset = np.float32([
                [1,     0,      -pnt0x*w],
                [0,     1,      -pnt0y*h],
                [0,	0, 	1]
                ])
	Mplace = np.float32([
                [1,     0,      (ref0x+xOffset)*w],
                [0,     1,      (ref0y+yOffset)*h],
                [0,	0, 	1]
                ])
	Mr  = np.float32([      
                [scale*math.cos(rot),	-scale*math.sin(rot),	0],
                [scale*math.sin(rot),	scale*math.cos(rot),	0],
                [0,     		0,      		1]
                ])
	Mr = cv2.getRotationMatrix2D([0,0], -math.degrees(rot), scale )
	Mr = np.append(Mr, [[0,0, 1]], axis = 0)


	M = np.matmul(Mr, Moffset)
	M = np.matmul(Mplace, M)
	return M[:2]

def getXY(trackerDict, frame, w, h):
	v = trackerDict['data'][str(frame)]
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

outputMovieWidth = int(movieWidth*(1+xPad))
outputMovieHeight = int(movieHeight*(1+yPad))
print("\noutput movie height by width is: \t%s X %s"% (outputMovieHeight, outputMovieWidth))

xOffset += xPad*.5
yOffset += yPad*.5


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
	print("Portrait mode is not True")
	portrait = False

#--- Open and setup output file
outputMovie = cv2.VideoWriter(
	outputMovieFile,
	cv2.VideoWriter_fourcc('m','p','4','v'), 120, 
	(outputMovieWidth, outputMovieHeight)
) if writeOutput else None

outputMaskMovie = cv2.VideoWriter(
	outputMovieFile.replace(".mp4","_mask.mp4"),
	cv2.VideoWriter_fourcc('m','p','4','v'), 120, 
	(outputMovieWidth, outputMovieHeight)
) if writeMask else None



#read from tracking data file
if trackData:
	trackers = [t for t in trackData.split(':')]
	trackersDict = {}
	for i, v in enumerate(trackers):
		ff, lf, data = readTrackerData(v, frameOffset)
		trackersDict[i] = {'ff':ff, 'lf':lf, 'data':data}

elif perspData:
	trackersDict = {}
	ff, lf, data = readPerspData(perspData, frameOffset)	
	trackersDict[0] = {'ff':ff, 'lf':lf, 'data':data}

if frameStart == -1:
	frameStart = trackersDict[0]['ff']
if frameEnd == -1:
	frameEnd = trackersDict[0]['lf']
if refFrame == -1:
	refFrame = frameStart
print(frameStart, frameEnd, refFrame)


#--- Go to first specified frame
jump_to_frame(frameStart, movie)

i = frameStart

startTime = time.perf_counter()
frameRange = frameEnd - frameStart
frameCurrent = frameStart



while frameCurrent < frameEnd:

	ret, srcFrame = read_frame(movie)
	if writeMask:
		srcFrame = cv2.cvtColor(srcFrame, cv2.COLOR_RGB2RGBA)
		srcFrame[:,:,3] = 255

	if ret != True:
		print("Problem reading frame %s"%i)
		exit

	if trackData:
		mtx = getMatrix(trackersDict, frameCurrent + 1, refFrame, movieWidth, movieHeight, posTrack)
		if not debug:
			# draw red dots for tracked points
			if False:
				for k in trackersDict.keys():
					x,y = getXY(trackersDict[k], frameCurrent + 1, movieWidth, movieHeight)
					cv2.circle(srcFrame,(x,y),20,(100,0,255),-1)
			# warp
			dstStab = cv2.warpAffine(srcFrame, mtx, (outputMovieWidth,outputMovieHeight))
			# draw reference on warped image
			if False:
				for k in trackersDict.keys():
					x,y = getXY(trackersDict[k], refFrame, movieWidth, movieHeight)
					cv2.circle(dstStab,(x,y),15,(100,255,0),-1)
			dstStabShow = cv2.resize(dstStab,(int(movieWidth*.25),int(movieHeight*.25)))
			# blue dot grid
			for x in range(50,dstStabShow.shape[0],50):
				for y in range(50,dstStabShow.shape[1],50):
					cv2.circle(dstStabShow,(y,x),3,(255,0,0),-1)
		else:
			for k in trackersDict.keys():
				x,y = getXY(trackersDict[k], frameCurrent + 1, movieWidth, movieHeight)
				cv2.circle(srcFrame,(x,y),15,(100,0,255),-1)
			dstStabShow = cv2.resize(srcFrame,(int(movieWidth*.25),int(movieHeight*.25)))

	elif perspData:
		dstPoints = np.float32(trackersDict[0]['data'][str(refFrame)])
		srcPoints = np.float32(trackersDict[0]['data'][str(frameCurrent)])
		mtx = cv2.getPerspectiveTransform(srcPoints, dstPoints)
		for p in srcPoints:
			cv2.circle(srcFrame,(int(p[0]),int(p[1])),15,(0,0,255),-1)
		dstStab = cv2.warpPerspective(srcFrame, mtx, (outputMovieWidth,outputMovieHeight))
		#dstStab=srcFrame
		for p in dstPoints:
			cv2.circle(dstStab,(int(p[0]),int(p[1])),10,(255,2,5),-1)
		dstStabShow = cv2.resize(dstStab,(int(movieWidth*.25),int(movieHeight*.25)))

	cv2.imshow("stab",  dstStabShow)


	if writeOutput:
		outputFrame = cv2.cvtColor(dstStab, cv2.COLOR_RGBA2RGB)
		outputMovie.write(outputFrame)
		if writeMask:
			outputFrame[:,:,0] = dstStab[:,:,3]
			outputFrame[:,:,1] = dstStab[:,:,3]
			outputFrame[:,:,2] = dstStab[:,:,3]
			outputMaskMovie.write(outputFrame)
	
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
outputMaskMovie.release() 					if writeMask else None

print("\ndone")
print("complete:")
print("open %s"%outputMovieFile)
print("")


