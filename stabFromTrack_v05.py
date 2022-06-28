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
parser.add_argument('-g', '--gamma',default=1, type=float, dest='gamma', help='gamma correction')
parser.add_argument('-ev', '--exposureAdjust',default=0, type=float, dest='exposureAdjust', help='exposure adjustment in STOPS')
parser.add_argument('-rf', '--refFrame',default=-1, type=int, dest='refFrame', help='frame to use as reference (default is "frameStart"')   
parser.add_argument('-sl', '--setLandscape', action='store_true', dest='setLandscape', help='force landscape orientation')   
parser.add_argument('-sp', '--setPortrait', action='store_true', dest='setPortrait', help='force portrait orientation')   
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
gamma = args.gamma
exposureAdjust = args.exposureAdjust
refFrame = args.refFrame
setLandscape = args.setLandscape
setPortrait = args.setPortrait
token = args.token
inputMovie = args.inputMovie 

patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')
outputMovieFile = re.sub(patt, '_%s.mp4' % token, inputMovie)
APPLY_LUT = False

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
print("gamma: %s" % gamma)
print("exposureAdjust: %s" % exposureAdjust)
print("refFrame: %s" % refFrame)
print("setLandscape: %s" % setLandscape)
print("setPortrait: %s" % setPortrait)
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
	if ret != True:
		print("Problem reading frame %s"%i)
		exit
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


def orientCoordinates(xy, applyScale = True):
	# swapping x and y
	if portrait:
		# When movie loads up rotated 90deg CCW in blender
		y = xy[0]
		x = xy[1]
	else:
		# When movie shows up landscape right side up in blender
		x = xy[0]
		y = 1-xy[1]
	# inversing x and y coordinates
	# x and y coordinates now refer to the x and y coordinates as they
	# appear in the window being displayed in this script (not in blender)
	if xFlip:
		x = 1.0-float(x)
	if yFlip:
		y = 1.0-float(y)
	if applyScale:
		x = x * movieWidth
		y = y * movieHeight

	return [x,y]


def flipFlop(pnt):
	# Default is for blender to opencv
	# Blender vertical 0 is bottom
	# OpenCV vertical 0 is top
	output = [pnt[0], 1-pnt[1]]
 
	return [pnt[0], 1-pnt[1]]

def readTrackData(trackData, myRefFrame, fo = 0):

	# pre sort track files into common dictionary
	trackers = [t for t in trackData.split(':')]
	tmpDict = {}
	with open(trackers[0]) as file:
		lines = file.readlines()
		st = int(lines[0].split()[0])
		frames = set([x for x in range(st, st+len(lines))])
	# read each tracker and
	for i, v in enumerate(trackers):
		tmpDict[i] = {}
		with open(v) as file:
	                tracker = file.readlines()
		for f in range(len(tracker)):
			line = tracker[f]
			frame = line.split()[0]
			frame = int(frame) + fo
			data = eval("".join(line.split()[1:]))[0]
			tmpDict[i][frame]=data
		frames = ( frames & set(tmpDict[i]))
	frames = sorted(list(frames))
	###  Build dictionary
	trackerData = {}
	refFrameData = False
	for f in range(len(frames)):
		frame = frames[f]
		markerData = []
		for k in tmpDict.keys():
			markerData.append(tmpDict[k][frame])
		for i, xy in enumerate(markerData):
			markerData[i] = orientCoordinates(xy, applyScale = False)	
		trackerData[f+1]={'markerData':markerData, 'movieFrameNumber':frame}
		if frame == myRefFrame:
			refFrameData = markerData
	if not refFrameData:
		refFrameData = trackerData[1]['markerData']
		myRefFrame = trackerData[1]['movieFrameNumber']
	return {'ff':1, 'lf':len(trackerData), 'trackerData':trackerData, 'refData': refFrameData, 'refFrameInMovie':myRefFrame }

def readPerspData(myFile, myRefFrame, fo = 0):
	# read values from file
	with open(myFile) as file:
		tracker = file.readlines()

	###  Build dictionary
	trackerData = {}
	refFrameData = False
	for f in range(len(tracker)):
		line = tracker[f]
		frame = line.split()[0]
		frame = int(frame) + fo
		markerData = eval(" ".join(line.split()[1:]))
		for i, xy in enumerate(markerData):
			markerData[i] = orientCoordinates(xy)	
		trackerData[f+1]={'markerData':markerData, 'movieFrameNumber':frame}
		if frame == myRefFrame:
			refFrameData = markerData
	if not refFrameData:
		refFrameData = trackerData[1]['markerData']
		myRefFrame = trackerData[1]['movieFrameNumber']
	return {'ff':1, 'lf':len(trackerData), 'trackerData':trackerData, 'refData': refFrameData, 'refFrameInMovie':myRefFrame }

def getMatrix(frameData, refFrameData, w, h, posTrack = 0, rot0Track = 0, rot1Track=1):
	rot = 0
	scale = 1
	
	def normToReal(pnt):
		return [pnt[0]*w, pnt[1]*h]

	pnt0 = normToReal(frameData[posTrack])
	ref0 = normToReal(refFrameData[posTrack])
	pnt0x, pnt0y = pnt0
	ref0x, ref0y = ref0

	if len(frameData) >= 2:
		pntR0 = normToReal(frameData[rot0Track])
		refR0 = normToReal(refFrameData[rot0Track])
		pntR0x, pntR0y = (pntR0)
		refR0x, refR0y = (refR0)
		pntR1 = normToReal(frameData[rot1Track])
		refR1 = normToReal(refFrameData[rot1Track])
		pntR1x, pntR1y = (pntR1)
		refR1x, refR1y = (refR1)

		refVectorx = (refR1x - refR0x)
		refVectory = (refR1y - refR0y)
		pntVectorx = (pntR1x - pntR0x)
		pntVectory = (pntR1y - pntR0y)

		refAngle = (math.atan2(refVectory, refVectorx)+2*math.pi)
		pntAngle = (math.atan2(pntVectory, pntVectorx)+2*math.pi)

		rot = (refAngle - pntAngle ) + rOffset
		scale = (math.dist([refR0x,refR0y],[refR1x,refR1y]) / math.dist([pntR0x,pntR0y],[pntR1x,pntR1y]))
		scale = scale * scaleMult

	Moffset = np.float32([
		[1,     0,      -pnt0x],
		[0,     1,      -pnt0y],
		[0,     0,      1]
		])
	Mplace = np.float32([
		[1,     0,      (ref0x+xOffset)],
		[0,     1,      (ref0y+yOffset)],
		[0,     0,      1]
		])
	Mr  = np.float32([
		[scale*math.cos(rot),   -scale*math.sin(rot),   0],
		[scale*math.sin(rot),   scale*math.cos(rot),    0],
		[0,		     0,		      1]
		])
	Mr = cv2.getRotationMatrix2D([0,0], -math.degrees(rot), scale )
	Mr = np.append(Mr, [[0,0, 1]], axis = 0)


	M = np.matmul(Mr, Moffset)
	M = np.matmul(Mplace, M)
	return M[:2]


def getXY(trackerDict, frame, w, h, transpose = False):
	v = trackerDict['data'][frame]
	X = int(v[0]*w)
	Y = int(v[1]*h)
	if transpose:
		return [Y,X]
	else:
		return [X,Y]



def srgb2lin(s):
	if s <= 0.0404482362771082:
		lin = s / 12.92
	else:
		lin = pow(((s + 0.055) / 1.055), 2.4)
	return lin

def lin2srgb(lin):
	if lin > 0.0031308:
		s = 1.055 * (pow(lin, (1.0 / 2.4))) - 0.055
	else:
		s = 12.92 * lin
	return s

def exposure(v, a):
	lv = srgb2lin(v)
	lv = lv * (2 ** a)
	v = lin2srgb(lv)
	return(v)

def createLUT():
	lookUpTable = np.empty((1,256), np.uint8)
	for i in range(256):
		f = i / 255.0
		f = exposure(f, exposureAdjust)
		lookUpTable[0,i] = np.clip(pow(f, 1.0/gamma) * 255.0, 0, 255)
	return lookUpTable


##     START
######################

### Open input file and print metadata width and height
movie = cv2.VideoCapture(inputMovie)
movieHeight = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
movieWidth = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
if setLandscape:
	movieHeight = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
	movieWidth = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))

outputMovieWidth = int(movieWidth + xPad)
outputMovieHeight = int(movieHeight + yPad)

xOffset += (xPad * .5)
yOffset += (yPad * .5)

### Load first frame and print shape
ret, srcFrame = read_frame(movie)

portrait = movieHeight > movieWidth

print("\n---- GEOMETRY ----")
print("Input movie height x width is: \t\t%s x %s"% (movieHeight, movieWidth))
print("Loaded frame height by width is: \t%s X %s"% (srcFrame.shape[0], srcFrame.shape[1]))
print("Portrait mode is: \t\t\t%s" % portrait)
print("Output movie height x width is: \t%s x %s"% (outputMovieHeight, outputMovieWidth))

### Open and setup output file
outputMovie = cv2.VideoWriter(
	outputMovieFile,
	cv2.VideoWriter_fourcc('m','p','4','v'), 30, 
	(outputMovieWidth, outputMovieHeight)
) if writeOutput else None
outputMaskMovie = cv2.VideoWriter(
	outputMovieFile.replace(".mp4","_mask.mp4"),
	cv2.VideoWriter_fourcc('m','p','4','v'), 30, 
	(outputMovieWidth, outputMovieHeight)
) if writeMask else None


### Read from tracking data file
if trackData:
	trackersDict = readTrackData(trackData, refFrame, frameOffset)
#	print(trackersDict)
elif perspData:
	trackersDict = readPerspData(perspData, refFrame, frameOffset)

if frameStart == -1:
	frameStart = trackersDict['ff']
if frameEnd == -1:
	frameEnd = trackersDict['lf']
if refFrame == -1:
	refFrame = frameStart

startTime = time.perf_counter()
frameRange = 1 + frameEnd - frameStart
frameCurrent = frameStart


def almostIdentity(x,m,n ):
	if x>m:
		return x
	a = 2.0*n - m
	b = 2.0*m - 3.0*n
	t = x/m
	return (a*t + b)*t*t + n


if gamma != 1 or exposureAdjust != 0:
	APPLY_LUT = True
	lookUpTable = createLUT()

print("\n---- RANGES ----")
print("Comp frame range: \t\t%s-%s (%s frames inclusive)"% (frameStart, frameEnd, frameRange))
print("Movie frame range: \t\t%s-%s" % (trackersDict['trackerData'][frameStart]['movieFrameNumber'], trackersDict['trackerData'][frameEnd]['movieFrameNumber']))
print("Reference frame: \t\t%s\n"% (trackersDict['refFrameInMovie']))

#--- Go to first specified frame
movieFrameCurrent = trackersDict['trackerData'][1]['movieFrameNumber']
movieFramePrevious = movieFrameCurrent-1
jump_to_frame(movieFramePrevious, movie)
framePrevious = 0

# MAIN LOOP
while True:
	if frameCurrent == frameEnd + 1:
		break

	movieFrameCurrent = trackersDict['trackerData'][frameCurrent]['movieFrameNumber']
	sourceTrackData = trackersDict['trackerData'][frameCurrent]['markerData']
	

	if movieFrameCurrent != movieFramePrevious + 1:
		jump_to_frame(movieFrameCurrent-1, movie)
	ret, srcFrame = read_frame(movie)

	if APPLY_LUT:
		srcFrame = cv2.LUT(srcFrame, lookUpTable)
	if writeMask:
		srcFrame = cv2.cvtColor(srcFrame, cv2.COLOR_RGB2RGBA)
		srcFrame[:,:,3] = 255


	if trackData:
		markerData = trackersDict['trackerData'][frameCurrent]['markerData']
		refMarkerData = trackersDict['trackerData'][refFrame]['markerData']
#		for p in markerData:
#			cv2.circle(srcFrame,(int(p[0]*movieWidth),int((p[1])*movieHeight)),20,(100,0,255),-1)
		mtx = getMatrix(markerData, refMarkerData, movieWidth, movieHeight, posTrack)
		dstStab = cv2.warpAffine(srcFrame, mtx, (outputMovieWidth,outputMovieHeight))
#		for p in refMarkerData:
#			cv2.circle(dstStab,(int(p[0]*movieWidth),int((p[1])*movieHeight)),20,(255,0,100),-1)
		dstStabShow = cv2.resize(dstStab,(int(outputMovieWidth*.25),int(outputMovieHeight*.25)))

	elif perspData:
		ar1 = movieWidth/movieHeight
		ar2 = outputMovieWidth/outputMovieHeight
		dstScale = np.float32([movieWidth, movieHeight])*scaleMult
		dstScale = np.float32([1, 1])*scaleMult
		dstOffset = np.float32([xOffset, yOffset])
		dstPoints = np.float32(trackersDict['refData'])*dstScale + dstOffset
		srcPoints = np.float32(trackersDict['trackerData'][frameCurrent]['markerData'])
		mtx = cv2.getPerspectiveTransform(srcPoints, dstPoints)
		R = cv2.getRotationMatrix2D([movieWidth/2, movieHeight/2], -math.degrees(rOffset) , 1);
		R = np.vstack([R,np.float32([0,0,1])])
		mtx = np.matmul(R,mtx)
		dstStab = cv2.warpPerspective(srcFrame, mtx, (outputMovieWidth,outputMovieHeight))
		dstStabShow = cv2.resize(dstStab,(int(outputMovieWidth*.25),int(outputMovieHeight*.25)))

	cv2.imshow("stab",  dstStabShow)

	if writeOutput:
		outputFrame = cv2.cvtColor(dstStab, cv2.COLOR_RGBA2RGB)
		outputMovie.write(outputFrame)
		if writeMask:
			outputFrame[:,:,0] = dstStab[:,:,3]
			outputFrame[:,:,1] = dstStab[:,:,3]
			outputFrame[:,:,2] = dstStab[:,:,3]
			outputMaskMovie.write(outputFrame)
	
	percentDone = 0.000001+100*float(frameCurrent-frameStart)/float(frameRange)
	elapsedTime = time.perf_counter()-startTime
	remainingTime = int(.99+elapsedTime/percentDone * (100-percentDone) )
	remainingTime = "%02d:%02d" % (int(remainingTime/60), int(remainingTime%60))
	sys.stdout.write("\rFrame %4d (source frame: %s) of %s -- %.02f%% complete -- %s seconds remaining" % (1+frameCurrent-frameStart, movieFrameCurrent, frameRange, percentDone, remainingTime))
	#sys.stdout.flush()

	frameCurrent = frameCurrent + 1
	movieFramePrevious = movieFrameCurrent  
	

	# Exit if ESC pressed
	k = cv2.waitKey(1) & 0xff
	if k == 27 : break

	while False:
		time.sleep(10)


movie.release()
outputMovie.release() 						if writeOutput else None
outputMaskMovie.release() 					if writeMask else None

print("\ndone")
print("complete:")
print("open %s"%outputMovieFile)
print("")


