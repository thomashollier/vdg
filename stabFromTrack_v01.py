#!/usr/local/bin/python3

import sys, argparse, re, time
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-fs', '--frameStart', default=1, type=int, dest='startFrame', help='first frame of the stabilize')  
parser.add_argument('-fe', '--frameEnd',default=1, type=int, dest='endFrame', help='first frame of the stabilize')   
parser.add_argument('-td', '--trackData',default=1, type=str, dest='trackData', help='frame, X and Y values')   
parser.add_argument('-wo', '--writeOutput', action='store_true', dest='writeOutput', help='first frame of the stabilize')

parser.add_argument('inputMovie', help='input movie')



args = parser.parse_args()

startFrame = args.startFrame
endFrame = args.endFrame
writeOutput = args.writeOutput
trackData = args.trackData
inputMovie = args.inputMovie 
patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')
outputMovieFile = re.sub(patt, '_stab.mp4', inputMovie)

print("inputMovie: %s" % inputMovie)
print("startFrame: %s" % startFrame)
print("endFrame: %s" % endFrame)
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

	# create dictionary of x y transforms per frame
	trackerDict = {}
	for v in tracker:
		k, x, y = v.rstrip().split()
		x = float(x)
		y = 1.0-float(y)
		trackerDict[k]=(x,y)

	return trackerDict

def getMatrix(trackerDict, frame, frameRef, w, h):
	v = trackerDict[str(frame)]
	r = trackerDict[str(frameRef)]
	X = (r[0]-v[0])*w
	Y = (r[1]-v[1])*h
	M = np.float32([[1,0,X],[0,1,Y]])
	return M

def getXY(trackerDict, frame, w, h):
	v = trackerDict[str(frame)]
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
print("input movie size is: %s %s"% (movieHeight, movieWidth))

#--- Load first frmae and print shape
ret, srcFrame = read_frame(movie)
if ret != True:
	print("Problem reading first frame")
	exit
print("loaded frame size is: %s %s"% (srcFrame.shape[0], srcFrame.shape[1]))

#--- Open and setup output file
outputMovie = cv2.VideoWriter(
	outputMovieFile,
	cv2.VideoWriter_fourcc('M','P','4','V'), 120, 
	(movieWidth,movieHeight)
) if writeOutput else None

#--- Go to first specified frame
frameRange = get_movie_range(movie)
jump_to_frame(startFrame, movie)


trackingData = readTrackerData(trackData)

refFrame = 2

i = startFrame

while i < endFrame:

	ret, srcFrame = read_frame(movie)
	if ret != True:
		print("Problem reading frame %s"%i)
		exit

	mtx = getMatrix(trackingData, i+1, refFrame, movieWidth, movieHeight)
	x,y = getXY(trackingData, i+1, movieWidth, movieHeight)
	if True:
		dstStab = cv2.warpAffine(srcFrame, mtx, (movieWidth,movieHeight))
		cv2.imshow("stab",  cv2.resize(dstStab,(int(movieWidth*.25),int(movieHeight*.25))))
	else:
		cv2.circle(srcFrame,(x,y),10,(255,0,0),-1)
		cv2.imshow("stab",  cv2.resize(srcFrame,(int(movieWidth*.25),int(movieHeight*.25))))
		dstStab = srcFrame

	outputMovie.write(dstStab) if writeOutput else None
	#
	print(i)
	i += 1
	# Exit if ESC pressed
	k = cv2.waitKey(1) & 0xff
	if k == 27 : break

movie.release()
outputMovie.release() 						if writeOutput else None
print("done")

