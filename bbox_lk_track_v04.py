#!/usr/bin/python
import sys
import cv2
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-ss', '--startFrame', default=1, type=int, dest='startFrame', help='first frame of the stabilize')  
parser.add_argument('-ef', '--endFrame', default=1, type=int, dest='endFrame', help='first frame of the stabilize')  
#parser.add_argument('-bbx', '--boundingBoxx', default="100:800:1:175", dest='bbox', help='x0:x1:y0:y1', action='append', nargs='*')  
#parser.add_argument('-bb', '--boundingBox', default=[], dest='bbox', help='x0:x1:y0:y1', action='append')  
parser.add_argument('-wo', '--writeOutput', default=False, type=bool, dest='writeOutput', help='first frame of the stabilize')  
parser.add_argument('-wx', '--writeXforms', default=False, type=bool, dest='writeXforms', help='first frame of the stabilize')  
parser.add_argument('-sp', '--showPoints', dest='showPoints', action='store_true', help='draw tracking points')  
parser.add_argument('-ff', '--findFeatures', dest='findFeatures', action='store_true', help='force find new features on each frame')  
parser.add_argument('-fq', '--featureQuality', default=.3, type=float, dest='featureQuality', help='first frame of the stabilize')  
parser.add_argument('-cl', '--clipLimit', default=2, type=float, dest='clipLimit', help='first frame of the stabilize')  
parser.add_argument('-bb', action='append', dest='bbox',
                    default=[],
                    help='Add repeated values to a list',)

parser.add_argument('inputMovie', help='input movie')



args = parser.parse_args()
print(args.bbox)

startFrame = args.startFrame
endFrame = args.endFrame
bbox= tuple([tuple([int(x) for x in y.split(":")]) for y in args.bbox])
writeOutput = args.writeOutput
writeXforms = args.writeXforms
showPoints = args.showPoints
findFeatures = args.findFeatures
featureQuality = args.featureQuality
clipLimit = args.clipLimit
inputMovie = args.inputMovie 

print(bbox)
print("inputMovie: %s" % inputMovie)
print("startFrame: %s" % startFrame)
print("endFrame: %s" % endFrame)
print("bbox: %s" % (bbox,))
print("writeOutput: %s" % writeOutput)
print("writeXforms: %s" % writeXforms)
print("showPoints: %s" % showPoints)
print("findFeatures: %s" % findFeatures)
print("featureQuality: %s" % featureQuality)
print("clipLimit: %s" % clipLimit)


# Parameters for ShiTomasi corner detection
####
feature_params = dict( maxCorners = 200,
                       qualityLevel = featureQuality,
                       minDistance = 3,
                       blockSize = 30 )

# Parameters for Lucas Kanade optical flow
####
lk_params = dict( winSize  = (40,40),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))

def matrix_to_transforms(matrix):
	if matrix is not None:
		# translation x
		dx = matrix[0, 2]
		# translation y
		dy = matrix[1, 2]
		# rotation
		da = np.arctan2(matrix[1, 0], matrix[0, 0])
	else:
		dx = dy = da = 0
	return [dx, dy, da]  

def transforms_to_matrix(transform):
	transform_matrix = np.zeros((2, 3))
	transform_matrix[0, 0] = np.cos(transform[2])
	transform_matrix[0, 1] = -np.sin(transform[2])
	transform_matrix[1, 0] = np.sin(transform[2]) 
	transform_matrix[1, 1] = np.cos(transform[2])
	transform_matrix[0, 2] = transform[0]
	transform_matrix[1, 2] = transform[1]
	return transform_matrix  

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
		
def prep_frame(frame, bbox):
	frameCrop = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]
	frameGray = cv2.cvtColor(frameCrop, cv2.COLOR_BGR2GRAY)
	return(frameGray)

def filterit(frame):
	sliding_window_size_x = 5
	sliding_window_size_y = 5
	mean_filter_kernel = np.ones((sliding_window_size_x,sliding_window_size_y),np.float32)/(sliding_window_size_x*sliding_window_size_y)
	filtered_image = cv2.filter2D(frame,-1,mean_filter_kernel)
	return filtered_image

def find_features(frame):
	pointsSrcOut = cv2.goodFeaturesToTrack(frame, mask = None, **feature_params)
	return(pointsSrcOut)

def track_frame(src, dst, pointsSrcInput, findFeatures): 
	if findFeatures:
		pointsSrcInput = find_features(src)
	pointsDstOutput = pointsSrcInput
	mtrx = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
	try:
		pointsDstOutput, st, err = cv2.calcOpticalFlowPyrLK(src, dst, pointsSrcInput, None, **lk_params)
		mtrx, inliers = cv2.estimateAffinePartial2D(pointsSrcInput, pointsDstOutput)
	except:
		mtrx = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
		err = np.array([[100],[100]])
	if cv2.mean(err)[0] > 30:
		mtrx = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
		err = np.array([[100],[100]])
		print("Tracking error too high, finding new points")
		mtrx, pointsDstOutput, err = track_frame(src, dst, pointsSrcInput, True)
	return(mtrx, pointsDstOutput, err)

def fix_bump(prev, curr, thresh):
	v = [prev[0]-curr[0], prev[1]-curr[1], 500*(prev[2]-curr[2])]
	mag = np.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
	if mag > thresh:
		#print("clamp %s" % mag)
		return prev
	else:
		return curr
	

######################
outputMovieFile = inputMovie.replace(".MP4", "_stab.MP4").replace(".MOV", "_stab.MP4")   
transformFile =   inputMovie.replace(".MP4", "_xforms.txt").replace(".MOV", "_xforms.txt") 
trajectoryFile =  inputMovie.replace(".MP4", "_trajec.txt").replace(".MOV", "_trajec.txt")
transforms = []
trajectory = []

movie = cv2.VideoCapture(inputMovie)
movieHeight = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
movieWidth = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
print("input movie size is: %s %s"% (movieHeight, movieWidth))
outputMovie = cv2.VideoWriter(
	outputMovieFile,
	cv2.VideoWriter_fourcc('M','P','4','V'), 120, 
	(movieWidth,movieHeight)
) if writeOutput else None

#frameRange = get_movie_range(movie)
#jump_to_frame(startFrame, movie)


# read the first frame and find some points to track
ret, dstFrame = read_frame(movie)
dstGray = prep_frame(dstFrame, bbox)
dstPoints = find_features(dstGray)
mtrx = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
trs = sum = sumW = [0,0,0]


i = 1
while True:
	if i == endFrame:
		break

	# use the previous frame as the source 
	srcFrame = dstFrame
	srcGray = dstGray
	srcPoints = dstPoints
	
	# read a new frame and make that the destination frame
	ret, dstFrame = read_frame(movie)
	if ret != True:
		break
	# apply the current source frame transform to the current dest frame
	dstGray = prep_frame(dstFrame, bbox)


	
	# find the features from source frame in the destination frame
	mtrx, dstPoints, err = track_frame(srcGray, dstGray, srcPoints, findFeatures)
	trs = matrix_to_transforms(mtrx)
	# the following zeros transforms if there is to much change
	if abs(trs[2])>np.radians(1):
		print(trs[2])	
		trs=[0,0,0]
	sum = [sum[0]-trs[0], sum[1]-trs[1], sum[2]-trs[2]]
	if showPoints:
		for p in dstPoints:
			cv2.circle(dstGray, (p[0][0], p[0][1]), 2, (255,0,0), -1)
	
	mtrxSum = transforms_to_matrix(sum)
	dstStab = cv2.warpAffine(dstFrame, mtrxSum, (movieWidth,movieHeight)) 
	dstStabGray = prep_frame(dstStab, bbox)
	

	cv2.imshow("gray",  dstGray)
	cv2.imshow("gray2",  dstStabGray)
	cv2.imshow("full",  cv2.resize(dstStab,(960,540)))
	outputMovie.write(dstFrameWW) if writeOutput else None
	transforms.append(trs)
	trajectory.append(sum)
	#
	i += 1
	# Exit if ESC pressed
	k = cv2.waitKey(1) & 0xff
	if k == 27 : break

movie.release()
outputMovie.release() 						if writeOutput else None
np.savetxt(transformFile, transforms, fmt='%4.8f %4.8f %4.8f') 	if writeXforms else None 
np.savetxt(trajectoryFile, trajectory, fmt='%4.8f %4.8f %4.8f') if writeXforms else None 
print("done")

