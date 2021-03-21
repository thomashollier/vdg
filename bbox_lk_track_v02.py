#!/usr/bin/python
import sys
import cv2
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-ss', '--startFrame', default=1, type=int, dest='startFrame', help='first frame of the stabilize')  
parser.add_argument('-bb', '--boundingBox', default="100:800:1:175", type=str, dest='bbox', help='first frame of the stabilize')  
parser.add_argument('-wo', '--writeOutput', default=False, type=bool, dest='writeOutput', help='first frame of the stabilize')  
parser.add_argument('-wx', '--writeXforms', default=False, type=bool, dest='writeXforms', help='first frame of the stabilize')  
parser.add_argument('-sp', '--showPoints', default=True, type=bool, dest='showPoints', help='first frame of the stabilize')  
parser.add_argument('-ff', '--findFeatures', default=True, type=bool, dest='findFeatures', help='first frame of the stabilize')  
parser.add_argument('-fq', '--featureQuality', default=.3, type=float, dest='featureQuality', help='first frame of the stabilize')  
parser.add_argument('-cl', '--clipLimit', default=2, type=float, dest='clipLimit', help='first frame of the stabilize')  
parser.add_argument('inputMovie', help='input movie')

args = parser.parse_args()
startFrame = args.startFrame
bbox= tuple([int(x) for x in args.bbox.split(":")])
writeOutput = args.writeOutput
writeXforms = args.writeXforms
showPoints = args.showPoints
findFeatures = args.findFeatures
featureQuality = args.featureQuality
clipLimit = args.clipLimit
inputMovie = args.inputMovie 


print("startFrame: %s \nbbox: %s \nwriteOutput: %s \nwriteXforms: %s \nshowPoints: %s \nfindFeatures: %s \nfeatureQuality: %s \nclipLimit: %s \ninputMovie: %s" % (startFrame, bbox, writeOutput, writeXforms, showPoints, findFeatures, featureQuality, clipLimit, inputMovie))


# Parameters for ShiTomasi corner detection
####
feature_params = dict( maxCorners = 200,
                       qualityLevel = featureQuality,
                       minDistance = 3,
                       blockSize = 16 )

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
	frameGray = clahe.apply(frameGray)
	return(frameGray)

def find_features(frame):
	#pointsSrc1 = cv2.goodFeaturesToTrack(frame[1:175, 1:150], mask = None, **feature_params)
	#pointsSrc2 = cv2.goodFeaturesToTrack(frame[1:175, 151:700], mask = None, **feature_params)
	#pointsSrc2Shift = np.add(pointsSrc2,np.full_like(pointsSrc2, [[150,0]]))	
	#pointsSrc = np.concatenate((pointsSrc1, pointsSrc2Shift))  
	#print(pointsSrc2.shape)
	pointsSrc = cv2.goodFeaturesToTrack(frame, mask = None, **feature_params)
	return(pointsSrc)

def track_frame(src, dst, pointsSrc): 
	pointsDst = pointsSrc
	try:
		pointsDst, st, err = cv2.calcOpticalFlowPyrLK(src, dst, pointsSrc, None, **lk_params)
		trs, inliers = cv2.estimateAffinePartial2D(pointsSrc, pointsDst)
	except:
		trs = transforms_to_matrix([0,0,0])
		err = np.array([[100],[100]])
	if cv2.mean(err)[0] > 5:
		print("error:", cv2.mean(err)[0])
		trs = transforms_to_matrix([0,0,0])
	return(trs, pointsDst, err)

def fix_bump(prev, curr, thresh):
	v = [prev[0]-curr[0], prev[1]-curr[1], 500*(prev[2]-curr[2])]
	mag = np.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
	if mag > thresh:
		print("clamp %s" % mag)
		return prev
	else:
		return curr
	

######################
outputMovieFile = inputMovie.replace(".MP4", "_stab.MP4")
transformFile = inputMovie.replace(".MP4", "_xforms.txt")
trajectoryFile = inputMovie.replace(".MP4", "_trajec.txt")
transforms = []
trajectory = []

movie = cv2.VideoCapture(inputMovie)
outputMovie = cv2.VideoWriter(
	outputMovieFile,
	cv2.VideoWriter_fourcc('M','P','4','V'), 120, 
	(1920,1080)
) if writeOutput else None
frameRange = get_movie_range(movie)
jump_to_frame(startFrame, movie)
ret, srcFrame = read_frame(movie)
srcGray = prep_frame(srcFrame, bbox)
srcPoints = find_features(srcGray)
trs = sum = [0,0,0]

i = 1
while True:
	ret, dstFrame = read_frame(movie)
	if ret != True:
		break
	dstGray = prep_frame(dstFrame, bbox)
	mtrx, dstPoints, err = track_frame(srcGray, dstGray, srcPoints)
	trs = matrix_to_transforms(mtrx)
	sum = [sum[0]-trs[0], sum[1]-trs[1], sum[2]-trs[2]]
	srcFrameW = cv2.warpAffine(dstFrame, transforms_to_matrix(sum), (1080,1920))
	# draw
	cv2.line(srcFrameW, (50,155), (640,155), (255,0,0), thickness = 4)
	cv2.line(srcFrameW, (50,30), (50,int(30+np.clip(np.mean(err)*10,a_min=-1000,a_max=1000))), (0,0,200), thickness = 4)
	try:
		if showPoints:
			for p in srcPoints:
				cv2.circle(srcGray, (p[0][0], p[0][1]), 2, (255,0,0), -1)
			for q in dstPoints:
				cv2.circle(srcGray, (q[0][0], q[0][1]), 2, (128,0,0), -1)
	except:
		pass
	# show
	#cv2.imshow("Tracking",  cv2.resize(srcFrameW, (1280,720)))          
	cv2.imshow("gray",  srcGray)
	cv2.imshow("Tracking",  cv2.resize(srcFrameW,(720,1280)))          
	outputMovie.write(srcFrameW) if writeOutput else None
	transforms.append(trs)
	trajectory.append(sum)
	srcGray = dstGray
	srcPoints = find_features(srcGray) if findFeatures else dstPoints 
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

