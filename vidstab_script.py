#!/usr/local/bin/python3

from vidstab.VidStab import VidStab
from vidstab import general_utils

import numpy as np
import os, argparse
from scipy.ndimage import gaussian_filter1d


def prepOffset(offset, template):
	try:
		offset = float(offset)
		#offsetXStatic = True
		offsetArray = np.full_like(template, offset)
	except:
		offset = str(offset)
		#offsetXStatic = False
		import sys
		sys.path.append("../vdg/python")
		import animcurves

		AC = animcurves.AnimCurve()
		AC.setCurveText(offset)
		AC.curveTextToCurveData()

		length = template.shape[0]
		offsetArray = np.empty(length)
		
		offsetArray = AC.fillArrayWithValues(np.arange(length))

	return(offsetArray)

def filterCurve(curve, filter, window):
	if gaussianFilter:
		print ("\tGaussian smoothing window %s" % window)
		return gaussian_filter1d(curve, window)
	else:
		print ("\tRolling Means smoothing window %s" % window)
		smoothed_trajectory = general_utils.bfill_rolling_mean(np.stack((curve, curve, curve), axis=1), window)
		return smoothed_trajectory[:,0]
		


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-sw', '--smoothWindow', default=240, type=int, dest='smoothWindow', help='number of frame to smooth')
parser.add_argument('-sx', '--XsmoothWindow', default=-1, type=int, dest='XsmoothWindow', help='number of frame to smooth for X stab')
parser.add_argument('-sy', '--YsmoothWindow', default=-1, type=int, dest='YsmoothWindow', help='number of frame to smooth for Y stab')
parser.add_argument('-sr', '--RsmoothWindow', default=-1, type=int, dest='RsmoothWindow', help='number of frame to smooth for R stab')
parser.add_argument('-gf', '--gaussianFilter', default=False, type=bool, dest='gaussianFilter', help='apply gaussian instead of rolling mean')
parser.add_argument('-gt', '--generateXforms', default=False, type=bool, dest='generateXforms', help='track movie and generate transforms before stabilizing')
parser.add_argument('-td', '--transformDir', default="transformCurves", type=str, dest='transformDir', help='directory for keeping the transform curves')
parser.add_argument('-ox', '--offsetX', default=0, type=str, dest='offsetX', help='offset X value to add to x value of transforms')
parser.add_argument('-oy', '--offsetY', default=0, type=str, dest='offsetY', help='offset Y value to add to y value of transforms')
parser.add_argument('-or', '--offsetR', default=0, type=str, dest='offsetR', help='offset R value to add to r value of transforms')
parser.add_argument('-of', '--offsets', default=[0,0,0], nargs=3, type=float, dest='offsets', help='offset values to add to x, y, y values of transforms')
parser.add_argument('-lf', '--lockframe', default=-1, type=int, dest='lockFrame', help='lock transforms to this frame')
parser.add_argument('-lx', '--lockframeX', default=-1, type=int, dest='lockFrameX', help='lock X transforms to this frame')
parser.add_argument('-ly', '--lockframeY', default=-1, type=int, dest='lockFrameY', help='lock Y transforms to this frame')
parser.add_argument('-lr', '--lockframeR', default=-1, type=int, dest='lockFrameR', help='lock R transforms to this frame')
parser.add_argument('-tk', '--token', default="smooth", type=str, dest='token', help='token added to processed output filename')
parser.add_argument('inputMovie', help='input movie')

args = parser.parse_args()
smoothWindow = args.smoothWindow
XsmoothWindow = args.XsmoothWindow
YsmoothWindow = args.YsmoothWindow
RsmoothWindow = args.RsmoothWindow
gaussianFilter = args.gaussianFilter
generateXforms = args.generateXforms
transformDir = args.transformDir
offsetX= args.offsetX
offsetY= args.offsetY
offsetR= args.offsetR
offsets= args.offsets
lockFrame = args.lockFrame
lockFrameX = args.lockFrameX
lockFrameY = args.lockFrameY
lockFrameR = args.lockFrameR
token = args.token
inputMovie = args.inputMovie

movie=inputMovie

base= movie.replace(".MOV","").replace(".MP4","").replace(".mov", "")
movie_smoothed="%s_%s.MP4"% (base, token)

print("----Stabilizing video---\ninput movie: %s\nsmooth window: %s \nsmooth windowX: %s\nsmooth windowY: %s\nsmooth windowR: %s" % ( movie, smoothWindow,XsmoothWindow, YsmoothWindow, RsmoothWindow, ))
print("gaussian Filter: %s" % gaussianFilter)

print("generate Xforms: %s\ntransformDir: %s\noffsets: %s \noffsetX: %s\noffsetY: %s\nlock frame: %s \nlock frame X: %s\nlock frame Y: %s\nlock frame r: %s\noutput movie: %s \n"% ( generateXforms, transformDir, offsets, offsetX, offsetY, lockFrame, lockFrameX, lockFrameY, lockFrameR, movie_smoothed))



stabilizer = VidStab()

####  Compute transforms
if generateXforms:
	print ("---- Generating transforms")
	stabilizer.gen_transforms(input_path=movie, smoothing_window=smoothWindow, show_progress=True)

	if os.path.exists(transformDir):
		if not os.path.isdir(transformDir):
			print("%s exists but is not a directory" % transformDir)
			exit(0)
	else:
		os.mkdir(transformDir)


	np.savetxt("%s/%s_transforms.txt" % (transformDir, base), stabilizer.transforms, fmt='%4.8f %4.8f %4.8f')
	np.savetxt("%s/%s_trajectory.txt" % (transformDir, base), stabilizer.trajectory, fmt='%4.8f %4.8f %4.8f')
	np.savetxt("%s/%s_smoothed_trajectory.txt" % (transformDir, base), stabilizer.smoothed_trajectory, fmt='%4.8f %4.8f %4.8f')
	np.savetxt("%s/%s_raw_transforms.txt" % (transformDir, base), stabilizer.raw_transforms, fmt='%4.8f %4.8f %4.8f')
####  or read pre-computed transforms and created smooth version
else:
	print ("---- Reading pre-computed transforms")
	stabilizer.trajectory = np.loadtxt("%s/%s_trajectory.txt" % (transformDir, base), delimiter=' ')
	stabilizer.raw_transforms = np.loadtxt("%s/%s_raw_transforms.txt" % (transformDir, base), delimiter=' ')


####   Smooth curves
stabilizer.smoothed_trajectory = stabilizer.trajectory.copy()

XsmoothWindow = smoothWindow if ( XsmoothWindow == -1 ) else XsmoothWindow
print ("---- Smoothing X transform accross %s frames with filter %s"  % (XsmoothWindow, "gaussianFilter" if gaussianFilter else "rolling mean filter"))
stabilizer.smoothed_trajectory[:,0] = filterCurve(stabilizer.trajectory[:,0], gaussianFilter, XsmoothWindow)

YsmoothWindow = smoothWindow if YsmoothWindow == -1 else YsmoothWindow
print ("---- Smoothing Y transform accross %s frames with filter %s"  % (YsmoothWindow, "gaussianFilter" if gaussianFilter else "rolling mean filter"))
stabilizer.smoothed_trajectory[:,1] = filterCurve(stabilizer.trajectory[:,1], gaussianFilter, YsmoothWindow)

RsmoothWindow = smoothWindow if RsmoothWindow == -1 else RsmoothWindow
print ("---- Smoothing R transform accross %s frames with filter %s"  % (RsmoothWindow, "gaussianFilter" if gaussianFilter else "rolling mean filter"))
stabilizer.smoothed_trajectory[:,2] = filterCurve(stabilizer.trajectory[:,2], gaussianFilter, RsmoothWindow)


####   Replace smoothed version with locked if specified
xyrLock = [-1,-1,-1]	
if lockFrame > 0:
	xyrLock = [lockFrame, lockFrame, lockFrame]
if lockFrameX > 0:
	xyrLock[0] = lockFrameX
if lockFrameY > 0:
	xyrLock[1] = lockFrameY
if lockFrameR > 0:
	xyrLock[2] = lockFrameR
for i,v in enumerate(xyrLock):
	if v > 0:
		print ("---- Locking transform index %s to frame %s" % (i, v))
		transformLock = stabilizer.trajectory[v]	
		stabilizer.smoothed_trajectory[:,i] = transformLock[i]
		
####   Apply offsets
if offsets != [0,0,0]:
	print ("---- Offsetting transforms by %s" % (offsets))
	stabilizer.smoothed_trajectory = np.add(stabilizer.smoothed_trajectory,np.full_like(stabilizer.smoothed_trajectory, [offsets]))
else:
	if offsetX != 0:
		print ("---- Offsetting X transform by %s" % (offsetX))
		stabilizer.smoothed_trajectory[:,0] = np.add(stabilizer.smoothed_trajectory[:,0], prepOffset(offsetX, stabilizer.smoothed_trajectory[:,0])) 
	if offsetY != 0:
		print ("---- Offsetting Y transform by %s" % (offsetY))
		stabilizer.smoothed_trajectory[:,1] = np.add(stabilizer.smoothed_trajectory[:,1], prepOffset(offsetY, stabilizer.smoothed_trajectory[:,1]))
	if offsetR != 0:
		print ("---- Offsetting R transform by %s" % (offsetR))
		stabilizer.smoothed_trajectory[:,2] = np.add(stabilizer.smoothed_trajectory[:,2], prepOffset(offsetR, stabilizer.smoothed_trajectory[:,2]))

print("---- Stabilization curve processed, applying transforms.\n")


stabilizer.transforms = stabilizer.raw_transforms + (stabilizer.smoothed_trajectory - stabilizer.trajectory)
stabilizer.apply_transforms(input_path=movie, output_path=movie_smoothed, output_fourcc="avc1")

exit(movie_smoothed)

