#!/usr/bin/python

from vidstab.VidStab import VidStab
from vidstab import general_utils
import numpy as np

import argparse

def prepOffset(offset, template):
	try:
		offset = float(offset)
		#offsetXStatic = True
		offsetArray = np.full_like(template, offset)
	except:
		offset = str(offset)
		#offsetXStatic = False
		import sys
		sys.path.append("/mnt/d/art_Slitscans_IP/bin/python")
		import animcurves

		AC = animcurves.AnimCurve()
		AC.setCurveText(offset)
		AC.curveTextToCurveData()

		length = template.shape[0]
		offsetArray = np.empty(length)
		
		offsetArray = AC.fillArrayWithValues(np.arange(length))

	return(offsetArray)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-sw', '--smoothWindow', default=240, type=int, dest='smoothWindow', help='number of frame to smooth')
parser.add_argument('-sx', '--XsmoothWindow', default=-1, type=int, dest='XsmoothWindow', help='number of frame to smooth for X stab')
parser.add_argument('-sy', '--YsmoothWindow', default=-1, type=int, dest='YsmoothWindow', help='number of frame to smooth for Y stab')
parser.add_argument('-sr', '--RsmoothWindow', default=-1, type=int, dest='RsmoothWindow', help='number of frame to smooth for R stab')
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

base= movie.replace(".MOV","").replace(".MP4","")
movie_smoothed="%s_%s.MP4"% (base, token)

print("----Stabilizing video---\ninput movie: %s\nsmooth window: %s \nsmooth windowX: %s\nsmooth windowY: %s\nsmooth windowR: %s\ngenerate Xforms: %s\ntransformDir: %s\noffsets: %s \noffsetX: %s\noffsetY: %s\nlock frame: %s \nlock frame X: %s\nlock frame Y: %s\nlock frame r: %s\noutput movie: %s \n"% ( movie, smoothWindow,XsmoothWindow, YsmoothWindow, RsmoothWindow, generateXforms, transformDir, offsets, offsetX, offsetY, lockFrame, lockFrameX, lockFrameY, lockFrameR, movie_smoothed))


#foo = np.loadtxt("%s_trajectory.txt" % base, delimiter=' ')
#newfoo = prepOffset(offsetX, foo[:,1])
#print(type(foo))
# exit()


stabilizer = VidStab()


####  Compute transforms
if generateXforms:
	print ("---- Generating transforms")
	stabilizer.gen_transforms(input_path=movie, smoothing_window=smoothWindow, show_progress=True)

	np.savetxt("%s/%s_transforms.txt" % (transformDir, base), stabilizer.transforms, fmt='%4.8f %4.8f %4.8f')
	np.savetxt("%s/%s_trajectory.txt" % (transformDir, base), stabilizer.trajectory, fmt='%4.8f %4.8f %4.8f')
	np.savetxt("%s/%s_smoothed_trajectory.txt" % (transformDir, base), stabilizer.smoothed_trajectory, fmt='%4.8f %4.8f %4.8f')
	np.savetxt("%s/%s_raw_transforms.txt" % (transformDir, base), stabilizer.raw_transforms, fmt='%4.8f %4.8f %4.8f')
####  or read pre-computed transforms and created smooth version
else:
	print ("---- Reading pre-computed transforms")
	stabilizer.trajectory = np.loadtxt("%s/%s_trajectory.txt" % (transformDir, base), delimiter=' ')
	stabilizer.raw_transforms = np.loadtxt("%s/%s_raw_transforms.txt" % (transformDir, base), delimiter=' ')
	stabilizer.smoothed_trajectory = general_utils.bfill_rolling_mean(stabilizer.trajectory, n=smoothWindow)

####   Replace smoothing window for any channels that have specific values
if XsmoothWindow > 0:
	print ("---- Smoothing X transform accross %s frames" % XsmoothWindow)
	smoothed_trajectoryX = general_utils.bfill_rolling_mean(stabilizer.trajectory, n=XsmoothWindow)
	stabilizer.smoothed_trajectory[:,0] = smoothed_trajectoryX[:,0]
if YsmoothWindow > 0:
	print ("---- Smoothing Y transform accross %s frames" % YsmoothWindow)
	smoothed_trajectoryY = general_utils.bfill_rolling_mean(stabilizer.trajectory, n=YsmoothWindow)
	stabilizer.smoothed_trajectory[:,1] = smoothed_trajectoryY[:,1]
if RsmoothWindow > 0:
	print ("---- Smoothing R transform accross %s frames" % RsmoothWindow)
	smoothed_trajectoryR = general_utils.bfill_rolling_mean(stabilizer.trajectory, n=RsmoothWindow)
	stabilizer.smoothed_trajectory[:,2] = smoothed_trajectoryR[:,2]


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
stabilizer.apply_transforms(input_path=movie, output_path=movie_smoothed)

