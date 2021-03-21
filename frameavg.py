#!/usr/bin/python

import cv2
import numpy as np
import pycubelut as cube
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lut', '--lut', default=None, type=str, dest='lut', help='path to lut file to apply')
parser.add_argument('-fo', '--frameOffset', default=0, type=int, dest='frameOffset', help='frame start offset')
parser.add_argument('-fr', '--frameRange', default=1000, type=int, dest='frameRange', help='number of frame to average')
parser.add_argument('-ad', '--add', default=False, type=bool, dest='add', help='do not devide result by number of frames')
parser.add_argument('-tk', '--token', default="avg", type=str, dest='token', help='file name token to add to input file name')
parser.add_argument('inputMovie', help='input movie')
args = parser.parse_args()


lut = args.lut
frameOffset = args.frameOffset
frameRange = args.frameRange
add = args.add
token= args.token
inputMovie = args.inputMovie


output = inputMovie.replace(".mov", "_%s.PNG" % token).replace(".MP4", "_%s.PNG" % token)
print(output)

cap = cv2.VideoCapture(inputMovie)
inputNumberOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
cap.set(cv2.CAP_PROP_POS_FRAMES,frameOffset)

print(lut)
if lut:
	lut = cube.CubeLUT("../luts/AC_A7S_709_.cube")

print("Input file: %s, %s frames" % (inputMovie, inputNumberOfFrames))
print("Output file: %s, average of %s frames" % (output, frameRange))
print("Starting frame: %s" % (frameOffset))

# take first frame of the video
n=1.0
ret,frame = cap.read()
buff=np.float32(frame)/255.

while True:
	ret,frame = cap.read()
	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break
	frameData = frame.astype(np.float32) / 255
	if lut:
		lut.transform_trilinear(frameData, in_place=True)
	buff = cv2.add(buff,frameData)
	n = n + 1.0
	print("frame %s" % n)
	if n == frameRange:
		break

if not add:
	buff = buff/n

buff = np.clip(buff * 255, 0, 255)
buff = buff.astype(np.uint8)
cv2.imwrite(output,buff)

cap.release()




