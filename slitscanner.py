#!/usr/bin/python

import cv2
import numpy as np
import pycubelut as cube
import argparse


parser = argparse.ArgumentParser(description='Make some slitscans.')
parser.add_argument('-ns', '--numberOfScans', default=0, type=int, dest='numberOfScans', help='how many scans from the start frame')
parser.add_argument('-lf', '--lutFile', default="", type=str, dest='lutFile', help='path to a LUT file to apply')
parser.add_argument('-mv', '--multVal', default="1", type=float, dest='multVal', help='multiply values')
parser.add_argument('-ro', '--rotate', default=False, type=bool, dest='rotate', help='transpose the frame')
parser.add_argument('-ff', '--flipflop', default="00", type=str, dest='flipflop', help='vertical horizontal boolean')
parser.add_argument('-sf', '--startFrame', default=1, type=int, dest='startFrame', help='frame number to start on')
parser.add_argument('-si', '--scanlineIndex', default=540, type=int, dest='scanlineIndex', help='scanline to use')
parser.add_argument('-ow', '--outputWidth', default=8192, type=int, dest='outputWidth', help='number of samples in the final scan')
parser.add_argument('inputMovie', help='input movie')
parser.add_argument('outputImage', help='outputImages')

args = parser.parse_args()

_inputMovie = args.inputMovie
_outputImage = args.outputImage
_outputWidth = args.outputWidth
_scanlineIndex = args.scanlineIndex
_rotate = args.rotate
_startFrame = args.startFrame
_numberOfScans = args.numberOfScans
_flipflop = args.flipflop
_lutFile = args.lutFile
_multVal = args.multVal
_applyLut = True if _lutFile != '' else False

movie = cv2.VideoCapture(_inputMovie)

movieFrameRange = int(movie.get(cv2.CAP_PROP_FRAME_COUNT))
movieHeight = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
movieWidth = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))

if _rotate:
	movieHeight = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
	movieWidth = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
	

print("\n----   PARAMETERS:")
print("input movie:\t\t%s" % _inputMovie)
print("number of frames:\t%s" % movieFrameRange)
print("movie width:\t\t%s" % movieWidth)
print("movie height:\t\t%s" % movieHeight)
print("rotate:\t\t\t%s" % _rotate)
print("output image:\t\t%s" % _outputImage)
print("output width:\t\t%s" % _outputWidth)
print("start frame\t\t%s" % _startFrame)
print("scanline index:\t\t%s" % _scanlineIndex)
print("number of scans\t\t%s" % _numberOfScans)
print("apply lut:\t\t%s" % (_lutFile if _applyLut else False))
print("multiply value\t\t%s" % _multVal)
print("-----------------\n")


if _numberOfScans == 0:
	endFrame = movieFrameRange
	_numberOfScans = ((movieFrameRange - _startFrame)/_outputWidth)
else:
	endFrame = _startFrame + _numberOfScans * _outputWidth

currentFrame =_startFrame
scanline = 0
scan = 1

movie.set(cv2.CAP_PROP_POS_FRAMES,_startFrame)

while currentFrame < endFrame:
	if scanline == 0:
		dst = np.zeros((_outputWidth,movieWidth,3), np.uint8)
		print("scanning to image: %s.%04d.png of %.2f" % ( _outputImage, scan, _numberOfScans))
	
	ret, im = movie.read()	

	### pre-process the frame
	if _rotate:
		im = cv2.transpose(im)


#	ratio = scanline/_outputWidth
#	multer = (np.cos(2*np.pi*ratio)*.5+.5)
#	M = np.float32([[1,0,np.sin(scanline*.02)*100*multer],[0,1,np.cos(scanline*0.02)*100*multer]])
#	im = cv2.warpAffine(im,M,(movieWidth,movieHeight))



	sliver = im[_scanlineIndex:_scanlineIndex+1,0:movieWidth]
	dst[scanline:scanline+1,0:movieWidth] = sliver

	currentFrame = currentFrame + 1
	scanline = scanline + 1

	if scanline == _outputWidth or currentFrame == endFrame:
		if _applyLut or _multVal != 0:
			im = dst.astype(np.float32) / 255
			if _applyLut:
				lut = cube.CubeLUT(_lutFile)
				lut.transform_trilinear(im, in_place=True)
			if _multVal != 1:
				im = np.power(im, 2.2)
				im = np.multiply(im, _multVal)
				im = np.power(im,0.4545)
			dst = np.clip(im * 255, 0, 255).astype(np.uint8)
		if _flipflop != "00":
			if _flipflop == "10":
				dst = cv2.flip(dst, 0)
			if _flipflop == "01":
				dst = cv2.flip(dst, 1)
			if _flipflop == "11":
				print("both")
				dst = cv2.flip(dst, -1)

		cv2.imwrite("%s.%04d.png" % (_outputImage, scan), cv2.transpose(dst))
		scan = scan + 1
		scanline = 0





