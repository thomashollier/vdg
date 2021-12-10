#!/usr/local/bin/python3

## #!/usr/bin/python

import cv2
import numpy as np
import pycubelut as cube
import argparse, sys, time


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lut', '--lut', default=None, type=str, dest='lut', help='path to lut file to apply')

parser.add_argument('-fs', '--frameStart', default=1, type=int, dest='startFrame', help='first frame of the stabilize')
parser.add_argument('-fe', '--frameEnd', default=1, type=int, dest='endFrame', help='first frame of the stabilize')

parser.add_argument('-fo', '--frameOffset', default=0, type=int, dest='frameOffset', help='frame start offset')
parser.add_argument('-fr', '--frameRange', default=1000, type=int, dest='frameRange', help='number of frame to average')
parser.add_argument('-ad', '--add', default=False, type=bool, dest='add', help='do not devide result by number of frames')
parser.add_argument('-b', '--brightness', default=1, type=float, dest='bright', help='multiply')
parser.add_argument('-g', '--gamma', default=1, type=float, dest='gamma', help='gamma')
parser.add_argument('-gb', '--gammaBefore', action='store_true', dest='gammaBefore', help='apply the gamma before adding to the stack')
parser.add_argument('-c', '--clahe', action='store_true', dest='clahe', help='clahe contrast. If on, default clip = 40, grid size = 8')
parser.add_argument('-cc', '--claheClip', default=40, type=float, dest='claheClip', help='clahe contrast clip')
parser.add_argument('-cs', '--claheGrid', default=8, type=int, dest='claheGrid', help='clahe contrast grid size')
parser.add_argument('-cb', '--claheBefore', action='store_true', dest='claheBefore', help='apply clahe filter to images before adding them')
parser.add_argument('-tk', '--token', default="avg", type=str, dest='token', help='file name token to add to input file name')
parser.add_argument('inputMovie', help='input movie')
args = parser.parse_args()



lut = args.lut
if args.startFrame != args.endFrame :
	frameOffset = args.startFrame
	frameRange = args.endFrame - args.startFrame
else:
	frameOffset = args.frameOffset
	frameRange = args.frameRange
add = args.add
bright = args.bright
gamma = args.gamma
gammaBefore = args.gammaBefore
clahe = args.clahe
claheClip = args.claheClip
claheGrid = args.claheGrid
claheBefore = args.claheBefore
token= args.token
inputMovie = args.inputMovie



import re 
patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')
output = re.sub(patt, '_%s.png' % token, inputMovie)



print("\n--- Parameters for frame averaging:")
print("inputMovie: %s" % inputMovie)
print("lut: %s" % lut)
print("frameOffset: %s" % frameOffset)
print("frameRange: %s" % frameRange)
print("add: %s" % add)
print("bright: %s" % bright)
print("gamma: %s" % gamma)
print("gammaBefore: %s" % gammaBefore)
print("clahe: %s" % clahe)
print("claheClip: %s" % claheClip)
print("claheSize: %s" % claheGrid)
print("claheBefore: %s" % claheBefore)
print("token: %s" % token)
print("output: %s\n" % output)




cap = cv2.VideoCapture(inputMovie)
inputNumberOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
cap.set(cv2.CAP_PROP_POS_FRAMES,frameOffset)

if lut:
	lut = cube.CubeLUT("../luts/AC_A7S_709_.cube")


# take first frame of the video
n=1.0
ret,frame = cap.read()
buff=np.float32(frame)/255.


startTime = time.perf_counter()

while True:
	ret,frame = cap.read()
	if clahe and claheBefore:
		r = frame[:,:,0]
		g = frame[:,:,1]
		b = frame[:,:,2]
		clahe = cv2.createCLAHE(clipLimit=claheClip, tileGridSize=(claheGrid,claheGrid))
		r = clahe.apply(r)
		g = clahe.apply(g)
		b = clahe.apply(b)
		frame = cv2.merge([r,g,b])

	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break
	frameData = frame.astype(np.float32) / 255
	if gamma != 1 and gammaBefore:
		frameData = np.power(frameData,1/gamma)
	if lut:
		lut.transform_trilinear(frameData, in_place=True)
	buff = buff + frameData
	n = n + 1.0

	percentDone = 100.0*float(n)/float(frameRange)
	elapsedTime = time.perf_counter()-startTime
	remainingTime = int(.99+elapsedTime/percentDone * (100-percentDone) )
	sys.stdout.write("\rFrame %4d of %s -- %.02f%% complete -- %s seconds remaining" % (n, frameRange, percentDone, remainingTime))
	sys.stdout.flush()


	if n == frameRange:
		break

if not add:
	buff = buff/n
	if bright != 1:
		buff *= bright
	if gamma != 1 and not gammaBefore:
		buff = pow(buff,1/gamma)
	if clahe and not claheBefore:
		print("doing the clahe")
		buff = buff * 65535
		buff = buff.astype(np.uint16) 
		r = buff[:,:,0]
		g = buff[:,:,1]
		b = buff[:,:,2]
		clahe = cv2.createCLAHE(clipLimit=claheClip, tileGridSize=(claheGrid,claheGrid))
		r = clahe.apply(r)
		g = clahe.apply(g)
		b = clahe.apply(b)
		buff = cv2.merge([r,g,b])	
		buff = buff.astype(np.float32)
		buff = buff/65535
		buff = np.clip(buff,0,1)
	
buff = np.clip(buff * 65535, 0, 65535)
buff = buff.astype(np.uint16)
cv2.imwrite(output,buff)

cap.release()

print("\n\nProcess complete:\n%s\n" % output)


