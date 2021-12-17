#!/usr/local/bin/python3

## #!/usr/bin/python

import cv2
import numpy as np
import pycubelut as cube
import argparse, sys, time


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lut', '--lut', default=None, type=str, dest='lut', help='path to lut file to apply')

parser.add_argument('-p', '--prefix', default="basename", type=str, dest='prefix', help='prefix for output file name')
parser.add_argument('-fs', '--frameStart', default=-1, type=int, dest='frameStart', help='first frame of the stabilize')
parser.add_argument('-fe', '--frameEnd', default=-1, type=int, dest='frameEnd', help='first frame of the stabilize')
parser.add_argument('-fo', '--frameOffset', default=0, type=int, dest='frameOffset', help='frame start offset')
parser.add_argument('-fr', '--frameRange', default=1000, type=int, dest='frameRange', help='number of frame to average')
parser.add_argument('-ad', '--add', default=False, type=bool, dest='add', help='do not devide result by number of frames')
parser.add_argument('-b', '--brightness', default=1, type=float, dest='bright', help='multiply')

parser.add_argument('-softContrast', '--softContrast', default=1, type=float, dest='softContrast', help='softContrast')
parser.add_argument('-g', '--gamma', default=1, type=float, dest='gamma', help='gamma')
parser.add_argument('-gb', '--gammaBefore', action='store_true', dest='gammaBefore', help='apply the gamma before adding to the stack')
parser.add_argument('-c', '--clahe', action='store_true', dest='clahe', help='clahe contrast. If on, default clip = 40, grid size = 8')
parser.add_argument('-cc', '--claheClip', default=40, type=float, dest='claheClip', help='clahe contrast clip')
parser.add_argument('-cs', '--claheGrid', default=8, type=int, dest='claheGrid', help='clahe contrast grid size')
parser.add_argument('-cb', '--claheBefore', action='store_true', dest='claheBefore', help='apply clahe filter to images before adding them')
parser.add_argument('-tk', '--token', default="avg", type=str, dest='token', help='file name token to add to input file name')
parser.add_argument('inputMovie', help='input movie')
args = parser.parse_args()



prefix = args.prefix
lut = args.lut
if args.frameStart != args.frameEnd :
	frameOffset = args.frameStart
	frameRange = args.frameEnd - args.frameStart
else:
	frameOffset = args.frameOffset
	frameRange = args.frameRange
frameStart = args.frameStart
frameEnd = args.frameEnd
add = args.add
bright = args.bright
softContrast = args.softContrast
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
output = "%s_%s" % (prefix, re.sub(patt, '_%s.png' % token, inputMovie))



print("\n--- FRAME AVERAGE")
print("\n--- Parameters for frame averaging:")
print("inputMovie: %s" % inputMovie)
print("prefix: %s" % prefix)
print("lut: %s" % lut)
print("frameOffset: %s" % frameOffset)
print("frameRange: %s" % frameRange)
print("frameStart: %s" % frameStart)
print("frameEnd: %s" % frameEnd)
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


####
#### define color correction luts
####
def softContrastCalc(x,k):
	x1 = .5*pow(2*x,k)
	x2 = .5*pow(2*(1-x),k)
	x3 = x1 if x < .5 else x2 
	return x3 if x < .5 else (1-x3)

def makeContrastLUT():
	identity = np.arange(256, dtype = np.dtype('uint8'))
	fidentity = identity.astype(np.float32)/255
	for i,v in enumerate(fidentity):
		fidentity[i]=softContrastCalc(v,softContrast)*255
	identity = fidentity.astype(np.uint8)	
	clut = np.dstack((identity, identity, identity))
	return clut

def makeXformLUT():
	# initialize cube LUT file
	lutXform = cube.CubeLUT(lut)

	# create LUT image
	identity = np.arange(256, dtype = np.dtype('uint8'))
	identity = np.dstack((identity, identity, identity))

	# convert to float and apply transform
	fidentity = identity.astype(np.float32)/255
	lutXform.transform_trilinear(fidentity, in_place=True)
	fidentity = np.clip(fidentity, 0, 1)

	# back to 8 bit and output
	clut = (fidentity*255).astype(np.uint8)	
	clut = np.clip(clut,0,255)
	return clut


if softContrast != 1:
	clut = makeContrastLUT()
if lut:
	xformLut = makeXformLUT()

#######
# work out in out frames
######

cap = cv2.VideoCapture(inputMovie)
inputNumberOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 

if frameStart == -1 and frameEnd == -1:
	print("Frame start and end not specified. Using all frames in clip.\n") 
	frameStart = 1
	frameEnd = inputNumberOfFrames - 1
frameRange = frameEnd - frameStart
frameCurrent = frameStart

# take first frame of the video
cap.set(cv2.CAP_PROP_POS_FRAMES,frameCurrent)
ret,frame = cap.read()
buffer=np.float32(frame)/255.


startTime = time.perf_counter()

while frameCurrent < frameEnd:
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
	if softContrast != 1:
		frameData = cv2.LUT((frameData*255).astype(np.uint8), clut).astype(np.float32)/255.0 	
	if lut:
		frameData = cv2.LUT((frameData*255).astype(np.uint8), xformLut).astype(np.float32)/255
		frameData = np.clip(frameData,0,1)
	buffer = buffer + frameData
	frameCurrent = frameCurrent + 1

	percentDone = 100.0*float(frameCurrent-frameStart)/float(frameRange)
	elapsedTime = time.perf_counter()-startTime
	remainingTime = int(.99+elapsedTime/percentDone * (100-percentDone) )
	remainingTime = "%02d:%02d" % (int(remainingTime/60), int(remainingTime%60))
	sys.stdout.write("\rFrame %4d of %s -- %.02f%% complete -- %s seconds remaining" % (frameCurrent-frameStart, frameRange, percentDone, remainingTime))
	sys.stdout.flush()


if not add:
	buffer = buffer/(frameRange)
	#buffer = cv2.LUT((buffer*255).astype(np.uint8), clut).astype(np.float32)/255.0
	if bright != 1:
		buffer *= bright
	if gamma != 1 and not gammaBefore:
		buffer = pow(buffer,1/gamma)
	if clahe and not claheBefore:
		print("doing the clahe")
		buffer = buffer * 65535
		buffer = buffer.astype(np.uint16) 
		r = buffer[:,:,0]
		g = buffer[:,:,1]
		b = buffer[:,:,2]
		clahe = cv2.createCLAHE(clipLimit=claheClip, tileGridSize=(claheGrid,claheGrid))
		r = clahe.apply(r)
		g = clahe.apply(g)
		b = clahe.apply(b)
		buffer = cv2.merge([r,g,b])	
		buffer = buffer.astype(np.float32)
		buffer = buffer/65535
		buffer = np.clip(buffer,0,1)
	
buffer = np.clip(buffer * 65535, 0, 65535)
buffer = buffer.astype(np.uint16)
cv2.imwrite(output,buffer)

cap.release()

print("\n\nProcess complete:\n%s\n" % output)


