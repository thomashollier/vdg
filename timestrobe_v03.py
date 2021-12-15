#!/usr/local/bin/python3

## #!/usr/bin/python

import cv2
import numpy as np
import pycubelut as cube
import argparse, sys, re


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lut', '--lut', default=None, type=str, dest='lut', help='path to lut file to apply')

parser.add_argument('-fs', '--frameStart', default=1, type=int, dest='frameStart', help='first frame of the stabilize')
parser.add_argument('-fe', '--frameEnd', default=1, type=int, dest='frameEnd', help='first frame of the stabilize')
parser.add_argument('-m' , '--mode', default=0, type=int, dest='mode', help='0 = max (default), 1 = over')


parser.add_argument('-ad', '--add', default=False, type=bool, dest='add', help='do not devide result by number of frames')
parser.add_argument('-b', '--brightness', default=1, type=float, dest='bright', help='multiply')
parser.add_argument('-g', '--gamma', default=1, type=float, dest='gamma', help='gamma')
parser.add_argument('-c', '--clahe', action='store_true', dest='clahe', help='clahe contrast. If on, default clip = 40, grid size = 8')
parser.add_argument('-cc', '--claheClip', default=40, type=float, dest='claheClip', help='clahe contrast clip')
parser.add_argument('-cs', '--claheGrid', default=8, type=int, dest='claheGrid', help='clahe contrast grid size')
parser.add_argument('-cb', '--claheBefore', action='store_true', dest='claheBefore', help='apply clahe filter to images before adding them')
parser.add_argument('-tk', '--token', default="strobe", type=str, dest='token', help='file name token to add to input file name')
parser.add_argument('inputMovie', help='input movie')
args = parser.parse_args()



lut = args.lut
frameStart = args.frameStart
frameEnd = args.frameEnd
frameRange = (frameEnd-frameStart)
mode = args.mode
add = args.add
bright = args.bright
gamma = args.gamma
clahe = args.clahe
claheClip = args.claheClip
claheGrid = args.claheGrid
claheBefore = args.claheBefore
token= args.token
inputMovie = args.inputMovie

patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')
output = re.sub(patt, '_%s.png' % token, inputMovie)



print("inputMovie: %s" % inputMovie)
print("lut: %s" % lut)
print("frameStart: %s" % frameStart)
print("frameEnd: %s" % frameEnd)
print("frameRange: %s" % frameRange)
print("mode: %s" % mode)
print("add: %s" % add)
print("bright: %s" % bright)
print("gamma: %s" % gamma)
print("clahe: %s" % clahe)
print("claheClip: %s" % claheClip)
print("claheSize: %s" % claheGrid)
print("claheBefore: %s" % claheBefore)
print("token: %s" % token)
print("output: %s" % output)




if lut:
	lut = cube.CubeLUT("../luts/AC_A7S_709_.cube")

# take first frame of the video

cap = cv2.VideoCapture(inputMovie)
inputNumberOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
cap.set(cv2.CAP_PROP_POS_FRAMES,frameStart)

ret,frame = cap.read()
canvas=np.float32(frame)/255.
background = canvas

if mode == 1:
	print("Creating averaged background.")
	frameCurrent = frameStart
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
		if lut:
			lut.transform_trilinear(frameData, in_place=True)
		background = background + frameData
		frameCurrent = frameCurrent + 1
		sys.stdout.write("\rframe %4d of %s" % (frameCurrent-frameStart, frameRange))
		sys.stdout.flush()
		if frameCurrent == frameStart + 5:
			break
	background = background / 6


if mode == 1 or mode == 2:
	erosion_size = 3
	el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
					(2 * erosion_size + 1, 2 * erosion_size + 1),
					(erosion_size, erosion_size))

if mode == 2:
	#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 
	#fgbg = cv2.createBackgroundSubtractorMOG2() 
	#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
	fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()


cap.set(cv2.CAP_PROP_POS_FRAMES,frameStart)
frameCurrent = frameStart

while frameCurrent < frameEnd:
	ret,frame = cap.read()
	frameData = frame.astype(np.float32)/255

	if mode == 0:
		canvas  = np.maximum(canvas,frameData)
	elif mode == 1:
		a = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
		b = cv2.cvtColor(frameData, cv2.COLOR_BGR2GRAY)
		diff = b-a
		diff = abs(diff)
		diff = np.clip(diff-.05,0,1)
		diff = cv2.erode(diff,el)
		diff = cv2.dilate(diff,el)
		diff = diff * 20
		diff = pow(diff,2)
		diff = cv2.GaussianBlur(diff,(13,13),.6)
		diff = np.clip((diff-.2)*1.25, 0,1)
		diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
		canvas = canvas*(1-diff)+ frameData * diff
	elif mode == 2:
		mask = fgbg.apply(frame)
		diff = mask.astype(np.float32)/255
		diff = cv2.erode(diff,el)
		diff = cv2.GaussianBlur(diff,(5,5),.6)
		#diff = cv2.dilate(diff,el)
		mask = (diff*255).astype(np.uint8)
		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		maskData = mask.astype(np.float32)/255
		frameData = frame.astype(np.float32)/255
		canvas = canvas * (1-maskData) + frameData * maskData

	look = canvas 
	look = (np.clip(look,0,1)*255).astype(np.uint8)
	


	cv2.imshow("stab",  cv2.resize(look,(720,1280)))


	frameCurrent = frameCurrent + 1
	sys.stdout.write("\rframe %4d of %s" % (frameCurrent-frameStart, frameRange))
	sys.stdout.flush()

	k = cv2.waitKey(1) & 0xff
	if k == 27 : break

canvas = np.clip(canvas * 65535, 0, 65535)
canvas = canvas.astype(np.uint16)
cv2.imwrite(output,canvas)

cap.release()

print("\n frame %s complete" % output)


