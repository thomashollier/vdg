#!/usr/local/bin/python3

import subprocess
import argparse

parser = argparse.ArgumentParser(
	description= '''
This takes an image with an arbitrary aspect ratio and figures out the thickness of the border
# necessary to fit that image into an arbitrarily sized poster while preserving the image's aspect ratio
# AKA: I'd like to center a randomly sized image into a standard (or also randomly) sized frame
# AKA: I want to put this image into this frame but I want an even border and I don't want to crop my image
''')

parser.add_argument('-dw', '--destWidth', default=-1, type=int, dest='destWidth', required=True, help='Width of the destination page in inches')  
parser.add_argument('-dh', '--destHeight', default=-1, type=int, dest='destHeight', required=True, help='Height of the destination page in inches')  
parser.add_argument('-dpi', '--dpi', default=300, type=int, dest='dpi', help='dpi of the destination image')  
parser.add_argument('-tk', '--token', default="print", type=str, dest='token', help='Output string token added to input file name')  
parser.add_argument('-dt', '--doTrim', default=False, type=bool, dest='doTrim', help='Trim the existing white borders before centering')  
parser.add_argument('-bc', '--borderColor', default="white", type=str, dest='borderColor', help='Color of the border')  

parser.add_argument('sourceImage')
args = parser.parse_args()

sourceImage = args.sourceImage
destWidth = args.destWidth
destHeight = args.destHeight
token= args.token
borderColor = args.borderColor
dpi = args.dpi
doTrim = args.doTrim

print("\n--- Fit image %s\ninto %s x %s page at %s dpi\n"% (sourceImage, destWidth, destHeight, dpi))


if doTrim:
	tmpImage = sourceImage.replace(".png", "_trimmed.png")
	trimCmd = "magick %s -trim -fuzz 2%% %s" %  (sourceImage, tmpImage)
	print("--- Trimming image: %s\n%s\n" % (sourceImage, trimCmd))
	trimRes = subprocess.run(trimCmd.split(), text=True, capture_output=True)
	sourceImage = tmpImage


#################################
### get surce image resolution
###

print("--- Calculate sizes")
cmd = "mediainfo --Inform=Image;%%Width%%:%%Height%%  %s" % sourceImage

res = subprocess.run(cmd.split(), text=True, capture_output=True)
imageWidth = float(res.stdout.split(':')[0])
imageHeight = float(res.stdout.split(':')[1])
print("Image Resolution:",imageWidth, "x", imageHeight)

imageAR = imageHeight/imageWidth
print("Image AR:", imageAR)

borderSize= .5 * ((destHeight/imageAR-destWidth)*imageAR)/(-imageAR + 1)

print( "Border Size:", borderSize)
print("Confirm AR in poster with border thickness: %s\n" % float((destHeight-(borderSize*2))/(destWidth-(borderSize*2))))

#################################
### calculate final resolutions and comp new image
###

imageSizeX = destWidth - (borderSize*2)
imageSizeY = destHeight - (borderSize*2)
 
finalImageX = round(imageSizeX * dpi)
finalImageY = round(imageSizeY * dpi)
finalPosterX = destWidth * dpi
finalPosterY = destHeight * dpi
finalBorderSize = round(borderSize * dpi)


#################################
### Compositing new image
###
cmd = "magick %s -resize %sx%s -bordercolor %s -border %s -density %s -units PixelsPerInch %s" % (sourceImage, finalImageX, finalImageY, borderColor, finalBorderSize, dpi, sourceImage.replace(".png", "_%s.png"%token))

print("--- Comping image\n%s"% cmd) 

res = subprocess.run(cmd.split(), text=True, capture_output=True)
