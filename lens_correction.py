#!/usr/bin/python

import cv2, lensfunpy

db = lensfunpy.Database()


cam_maker = "Sony"
cam_model = "DSC-RX100M6"
lens_maker = "Sony"
lens_model = "DSC-RX100 VI & compatibles"

cam = db.find_cameras(cam_maker, cam_model)[0]
lens = db.find_lenses(cam, lens_maker, lens_model)[0]

print(cam)
print(lens)

focal_length = 9
aperture = 8
distance = 10

movie = cv2.VideoCapture("C0020.MP4")

ret, im = movie.read()
height, width = im.shape[0], im.shape[1]
mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
mod.initialize(focal_length, aperture, distance)

undist_coords = mod.apply_geometry_distortion()

showdist = True

while True:
	ret, im = movie.read()

	if showdist:
		im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
	else:
		im_undistorted = im

	cv2.imshow("foo", cv2.resize(im_undistorted,(1280,720)))


	k = cv2.waitKey(1) & 0xff
	#k = cv2.waitKey(1)
	if k == 27 : 
		cv2.imwrite("dist.png", im)
		cv2.imwrite("undist.png", im_undistorted)
		break
	elif k == 116:
		print(k)
		showdist = not showdist




