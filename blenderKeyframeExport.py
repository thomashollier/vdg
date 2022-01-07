import re, bpy

patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')
p2 = re.compile('(track[0-9]{2})_([0-9]{2})')
testTrackName = 'track02'
m = re.match(p2, testTrackName)
if m and len(m.groups()) == 2:
	n = m.group(1)
	nb = m.group(2)
	print("name:", n, "number:", nb)
else:
	print(testTrackName)


print("hello")

p2 = re.compile('(track[0-9]{2})_([0-9]{2})')

for clip in bpy.data.movieclips:
	# dictionary of all the tracks for a clip
	trackDict = {}
	for track in clip.tracking.tracks:
		match = re.match(p2, track.name)
		if match and len(match.groups()) == 2:
			if match.group(1) in trackDict.keys():
				trackDict[match.group(1)].append(track.name)
			else:
				trackDict[match.group(1)] = [track.name]
		else:
			trackDict[track.name] = track.name
			filepath = clip.filepath
			filename = re.sub(patt, '_%s.crv'%track.name, filepath)
			filename = "/Users/thomas/Documents/slitscans/20211227_parrishClouds/%s" % filename
			f = open(filename, 'w')
			for k,v in track.markers.items():
				s = ("%s %s %s\n"% (v.frame,v.co[0],v.co[1]))
				ret = f.write(s)
			f.close()
	print("%s:\n\t%s\n" % (clip.name, trackDict))





'''

# Below was a technique that required tracks to be 
# baked on null object animation
# not a good way...

action = bpy.data.actions["track4741_01Action"]

xCurves=action.fcurves[0]
yCurves=action.fcurves[2]

xKeys = xCurves.keyframe_points
yKeys = yCurves.keyframe_points

f = open('/Users/thomas/Documents/slitscans/20211127_newyork1/track4741_01', 'w')

for xk, yk in zip(xKeys, yKeys):
	s = "%s %s %s\n" % (int(xk.co[0]), xk.co[1], yk.co[1])
	f.write(s)

f.close()
'''

