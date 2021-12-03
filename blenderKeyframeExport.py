prefix='11222021173801'
movie='%s.mov' % prefix
track='%s_track_01' % prefix
filepath='/Users/thomas/Documents/slitscans/20211122_pier/%s_track_01.crv' % prefix

# movie clip that the track is done to
clip = bpy.data.movieclips[movie]
# track name in blender
track = clip.tracking.tracks[track]
# markers are collections of markers which contain the per-frame info
markers=track.markers

f = open(filepath, 'w')
for k,v in markers.items():
	s = ("%s %s %s\n"% (v.frame,v.co[0],v.co[1]))
	f.write(s)

f.close()
print("%s done")





'
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

