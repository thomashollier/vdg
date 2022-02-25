import re, bpy

def reformatTrackData(trackObject, fileObject, offsetXY, IS_LANDSCAPE):
    # check the logic of IS_LANDSCAPE when using 
    # already swapped X and Y offsets
    trackStartLocation = list(trackObject.markers.values())[0].co
    if offsetXY != (0,0):
        xOffset = trackStartLocation[0]-offsetXY[0]
        yOffset = trackStartLocation[1]-offsetXY[1]
    else:
        xOffset = 0
        yOffset = 0
    for k,v in trackObject.markers.items():
        if IS_LANDSCAPE:
            s = ("%s %s %s\n"% (v.frame,v.co[0]-xOffset,v.co[1]-yOffset))
        else:
            s = ("%s %s %s\n"% (v.frame,v.co[1]-xOffset,v.co[0]-yOffset))
        ret = fileObject.write(s)
    if IS_LANDSCAPE:
        return (v.co[0]-xOffset,v.co[1]-yOffset)
    else:
        return (v.co[1]-xOffset,v.co[0]-yOffset)
    

multiTrackPatt = re.compile('(track[0-9]{2})_([0-9]{2})')
patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')

for clip in bpy.data.movieclips:
    # dictionary of all the tracks for a clip
    if clip.size[0]>clip.size[1]:
        IS_LANDSCAPE=True
    else:
        IS_LANDSCAPE=False
    trackDict = {}
    for track in clip.tracking.tracks:
        match = re.match(multiTrackPatt, track.name)
        if match and len(match.groups()) == 2:
            if match.group(1) in trackDict.keys():
                trackDict[match.group(1)].append(track)
            else:
                trackDict[match.group(1)] = [track]
        else:
            trackDict[track.name] = track
    for k, v in trackDict.items():
        filepath = "%s/%s_%s.crv"%(bpy.path.abspath('//'), re.sub(patt, "", clip.name), k)
        print(filepath)
        f = open(filepath, 'w')
        offset = (0,0)
        if isinstance(v, list):
            for t in v:
                offset = reformatTrackData(t, f, offset, IS_LANDSCAPE)
        else:
            reformatTrackData(v, f, (0,0), IS_LANDSCAPE)
        f.close()



