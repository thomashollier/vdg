import re, bpy

def reformatTrackData(trackObject, fileObject, offsetXY):
    # TO DO: check the logic of IS_LANDSCAPE when using 
    # already swapped X and Y offsets
    trackStartLocation = list(trackObject.markers.values())[0].co
    if offsetXY != (0,0):
        xOffset = trackStartLocation[0]-offsetXY[0]
        yOffset = trackStartLocation[1]-offsetXY[1]
    else:
        xOffset = 0
        yOffset = 0
    for k,v in trackObject.markers.items():
        s = ("%s %s %s\n"% (v.frame,v.co[1]-xOffset,v.co[0]-yOffset))
        ret = fileObject.write(s)
        return (v.co[1]-xOffset,v.co[0]-yOffset)
    

print("\n\n\n\t--------------------------------------------\n")
print("\t-----------   WRITE OUT TRACKERS  ----------")
print("\n\n\t--------------------------------------------\n")



# 
trsTrack = re.compile('(^track[0-9]{2})')
trsMultiTrackPatt = re.compile('(track[0-9]{2})_([0-9]{2})')
perspTrackPatt = re.compile('(trackPersp[0-9]{2})')
#
patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')
#

for clip in bpy.data.movieclips:
    xIdx = 0
    yIdx = 1

    print("\n--- Processing trackers for clip %s"% clip.name)
    tracks = clip.tracking.tracks
    trackersDict = {}
    for track in tracks:
        trackersDict[track.name]={}
        trackersDict[track.name]['data']={}
        ff = 1000000
        lf = -1
        data = {}
        for marker in track.markers:
            data[marker.frame] = marker.co
        for k in data.keys():
            if k-1 in data.keys() and k+1  in data.keys():
                trackersDict[track.name]['data'][k] = data[k]
        for k in trackersDict[track.name]['data'].keys():
            if k < ff: ff = k
            if k > lf: lf = k
        trackersDict[track.name]['ff'] = ff
        trackersDict[track.name]['lf'] = lf
        print("%s %s %s"% (track.name, ff, lf))


    # Split them out in types of tracking data
    DO_PERSP=False
    DO_TRS=False
    DO_MULTI_TRS=False
    # process the types of trackers
    print("")
    perspTrackers = [(t) for t in trackersDict.keys() if re.match(perspTrackPatt, t) != None]
    numPerspTracker = len(perspTrackers)
    if numPerspTracker>3 and numPerspTracker%4 == 0:
        perspTrackersDict = {k:[] for k in list(set([str(t)[-2] for t in perspTrackers]))}
        [perspTrackersDict[str(t)[-2]].append(t) for t  in perspTrackers]
        {k:sorted(v) for k, v in perspTrackersDict.items()}
        print("%s perspective trackers: %s" % (len(perspTrackersDict.keys()), str(perspTrackersDict)))

        DO_PERSP=True
    elif len(perspTrackers)>0:
        print("need perspective trackers to be multiples of 4 but %s were found." % (len(perspTrackers)))
    else:
        print("0 persp trackers")

    trsMultiTrackers = [t for t in trackersDict.keys() if re.match(trsMultiTrackPatt, t) != None]
    if len(trsMultiTrackers)>1:
        print("%s trs multi trackers" % (len(trsMultiTrackers)))
        DO_TRS=True
    elif len(trsMultiTrackers)==1:
        print("only 1 trs multi tracker. Deosn't make sense")
    else:
        print("0 multi trs trackers")
    
    trsTrackers = [t for t in trackersDict.keys() if re.match(trsTrack, t) != None]
    if len(trsTrackers)>0:
        print("%s trs trackers" % len(trsTrackers))
        DO_TRS=True
    else:
        print("0 trs trackers")


    # Write out regular TRS tracking data (frameNum x y)
    if DO_TRS:
        for tracker in trsTrackers:
            filepath = "%s/%s_%s.crv"%(bpy.path.abspath('//'), re.sub(patt, "", clip.name), tracker)
            filepath = filepath.replace("_sm","")
            print("Writing", filepath)
            f = open(filepath, 'w')
            for k,d in trackersDict[tracker]['data'].items():
                txt = "%s [[ %s, %s]]\n" % (k, d[xIdx], d[yIdx])
                ret = f.write(txt)
            f.close()
            print("Done")


    # Write out regular TRS tracking data (frameNum x y)
    if DO_PERSP:
        
        ff = -1
        lf = 100000000
        for tracker in perspTrackers:
            if trackersDict[tracker]['ff'] > ff: ff = trackersDict[tracker]['ff']
            if trackersDict[tracker]['lf'] < lf: lf = trackersDict[tracker]['lf']
        print("------------", ff, lf)
        filepath = "%s/%s_%s.crv"%(bpy.path.abspath('//'), re.sub(patt, "", clip.name), "persp")
        filepath = filepath.replace("_sm","")
        print("Writing", filepath)
        f = open(filepath, 'w')
        for frame in range(ff,lf):
            try:
                txt = "%s [ " % frame 
                for tracker in perspTrackers:
                    d = trackersDict[tracker]['data'][frame]
                    txt += "[%s, %s], " % (d[xIdx], d[yIdx])
                txt += " ]\n"
                ret = f.write(txt.replace(",  ]", " ]"))
            except:
                pass
        print("Done")
        f.close()
    



