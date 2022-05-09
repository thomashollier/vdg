import re, bpy

def reformatTrackData(trackObject, fileObject, offsetXY, IS_LANDSCAPE):
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
        if IS_LANDSCAPE:
            s = ("%s %s %s\n"% (v.frame,v.co[0]-xOffset,v.co[1]-yOffset))
        else:
            s = ("%s %s %s\n"% (v.frame,v.co[1]-xOffset,v.co[0]-yOffset))
        ret = fileObject.write(s)
    if IS_LANDSCAPE:
        return (v.co[0]-xOffset,v.co[1]-yOffset)
    else:
        return (v.co[1]-xOffset,v.co[0]-yOffset)
    

# 
trsTrack = re.compile('(^track[0-9]{2})')
trsMultiTrackPatt = re.compile('(track[0-9]{2})_([0-9]{2})')
perspTrackPatt = re.compile('(trackPersp[0-9]{2})')
#
patt = re.compile('(.mov)|(.MP4)|(.MOV)|(.mp4)')
#

for clip in bpy.data.movieclips:
    # movie orientation
    if clip.size[0]>clip.size[1]:
        IS_LANDSCAPE=True
        xIdx = 0
        yIdx = 1
    else:
        IS_LANDSCAPE=False
        xIdx = 1
        yIdx = 0


    # build dictionary of all track data
    # [trackname]['ff']
    #            ['lf']
    #            ['data'][frame]
    #                           vector x, y
    tracks = clip.tracking.tracks
    trackersDict = {}
    for track in tracks:
        trackersDict[track.name]={}
        trackersDict[track.name]['data']={}
        ff = 1000000
        lf = -1
        data = {}
        for marker in track.markers:
            if marker.frame < ff: ff = marker.frame
            if marker.frame > lf: lf = marker.frame
            data[marker.frame] = marker.co
        for k in data.keys():
            #if k-1 in data.keys() and k+1  in data.keys():
            trackersDict[track.name]['data'][k] = data[k]
        trackersDict[track.name]['ff'] = ff
        trackersDict[track.name]['lf'] = lf


    # Split them out in types of tracking data
    DO_PERSP=False
    DO_TRS=False
    DO_MULTI_TRS=False
    # process the types of trackers
    perspTrackers = [t for t in trackersDict.keys() if re.match(perspTrackPatt, t) != None]
    if len(perspTrackers)>3:
        print("Clip %s: found %s perspective trackers." % (clip.name,len(perspTrackers)))
        DO_PERSP=True
    elif len(perspTrackers)>0:
        print("Clip %s: need at least 4 perspective trackers. Only %s found." % (clip.name,len(perspTrackers)))
    else:
        print("Clip %s: found no persp trackers." % (clip.name))

    trsMultiTrackers = [t for t in trackersDict.keys() if re.match(trsMultiTrackPatt, t) != None]
    if len(trsMultiTrackers)>1:
        print("Clip %s: found %s trs multi trackers." % (clip.name,len(trsMultiTrackers)))
        DO_TRS=True
    elif len(trsMultiTrackers)==1:
        print("Clip %s: found only 1 trs multi tracker. Deosn't make sense." % (clip.name))
    else:
        print("Clip %s: found no multi trs trackers." % (clip.name))
    
    trsTrackers = [t for t in trackersDict.keys() if re.match(trsTrack, t) != None]
    if len(trsTrackers)>0:
        print("Clip %s: found %s trs trackers." % (clip.name,len(trsTrackers)))
        DO_TRS=True
    else:
        print("Clip %s: found no  trs trackers." % (clip.name))


    # Write out regular TRS tracking data (frameNum x y)
    if DO_TRS:
        for tracker in trsTrackers:
            filepath = "%s/%s_%s.crv"%(bpy.path.abspath('//'), re.sub(patt, "", clip.name), tracker)
            print("doing", filepath)
            f = open(filepath, 'w')
            for k,d in trackersDict[tracker]['data'].items():
                txt = "%s [[ %s, %s]]\n" % (k, d[xIdx], d[yIdx])
                ret = f.write(txt)
            f.close()

    # Write out regular TRS tracking data (frameNum x y)
    if DO_PERSP:
        ff = -1
        lf = 100000000
        for tracker in perspTrackers:
            if trackersDict[tracker]['ff'] > ff: ff = trackersDict[tracker]['ff']
            if trackersDict[tracker]['lf'] < lf: lf = trackersDict[tracker]['lf']
        print("------------", ff, lf)
        filepath = "%s/%s_%s.crv"%(bpy.path.abspath('//'), re.sub(patt, "", clip.name), "persp")
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
        f.close()
    



