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
    else:
        IS_LANDSCAPE=False


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
        for marker in track.markers:
            if marker.frame < ff: ff = marker.frame
            if marker.frame > lf: lf = marker.frame
            trackersDict[track.name]['data'][marker.frame] = marker.co
        trackersDict[track.name]['ff'] = ff
        trackersDict[track.name]['lf'] = lf


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

    if DO_TRS:
        for tracker in trsTrackers:
            filepath = "%s/%s_%s.crv"%(bpy.path.abspath('//'), re.sub(patt, "", clip.name), tracker)
            print("doing", filepath)
            f = open(filepath, 'w')
            for k,d in trackersDict[tracker]['data'].items():
                txt = "%s [[ %s, %s]]\n" % (k, d[0], d[1])
                ret = f.write(txt)
            f.close()
    
    if False:
        # dictionary of all the tracks for a clip
        trackDict = {}
        for track in set(trsMultiTrackers+trsTrackers):
            match = re.match(trsMultiTrackPatt, track.name)
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
    if DO_PERSP:
        filepath = "%s/%s_%s.crv"%(bpy.path.abspath('//'), re.sub(patt, "", clip.name), "persp")
        f = open(filepath, 'w')
        for i, m in enumerate(perspTrackers[0].markers):
            try:
                thisFrame = m.frame
                txt = '%s ['% m.frame
                for t in perspTrackers:
                    txt += '[%s, %s], ' % ((clip.size[0]*(t.markers[thisFrame].co[0])), clip.size[1]*(1-t.markers[thisFrame].co[1]))
                txt = txt.rstrip().rstrip(',')
                txt += ']\n'
                ret = f.write(txt)
            except:
                print("problem at frame %s"%thisFrame)
        f.close()
        print(clip.size[0], clip.size[1])



