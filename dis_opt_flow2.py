#!/usr/bin/python

'''
example to show optical flow estimation using DISOpticalFlow

USAGE: dis_opt_flow.py [<video_source>]

Keys:
 1  - toggle HSV flow visualization
 2  - toggle glitch
 3  - toggle spatial propagation of flow vectors
 4  - toggle temporal propagation of flow vectors
ESC - exit'''


import numpy as np
import cv2


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res



def main():
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    inputMovie = sys.argv[1]
    outputMovie = inputMovie.replace(".MOV", "-flow.MOV")
    outputMovie = outputMovie.replace(".MP4", "-flow.MP4")
    print(inputMovie, outputMovie)


    cap = cv2.VideoCapture(inputMovie)
    ret, prev = cap.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    use_spatial_propagation = False
    use_temporal_propagation = False

    flobjDIS = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    #flobjFarneback = cv2.FarnebackOpticalFlow.create()
#    flobjVarRef = cv2.VariationalRefinement.create()
#    flobjTVL1 = cv2.DualTVL1OpticalFlow.create() # very slow
#    flobjDenseRLOF = cv2.optflow.createDenseRLOF() # needs sparse data
#    flobjPCA = cv2.optflow.createOptFlow_PCAFlow()
#    flobjSimple = cv2.optflow.createOptFlow_SimpleFlow()
#    flobjDeep = cv2.optflow.createOptFlow_DeepFlow()

    flow = None

    inst = flobjDIS
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if flow is not None and use_temporal_propagation:
            #warp previous flow to get an initial approximation for the current flow:
            flow = inst.calc(prevgray, gray, warp_flow(flow,flow))
        else:
            flow = inst.calc(prevgray, gray, None)
        prevgray = gray

        cv2.imshow('flow HSV', draw_hsv(flow))

        ch = cv2.waitKey(5)
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()


