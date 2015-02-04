import numpy as np
import cv2
import sys

def processFrame(cap, kernel, fgbg, framecount):
    k = cv2.waitKey(30)
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    fgmask_copy = fgmask.copy()

    contours,hierarchy = cv2.findContours(fgmask_copy, 0, 2)

    frame_area = 0
    for contour in contours:

        #filter small boxes
        area = cv2.contourArea(contour)
        frame_area = frame_area + area

    #draw framecount
    cv2.putText(frame, 'frame: %s' % framecount, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    #draw area
    cv2.putText(frame, 'area: %s' % frame_area, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv2.imshow('threshold',fgmask)
    cv2.imshow('frame',frame)

cap = cv2.VideoCapture(str(sys.argv[1]))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
fgbg = cv2.BackgroundSubtractorMOG(1,5,0.5,1)

framecount = 0

k = 32
while(1):
    if k == 32:
        processFrame(cap, kernel, fgbg, framecount)
        framecount = framecount + 1
    elif k == 27:
        break
    k = cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()





