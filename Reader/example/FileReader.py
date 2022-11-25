import cv2
import numpy as np
import sys

filename = sys.argv[1]
print('Starting to load video file %s' % filename)
cap = cv2.VideoCapture(2)

if (cap.isOpened() == False):
    print("Error opening video stream of file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break;
    else:
        break

cap.release()

cv2.destroyAllWindows()
