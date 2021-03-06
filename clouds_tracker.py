#cv libs import
import cv2
import numpy as np
import os
from imutils.video import VideoStream
import imutils
import datetime
import random as rng
import time
from centroid_tracker import CentroidTracker

# read video file
vs = cv2.VideoCapture("sample_video_3.mp4")
ct = CentroidTracker()
firstFrame = None

# tracked clouds 
W = 500
H = 900


# blue range
lower_blue = np.array([0, 10, 96])
upper_blue = np.array([90, 200, 255])

# white range
lower_white = np.array([230, 230, 230])
upper_white = np.array([240, 240, 240])

# green range
lower_green = np.array([0, 100, 0])
upper_green = np.array([80, 255, 40])

# red range
lower_red = np.array([37, 0, 0])
upper_red = np.array([255,100, 3])

# yellow range
lower_yellow = np.array([79, 79, 0])
upper_yellow = np.array([255,255, 0])

while True:
    time.sleep(0.01)
    frame = vs.read()


    if frame is None:
        break
    clouds = []
    # Resize Frame
    frame = imutils.resize(frame[1], width=W)
    # cut frame
    frame = frame[60:frame.shape[0]-60, 50:frame.shape[1]-50]


    # Convert the frame to grayscale, blur it, and detect edges
    full_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # preparing the mask to overlay
    mask_blue = cv2.inRange(full_color, lower_blue, upper_blue)
    mask_white = cv2.inRange(full_color, lower_white, upper_white )
    mask_green = cv2.inRange(full_color, lower_green, upper_green )
    mask_red = cv2.inRange(full_color, lower_red, upper_red )
    mask_yellow = cv2.inRange(full_color, lower_yellow, upper_yellow )
    mask = mask_blue+mask_green+mask_yellow+mask_red
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask = mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # filter noise
    gray = cv2.GaussianBlur(gray, (27,27), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    thresh = cv2.threshold(gray, 0, 100, cv2.THRESH_BINARY)[1] # Tresh 
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Canny for edges detection 
    canny = cv2.Canny(thresh.copy(), 10, 20)
    # Border detection
    cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for index, c in enumerate(cnts):
        contour_area = cv2.contourArea(c)
        #rnd_color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # detect contour color
        # empty clouds
        if contour_area:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw contour and center of the shape on the image
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            #cv2.putText(frame, f"Nube {index}", (cX - 20, cY - 20),
            #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # draw contour and center of the shape on the image
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            # add cloud to deque
            clouds.append((cX, cY)*2)

    object = ct.update(clouds)
    # loop over the tracked objects
    for (objectID, centroid) in object.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            

    cv2.putText(frame, f"Clouds: {len(cnts)}", (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


    # draw countour
    cv2.imshow("Contornos", thresh)
    cv2.imshow("Color filter", result)
    cv2.imshow("Cloud motion detection", frame)
    # wait for keypress
    cv2.waitKey(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()




