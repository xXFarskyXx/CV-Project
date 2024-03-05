import cv2 as cv
from cv2 import aruco
import numpy as np

# dictionary to specify type of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# Specifying a way to detect markers (Ex. adaptiveThreshold / cornerRefinement / polygonalApproxAccuracyRate) in this case default
param_markers = aruco.DetectorParameters()

cap = cv.VideoCapture(1)

#Setting Resolution and Brightness of webcam
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv.CAP_PROP_BRIGHTNESS, 120)

i = 0

while cap.isOpened():
    #Get the current frame of the img
    succes, img = cap.read()
    img_draw = img.copy()
    
    #Create grayscale to detect markers
    gray_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame , marker_dict , parameters= param_markers)

    #If there is atleast 1 corner loop through it to make a polyline
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(img_draw , [corners.astype(np.int32)] , True , (0, 255, 255) , 4 , cv.LINE_AA)

    k = cv.waitKey(5)

    #If press esc then break
    if k == 27:
        break
    #If press "s" and there are 4 corners then the frame is savable
    elif k == ord('s') and len(marker_corners) == 4: # wait for 's' key to save and exit
        cv.imwrite(f"Test Image/img{i}.png", img)
        print("image saved")
        i += 1

    cv.imshow("Detect" , img_draw)
    #cv.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv.destroyAllWindows()