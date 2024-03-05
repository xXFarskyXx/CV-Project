import cv2 as cv
from cv2 import aruco
import numpy as np

# dictionary to specify type of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# detect the marker
param_markers = aruco.DetectorParameters()

cap = cv.VideoCapture(1)

# Try setting a desired resolution
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

# Increase brightness
cap.set(cv.CAP_PROP_BRIGHTNESS, 120)

i = 0

while cap.isOpened():

    succes, img = cap.read()
    img_draw = img.copy()
    gray_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame , marker_dict , parameters= param_markers)
    print(len(marker_corners))
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(img_draw , [corners.astype(np.int32)] , True , (0, 255, 255) , 4 , cv.LINE_AA)

    k = cv.waitKey(5)

    if k == 27:
        break
    elif k == ord('s') and len(marker_corners) == 4: # wait for 's' key to save and exit
        cv.imwrite(f"Test Image/img{i}.png", img)
        print("image saved")
        i += 1

    cv.imshow("Detect" , img_draw)
    cv.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv.destroyAllWindows()