import cv2 as cv
from cv2 import aruco
import numpy as np

# dictionary to specify type of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# detect the marker
param_markers = aruco.DetectorParameters()

#Camera
cap = cv.VideoCapture(0)

#Video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame , marker_dict , parameters= param_markers)
    #"Corners: " , marker_corners , "ID: " , marker_IDs , "Reject: " , reject
    print("Corners: " , marker_corners)

    # Corners
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(frame , [corners.astype(np.int32)] , True , (0, 255, 255) , 4 , cv.LINE_AA)
            corners = corners.reshape(4 , 2)
            corners = corners.astype(int)
            top_right = corners[0]
            top_left = corners[1]
            bottom_right = corners[2]
            bottom_left = corners[3]
            cv.putText(frame , f"id: {ids[0]}", top_left, cv.FONT_HERSHEY_COMPLEX , 1.3 , (200, 100, 0) , 2)
            # print(ids, "  ", corners)
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()