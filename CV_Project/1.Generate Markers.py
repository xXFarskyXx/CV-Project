import cv2 as cv
from cv2 import aruco

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

for i in range(4):
    marker_img = aruco.generateImageMarker(marker_dict , i , 500)
    cv.imshow(f"Marker {i}" , marker_img)
    cv.imwrite(f"Marker_{i}.png" , marker_img)
    cv.waitKey(0)
