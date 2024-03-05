import cv2 as cv
from cv2 import aruco

#Call a dictionary of markers with 5x5 grid and 250 total unique markers
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

#Generate markers from id 0 to 3 with side pixel of size 500x500
for i in range(4):
    marker_img = aruco.generateImageMarker(marker_dict , i , 500)
    #Show Markers
    cv.imshow(f"Marker {i}" , marker_img)
    #Save Markers
    cv.imwrite(f"Marker_{i}.png" , marker_img)
    cv.waitKey(0)