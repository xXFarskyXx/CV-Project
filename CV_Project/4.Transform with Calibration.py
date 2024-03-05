import cv2 as cv
from cv2 import aruco
import numpy as np


#Load in the calibration data
calib_data_path = "2.Camera Calibration\img\MultiMatrix.npz"
calib_data = np.load(calib_data_path)
#print(calib_data.files)

#Get camera matrix and distortion coefficient
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

#Change the marker size according to real life (centimeters)
MARKER_SIZE = 4.3

def get_corner(coordinate , top_corner):

    #Get max and min of x and y of image point
    x_max = coordinate[np.argmax(coordinate[: , 0])]
    x_min = coordinate[np.argmin(coordinate[: , 0])]
    y_max = coordinate[np.argmax(coordinate[: , 1])]
    y_min = coordinate[np.argmin(coordinate[: , 1])]

    #Case for top right corner on top
    if top_corner == "right":

        top_left = x_min
        bottom_right = x_max
        top_right = y_min
        bottom_left = y_max

        return [top_left , top_right , bottom_right , bottom_left]
    
    #Case for top left corner on top
    if top_corner == "left":
        top_left = y_min
        bottom_right = y_max
        top_right = x_max
        bottom_left = x_min

        return [top_left , top_right , bottom_right , bottom_left]

#Main process
def transform():
    # dictionary to specify type of the marker
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

    # Specifying a way to detect markers (Ex. adaptiveThreshold / cornerRefinement / polygonalApproxAccuracyRate) in this case default
    param_markers = aruco.DetectorParameters()

    #Load Image
    path = "3.Test Image\img2.png"
    img = cv.imread(path)

    #Undistort the picture and convert it to gray scale for detecting corners
    img = cv.undistort(img, cam_mat, dist_coef)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(gray , marker_dict , parameters= param_markers)

    #Additionionally we can find the distance between camera and the marker(3d point in space = Object point / 2d point = Image point)
    if marker_corners:
        #Rotation and Translation Matrix of each marker
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners , MARKER_SIZE , cam_mat , dist_coef)

    #Create a blank plane
    plane = np.int32([])

    #Get each corner and insert it into a plane(So we can plot or do the numpy operation on whole 2d array)
    for corners in marker_corners:
        corners = corners.astype(int)
        if plane.size == 0:
            plane = np.squeeze(corners)
        plane = np.vstack((plane , np.squeeze(corners)))

    #Verifying that there aren't any duplicate of the same point(Sometimes it happens)
    plane = np.unique(plane , axis = 0)

    #print(plane)

    #Get corners in float32(Perspective Transform Matrix need to be np.float32)
    corners = np.float32(get_corner(plane , "left"))

    
    #print(corners)

    # for corner in corners:
    #     cv.circle(img , tuple(np.int32(corner)) , 1 , (0 , 255 , 255) , 3)


    #Perspective Transform
    col , row , _ = img.shape
    trans = np.float32([[0 , 0] , [col , 0] , [col , row] , [0 , row]])
    matrix = cv.getPerspectiveTransform(corners, trans)
    result = cv.warpPerspective(img, matrix, (col, row))
    result = cv.resize(result , (500 , 500))

    #Plot the corners / Show the result / Save the file
    cv.polylines(img , corners.astype(int).reshape(-1 , 1 , 2) , True , (0, 255, 0) , 4 , cv.LINE_AA)
    cv.imshow("Pics", img)
    cv.imshow("result", result)
    cv.imwrite("result/result5.png" , result)
    cv.waitKey()

transform()