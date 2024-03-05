import cv2 as cv
import os

# Get the current working directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the target file
path = "../Test Image/Valid 1.jpg"

# Join the current directory and the relative path to get the absolute path
path = os.path.join(current_directory, path)


img = cv.imread(path)
cv.imshow("Pics" , img)
cv.waitKey()