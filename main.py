import cv2
import numpy as np
import utils

widthImg = 700
heightImg = 700

path = "./images/1.jpeg"
img = cv2.imread(path)

img = cv2.resize(img, (widthImg, heightImg))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

imageArray = ([img, imgGray, imgBlur, imgCanny])

imgStacked = utils.stackImages(imageArray, 0.5)

cv2.imshow("Stacked image", imgStacked)
cv2.waitKey(0)
