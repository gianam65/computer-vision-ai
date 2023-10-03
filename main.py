import cv2
import numpy as np
import utils

widthImg = 700
heightImg = 700

path = "./images/1.jpeg"
img = cv2.imread(path)

img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# FIND THE CONTOURS OF IMAGE
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)

# FIND RECTANGLES
rectCon = utils.rectContour(contours) 
biggestContour = utils.getCornerPoints(rectCon[0])
gradePoints = utils.getCornerPoints(rectCon[1]) 

if(biggestContour.size != 0 and gradePoints.size != 0):
  cv2.drawContours(imgBiggestContours, biggestContour, -1, (0,255,0), 20)
  cv2.drawContours(imgBiggestContours, gradePoints, -1, (255,0,0), 20)

  utils.reorderPoints(biggestContour)

imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny], [imgContours, imgBiggestContours, imgBlank, imgBlank])
imgStacked = utils.stackImages(imageArray, 0.5)

cv2.imshow("Stacked image", imgStacked)
cv2.waitKey(0)
