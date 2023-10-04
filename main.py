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

  biggestContour = utils.reorderPoints(biggestContour)
  gradePoints = utils.reorderPoints(gradePoints)

  firstPoint = np.float32(biggestContour)
  secondPoint = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
  pointsMatrix = cv2.getPerspectiveTransform(firstPoint, secondPoint)
  imgWarpColorRed = cv2.warpPerspective(img, pointsMatrix, (widthImg, heightImg))

  firstGradePoint = np.float32(gradePoints)
  secondGradePoint = np.float32([[0,0], [325, 0], [0, 150], [325, 150]])
  gradePointsMatrix = cv2.getPerspectiveTransform(firstGradePoint, secondGradePoint)
  imgGrade = cv2.warpPerspective(img, gradePointsMatrix, (325, 150))

  # Apply threshold
  imgWarpGray = cv2.cvtColor(imgWarpColorRed, cv2.COLOR_BGR2GRAY)
  imgThresh = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]

imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny], [imgContours, imgBiggestContours, imgWarpColorRed, imgThresh])
imgStacked = utils.stackImages(imageArray, 0.5)

cv2.imshow("Stacked image", imgStacked)
cv2.waitKey(0)
