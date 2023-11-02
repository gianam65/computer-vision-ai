import cv2
import numpy as np
import utils

widthImg = 700
heightImg = 700

questions = 5
choices = 5

answers = [1, 2, 0, 1, 4]
webcamFeed = True
cameraNumber = 1
path = "./images/1.jpeg"

cap = cv2.VideoCapture(cameraNumber)
cap.set(10, 150)

while True:
    if webcamFeed: success, img = cap.read()
    else: img = cv2.imread(path)

    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    try:
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
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
            boxes = utils.splitBoxes(imgThresh)

            questionsAndChoices = np.zeros((questions, choices))
            columnCount = 0
            rowCount = 0

            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                questionsAndChoices[rowCount][columnCount] = totalPixels
                columnCount += 1
                if columnCount == choices: rowCount += 1; columnCount = 0

            currentIndexes = []
            for x in range (0, questions):
                currentRow = questionsAndChoices[x]
                selectedChoice = np.where(currentRow == np.amax(currentRow))
                currentIndexes.append(selectedChoice[0][0])

            grading = []
            for i in range(0, questions):
                if answers[i] == currentIndexes[i]:
                    grading.append(1)
                else: 
                    grading.append(0)
            
            score = (sum(grading) / questions) * 100

            imgResult = imgWarpColorRed.copy()
            imgResult = utils.showAnswers(imgWarpColorRed, currentIndexes, grading, answers, questions, choices)
            imRawDrawing = np.zeros_like(imgWarpColorRed)
            imRawDrawing = utils.showAnswers(imRawDrawing, currentIndexes, grading, answers, questions, choices)
            invMatrix = cv2.getPerspectiveTransform(secondPoint, firstPoint)
            imgInvWarp = cv2.warpPerspective(imRawDrawing, invMatrix, (widthImg, heightImg))
            
            imgRawGrade = np.zeros_like(imgGrade)
            cv2.putText(imgRawGrade, str(int(score)) + "%", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
            invMatrixGrade = cv2.getPerspectiveTransform(secondGradePoint, firstGradePoint)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixGrade, (widthImg, heightImg))
                
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)

            imgBlank = np.zeros_like(img)
            imageArray = ([img, imgGray, imgBlur, imgCanny], 
                        [imgContours, imgBiggestContours, imgWarpColorRed, imgThresh], 
                        [imgResult, imRawDrawing, imgInvWarp, imgFinal])
            cv2.imshow('Final result', imgFinal)
    except:    
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny], 
                        [imgBlank, imgBlank, imgBlank, imgBlank], 
                        [imgBlank, imgBlank, imgBlank, imgBlank])
        
        labels = [["Original", "Gray", "Blur", "Canny"], 
                ["Contour", "BiggestContours", "Warp", "ThreshHold"], 
                ["Result", "Raw drawing", "InvWarp", "Final"]]
        imgStacked = utils.stackImages(imageArray, 0.5, labels)

        cv2.imshow("Stacked image", imgStacked)
        if cv2.waitKey(1) & 0XFF == ord('s'):
            cv2.imwrite('finalresult.jpg', imgFinal)
            cv2.waitKey(300)

# from flask import Flask, request, jsonify
# from flask_cors import CORS 
# import cv2
# import numpy as np
# import base64
# import utils

# app = Flask(__name__)
# CORS(app)

# widthImg = 700
# heightImg = 700
# questions = 5
# choices = 5
# answers = [1, 2, 0, 1, 4]
# path = "./images/1.jpeg"

# def process_image(image):
#     img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (widthImg, heightImg))
#     imgContours = img.copy()
#     imgBiggestContours = img.copy()
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
#     imgCanny = cv2.Canny(imgBlur, 10, 50)

#     # FIND THE CONTOURS OF IMAGE
#     contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)

#     # Find rectangles and perform perspective transformation
#     rectCon = utils.rectContour(contours) 
#     biggestContour = utils.getCornerPoints(rectCon[0])
#     gradePoints = utils.getCornerPoints(rectCon[1])

#     if (biggestContour.size != 0 and gradePoints.size != 0):
#         cv2.drawContours(imgBiggestContours, biggestContour, -1, (0,255,0), 20)
#         cv2.drawContours(imgBiggestContours, gradePoints, -1, (255,0,0), 20)

#         biggestContour = utils.reorderPoints(biggestContour)
#         gradePoints = utils.reorderPoints(gradePoints)

#         firstPoint = np.float32(biggestContour)
#         secondPoint = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
#         pointsMatrix = cv2.getPerspectiveTransform(firstPoint, secondPoint)
#         imgWarpColorRed = cv2.warpPerspective(img, pointsMatrix, (widthImg, heightImg))

#         firstGradePoint = np.float32(gradePoints)
#         secondGradePoint = np.float32([[0,0], [325, 0], [0, 150], [325, 150]])
#         gradePointsMatrix = cv2.getPerspectiveTransform(firstGradePoint, secondGradePoint)
#         imgGrade = cv2.warpPerspective(img, gradePointsMatrix, (325, 150))

#         # Apply threshold
#         imgWarpGray = cv2.cvtColor(imgWarpColorRed, cv2.COLOR_BGR2GRAY)
#         imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
#         boxes = utils.splitBoxes(imgThresh)

#         questionsAndChoices = np.zeros((questions, choices))
#         columnCount = 0
#         rowCount = 0

#         for image in boxes:
#             totalPixels = cv2.countNonZero(image)
#             questionsAndChoices[rowCount][columnCount] = totalPixels
#             columnCount += 1
#             if columnCount == choices:
#                 rowCount += 1
#                 columnCount = 0

#         currentIndexes = []
#         for x in range(0, questions):
#             currentRow = questionsAndChoices[x]
#             selectedChoice = np.where(currentRow == np.amax(currentRow))
#             currentIndexes.append(selectedChoice[0][0])

#         grading = []
#         for i in range(0, questions):
#             if answers[i] == currentIndexes[i]:
#                 grading.append(1)
#             else: 
#                 grading.append(0)
        
#         score = (sum(grading) / questions) * 100

#     imgResult = imgWarpColorRed.copy()
#     imgResult = utils.showAnswers(imgWarpColorRed, currentIndexes, grading, answers, questions, choices)
#     _, img_encoded = cv2.imencode('.png', imgResult)
#     # Encode the image as base64
#     img_bytes = img_encoded.tobytes()
#     img_base64 = base64.b64encode(img_bytes).decode('utf-8')

#     return img_base64 
#     # return imgContours  

# @app.route('/analyze', methods=['POST'])
# def analyze_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'})

#     image = request.files['image']
#     result = process_image(image)

#     return jsonify({'processed_image': result})  

# if __name__ == '__main__':
#     app.run(debug=True)
