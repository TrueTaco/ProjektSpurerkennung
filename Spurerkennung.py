import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time
from time import sleep
from IPython import display
from matplotlib.patches import Rectangle as Rect
import statistics as stat

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv.fillPoly(mask, np.array([vertices],np.int32), match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

def process_frame(frame):
    img1_hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)

    height = len(img1_hsv)
    width = len(img1_hsv[0])
    region_of_interest_vertices = [[0, height],[width / 2, height / 2],[width, height],]   
    img_region_of_interest = region_of_interest(img1_hsv, region_of_interest_vertices)
    

    lower_white = np.array([60,0,220], dtype=np.uint8)
    upper_white = np.array([110,10,255], dtype=np.uint8)
    left_curve = cv.inRange(img_region_of_interest, lower_white, upper_white)
    right_curve = cv.inRange(img_region_of_interest, (15, 40, 230), (255, 255, 255))

    kernel_small5 = np.array([[0,1,0],[1,1,1],[0,1,0]], 'uint8')
    left_curve = cv.dilate(left_curve, kernel_small5, iterations=5)
    right_curve = cv.dilate(right_curve, kernel_small5, iterations=5)

    left_curve_filtered = frame.copy()
    left_curve_filtered[np.where(left_curve==0)] = 0
    right_curve_filtered = frame.copy()
    right_curve_filtered[np.where(right_curve==0)] = 0

    

    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]],np.float32)
    left_curve_filtered = cv.filter2D(left_curve_filtered,-1,kernel)
    right_curve_filtered = cv.filter2D(right_curve_filtered,-1,kernel)

    return left_curve_filtered,right_curve_filtered

def findFourVertices(img_filtered):
    height = len(img_filtered)
    width = len(img_filtered[0])

    startHeight = int(height/1.5)
    maxWidth = int(width/2)
    points = []

    tempHeight = startHeight
    valueFound = False
    while valueFound == False:
        # Top left point
        pointCloud = np.where(img_filtered[startHeight][:maxWidth - 1] != 0)
        if(len(pointCloud[0]) != 0):
            # print(pointCloud)
            xMean = int((pointCloud[0][0] + pointCloud[0][-1])/2)
            points.append((xMean, startHeight))

            valueFound = True
        else:
            startHeight += 1

    startHeight = tempHeight
    valueFound = False
    while valueFound == False:
        # Top right point
        pointCloud = np.where(img_filtered[startHeight][maxWidth + 1:] != 0)
        if(len(pointCloud[0]) != 0):
            xMean = int((pointCloud[0][0] + pointCloud[0][-1])/2)
            points.append((xMean, startHeight))
            valueFound = True
        else:
            startHeight += 1

    
    valueFound = False
    startHeight = height - 1
    tempHeight = startHeight
    while valueFound == False:
        # Bottom left point
        pointCloud = np.where(img_filtered[startHeight][:maxWidth - 1] != 0)
        if(len(pointCloud[0]) != 0):
            # print(pointCloud)
            xMean = int((pointCloud[0][0] + pointCloud[0][-1])/2)
            points.append((xMean, startHeight))
            valueFound = True
        else:
            startHeight -= 1

    startHeight = tempHeight
    valueFound = False
    while valueFound == False:
        # Bottom right point
        pointCloud = np.where(img_filtered[startHeight][maxWidth + 1:] != 0)
        if(len(pointCloud[0]) != 0):
            xMean = int((pointCloud[0][0] + pointCloud[0][-1])/2)
            points.append((xMean, startHeight))
            valueFound = True
        else:
            startHeight -= 1

    print("Points: ", points)


def perspective_transformation():
    return "hello"

capture = cv.VideoCapture('img/Udacity/project_video.mp4')
frameNr = 0

newFrameTime = 0
prevFrameTime = 0
font = cv.FONT_HERSHEY_SIMPLEX

# success = True
# capture.set(1,2)
# success, frame = capture.read()

while(True):
    success, frame = capture.read() 

    if(success):
        newFrameTime = time.time()

        left_curve, right_curve = process_frame(frame)
        all_curves = left_curve + right_curve
        findFourVertices(all_curves)
        alpha = 0.4
        beta = (1.0 - alpha)
        dark_mask = cv.addWeighted(frame, alpha, all_curves, beta, 0.0)
        darkened_image = all_curves + dark_mask
        fps = 1/(newFrameTime - prevFrameTime)
        prevFrameTime = newFrameTime
        fps = int(fps)
        fps = str(fps)

        cv.putText(darkened_image, fps, (7, 30), font, 1, (100, 255, 0), 1, cv.LINE_AA)
        cv.imshow("Current Frame", darkened_image)

        frameNr += 1
        key = cv.waitKey(1)
        if key == ord("q"):
            cv.destroyAllWindows()
            print("Playback interrupted by user.")
            break
        if key == ord('p'):
            cv.waitKey(-1) #wait until any key is pressed
    else:
        print("Playback finished.")
        cv.destroyAllWindows()
        capture.release()
        break
    