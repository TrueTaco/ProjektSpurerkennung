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

    both_curves = left_curve + right_curve

    kernel_small5 = np.array([[0,1,0],[1,1,1],[0,1,0]], 'uint8')
    both_curves = cv.dilate(both_curves, kernel_small5, iterations=5)

    both_curves_filtered = frame.copy()
    both_curves_filtered[np.where(both_curves==0)] = 0
    

    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]],np.float32)
    both_curves_filtered = cv.filter2D(both_curves_filtered,-1,kernel)

    return both_curves_filtered

def getXValues(img, right):
    img_hsv = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    img_gray = cv.cvtColor(img_hsv, cv.COLOR_BGR2GRAY)
    
    height = len(img)
    width = len(img[0])

    maxWidth = int(width/2)
    leftPoints = []
    rightPoints = []

    for i in range(height):
        pointCloudLeft = np.where(img_gray[i][0:maxWidth - 1] != 0)
        if(len(pointCloudLeft[0]) != 0):
            xMean = np.mean(pointCloudLeft[0])
            # xMedian = int((pointCloudLeft[0][0] + pointCloudLeft[0][-1])/2)
            leftPoints.append(xMean)
            # print("mean left: ", xMean)
            # print("median left: ", xMedian)
        else:
            leftPoints.append(0)

        pointCloudRight = np.where(img_gray[i][maxWidth + 1: width - 1] != 0)
        if(len(pointCloudRight[0]) != 0):
            xMean = np.mean(pointCloudLeft[0])
            # xMedian = int((pointCloudRight[0][0] + pointCloudRight[0][-1])/2)
            rightPoints.append(xMean)
            # print("mean right: ", xMean)
            # print("median right: ", xMedian)
        else:
            rightPoints.append(1)

    return leftPoints, rightPoints

def curveRadius(frame, xleft, xright):
    y = np.linspace(0, frame.shape[0], frame.shape[0])
    x_m_per_pix = 30.5/720
    y_m_per_pix = 3.7/720
    y_eval = np.max(y)
    
    curveleft = np.polyfit((y*y_m_per_pix), (np.array(xleft)*x_m_per_pix), 2)
    curveright = np.polyfit((y*y_m_per_pix), (np.array(xright)*x_m_per_pix), 2)

    curverad_left = ((1 + (2*curveleft[0]*y_eval*y_m_per_pix + curveleft[1])**2)**1.5) / np.absolute(2*curveleft[0])
    curverad_right = ((1 + (2*curveright[0]*y_eval*y_m_per_pix + curveright[1])**2)**1.5) / np.absolute(2*curveright[0])

    return curverad_left, curverad_right


capture = cv.VideoCapture('img/Udacity/project_video.mp4')
frameNr = 0

newFrameTime = 0
prevFrameTime = 0
font = cv.FONT_HERSHEY_SIMPLEX

# success = True
# capture.set(1,75)
# success, frame = capture.read()

while(True):
    success, frame = capture.read() 
    if frameNr == 1 or frameNr == 0:
        all_curves = process_frame(frame)

    if(success):
        newFrameTime = time.time()

        # Updates detected lanes every 3 seconds in-order to increase performance
        # if frameNr%10 == 0:
        all_curves = process_frame(frame)

        # Perspective transformation:
        # Chose points:
        old_points = np.float32([[200,720],[1000,720],[510,460],[690,460]])
        new_points = np.float32([[0,1200],[720,1200],[0,0],[720,0]])

        # Apply transformation
        perspective_transform = cv.getPerspectiveTransform(old_points,new_points)
        all_curves_transformed = cv.warpPerspective(all_curves, perspective_transform,(1200,720))
        frame_transformed = cv.warpPerspective(frame, perspective_transform,(1200,720))

        # Get current X values of the left line:        
        leftPoints, rightPoints = getXValues(all_curves_transformed, False)
        # Calculate 
        curveRadiusLeft, curveRadiusRight = curveRadius(all_curves_transformed, leftPoints, rightPoints)

        alpha = 0.4
        beta = (1.0 - alpha)
        dark_mask = cv.addWeighted(frame, alpha, all_curves, beta, 0.0)

        birds_eye_view = frame_transformed
        darkened_image = all_curves + dark_mask

        fps = 1/(newFrameTime - prevFrameTime)
        prevFrameTime = newFrameTime
        fps = int(fps)
        fps = str(fps)


        cv.putText(darkened_image, fps, (7, 30), font, 1, (100, 255, 0), 1, cv.LINE_AA)
        cv.imshow("Normal POV", darkened_image)

        curveLeftText = "Radius left: " + str(curveRadiusLeft)
        curveRightText = "Radius left: " + str(curveRadiusRight)
        cv.putText(birds_eye_view, curveLeftText, (7, 60), font, 1, (100, 255, 0), 1, cv.LINE_AA)
        cv.putText(birds_eye_view, curveRightText, (7, 90), font, 1, (100, 255, 0), 1, cv.LINE_AA)
        cv.imshow("Birds-Eye View", birds_eye_view)




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
    