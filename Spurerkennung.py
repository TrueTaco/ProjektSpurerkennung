import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time

def region_of_interest(img, vertices):
    """
    - Sets all pixels to black outside of the region of interest defined by given verticies
    """
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
    """
    - Filters lanes by filtering out all colors not matching the color yellow and white (lane colors)
    - Dilates filtered lanes to increase visibility 
    """
    img1_hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)

    height = len(img1_hsv)
    width = len(img1_hsv[0])
    region_of_interest_vertices = [[0, height],[width / 2, height / 2],[width, height],]   
    img_region_of_interest = region_of_interest(img1_hsv, region_of_interest_vertices)

    # Filter colors
    lower_white = np.array([60,0,220], dtype=np.uint8)
    upper_white = np.array([110,10,255], dtype=np.uint8)
    left_curve = cv.inRange(img_region_of_interest, lower_white, upper_white)
    right_curve = cv.inRange(img_region_of_interest, (15, 40, 230), (255, 255, 255))

    # Enlarge lane for better visibility
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

def process_frame_gray_slicing(frame):
    """
    - Second attempt of filtering out the lanes by using slicing of grays
    - Failes in certain areas because of not enough contrast between road an lane
    """
    img_grey = cv.cvtColor(frame, cv.COLOR_RGB2HSV)

    thresh1 = 90

    w = len(img_grey[0])
    h = len(img_grey)

    height = len(img_grey)
    width = len(img_grey[0])
    region_of_interest_vertices = [[0, height],[width / 2, height / 2],[width, height],]   
    img_region_of_interest = region_of_interest(img_grey, region_of_interest_vertices)

    img_region_of_interest_old = cv.cvtColor(img_region_of_interest, cv.COLOR_RGB2GRAY)

    img_region_of_interest = np.zeros((h,w), dtype = int)

    for i in range(h-1):
        for j in range(w-1):
            if img_region_of_interest_old[i,j] <= thresh1: 
                img_region_of_interest[i,j] = 0
            else:
                img_region_of_interest[i,j] = 255
                
    plt.imshow(img_region_of_interest)

    return img_region_of_interest

def process_frame_gray(frame):
    """
    - Attempt at filtering out the lanes by using canny edge detection in combination with houghline detection
    """
    img_grey = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    edges = cv.Canny(img_grey,100,200,apertureSize = 3)
    lines = cv.HoughLines(edges,1,np.pi/180, 200)

    lines = [x for x in lines if x is not None]

    for r,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img_grey,(x1,y1), (x2,y2), (0,0,255),2)

    return img_grey

def findFourVertices(img_filtered):
    """
    - Previously used to detect start and endpoints of lanes
    - not used because of bad performance
    """
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

capture = cv.VideoCapture('img/Udacity/project_video.mp4')
frameNr = 0
newFrameTime = 0
prevFrameTime = 0
font = cv.FONT_HERSHEY_SIMPLEX

while(True):
    success, frame = capture.read() 
    if frameNr == 1 or frameNr == 0:
        left_curve, right_curve = process_frame(frame)

    if(success):
        newFrameTime = time.time()

        # Updates detected lanes every 3 seconds in-order to increase performance
        if frameNr%3 == 0:
            left_curve, right_curve = process_frame(frame)
        all_curves = left_curve + right_curve

        # Puts a mask over the old frame to improve visibility of the filtered lanes in "all_curves"
        alpha = 0.4
        beta = (1.0 - alpha)
        dark_mask = cv.addWeighted(frame, alpha, all_curves, beta, 0.0)
        darkened_image = all_curves + dark_mask

        # Calcultes FPS
        fps = 1/(newFrameTime - prevFrameTime)
        prevFrameTime = newFrameTime
        fps = int(fps)
        fps = str(fps) 

        # Perspective transformation for curve fitting
        old_points = np.float32([[200,720],[1000,720],[460,460],[740,460]])
        new_points = np.float32([[0,800],[600,800],[0,0],[600,0]])
        perspective_transform = cv.getPerspectiveTransform(old_points,new_points)
        new_perspecitve = cv.warpPerspective(darkened_image,perspective_transform,(800,600))
        
        # Boolean for activating/deactivating perspektive transform
        transform_perspective = True

        if transform_perspective == True:
            cv.putText(darkened_image, fps, (7, 30), font, 1, (100, 255, 0), 1, cv.LINE_AA)
            cv.imshow("Current Frame", darkened_image)
        else:
            cv.putText(new_perspecitve, fps, (7, 30), font, 1, (100, 255, 0), 1, cv.LINE_AA)
            cv.imshow("Current Frame", new_perspecitve)

        frameNr += 1
        key = cv.waitKey(1)
        # Control buttons for video playback
        # Press P to pause
        # Press q to quit
        if key == ord("q"):
            cv.destroyAllWindows()
            print("Playback interrupted by user.")
            break
        if key == ord('p'):
            cv.waitKey(-1) # Wait until any key is pressed
    else:
        print("Playback finished.")
        cv.destroyAllWindows()
        capture.release()
        break