import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from IPython import display

print('Willkommen beim Projekt "Erkennung von Spurmarkierungen"')


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
    #plt.imshow(frame)
    #plt.show()

    height = len(img1_hsv)
    width = len(img1_hsv[0])

    region_of_interest_vertices = [[0, height],[width / 2, height / 2],[width, height],]   

    img1_hsv = region_of_interest(img1_hsv, region_of_interest_vertices)

    lower_white = np.array([60,0,220], dtype=np.uint8)
    upper_white = np.array([110,10,255], dtype=np.uint8)
    mask = cv.inRange(img1_hsv, lower_white, upper_white)

    img_sign1 = cv.inRange(img1_hsv, (15, 40, 230), (255, 255, 255))

    img_sign = img_sign1 + mask
    kernel_small5 = np.array([[0,1,0],[1,1,1],[0,1,0]], 'uint8')
    img_sign = cv.dilate(img_sign, kernel_small5, iterations=5)

    img_filtered = frame.copy()
    img_filtered[np.where(img_sign==0)] = 0

    

    return img_filtered


capture = cv.VideoCapture('img/Udacity/project_video.mp4')
frameNr = 0


newFrameTime = 0
prevFrameTime = 0
font = cv.FONT_HERSHEY_SIMPLEX

while(True):
    success, frame = capture.read() 
    if(success):
        newFrameTime = time.time()

        processed_img = process_frame(frame)
        alpha = 0.4
        beta = (1.0 - alpha)
        dst = cv.addWeighted(frame, alpha, processed_img, beta, 0.0)
        test = dst + processed_img

        fps = 1/(newFrameTime - prevFrameTime)
        prevFrameTime = newFrameTime
        fps = int(fps)
        fps = str(fps)
        cv.putText(test, fps, (7, 30), font, 1, (100, 255, 0), 1, cv.LINE_AA)
        cv.imshow("Current Frame", test)
        #display.clear_output(wait=True)
        #plt.show()   
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
    