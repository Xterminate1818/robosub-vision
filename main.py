import numpy as np
import cv2 as cv

cv.setUseOptimized(True)

# HSV image -> red bitmask
def red_mask(image_hsv):
    # lower boundary RED color range values; Hue (0 - 10)
    lower_1 = np.array([0, 100, 20])
    upper_1 = np.array([10, 255, 255])
     
    # upper boundary RED color range values; Hue (160 - 180)
    lower_2 = np.array([160,100,20])
    upper_2 = np.array([179,255,255])
     
    lower_mask = cv.inRange(image_hsv, lower_1, upper_1)
    upper_mask = cv.inRange(image_hsv, lower_2, upper_2)
     
    full_mask = lower_mask + upper_mask
    return full_mask

# HSV image -> blue bitmask
def blue_mask(image_hsv):
    # define range of blue color in HSV
    lower = np.array([110,100,20])
    upper = np.array([130,255,255])
    return cv.inRange(image_hsv, lower, upper)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    is_recieved, frame = cap.read()
    
    # if frame is read correctly, exit
    if not is_recieved:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert to hsv
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Mask of red
    mask = red_mask(hsv)

    # Harris corner detection

    masked_image = cv.bitwise_and(frame, frame, mask=mask)

    # Display the resulting frame
    cv.imshow("frame", mask)
    if cv.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
