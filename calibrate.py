import cv2
import numpy as np
import os
import glob
import pipeline
import json
from pipeline import *

def calibrate(image_path, output_path):
    """
    image_path: where the pictures of the chessboard are
    output_path: file to store the calibration in
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    print("Calibrating camera from images...")
    CHECKERBOARD = (6,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = [] 
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    images = glob.glob(image_path + '*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE) 
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    h,w = img.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Done!")
    np.savez(output_path, mtx, dist)
    print("Saved calibration to", output_path)
     
if __name__ == "__main__": 
    calibrate("./calib/", "./calibrate")
