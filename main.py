import numpy as np
import cv2 as cv
from pipeline import *

# https://datahacker.rs/006-opencv-projects-how-to-detect-contours-and-match-shapes-in-an-image-in-python/
# Trying to follow this tutorial
    
camera = Camera(0)
# This tries to find the edges of
# red objects
pipe = Pipeline(
        red_mask, # isolate red
        grayscale, # convert to grayscale
        gaussian_blur, # blur
        lambda image: canny_edge(image, 70, 120), # edge enhance
        dilate, # idk what this does
        get_contours # return edges
        )

while (frame := camera.grab_frame()) is not None:
    contours = pipe.process(frame)
    # draw edges on top of original image
    frame = draw_contours(frame, contours)
    # show image on screen
    show(frame)


