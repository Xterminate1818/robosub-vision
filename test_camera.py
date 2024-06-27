import numpy as np
import cv2 as cv
from pipeline import *

camera = Camera(0)
pp1 = Pipeline(show)

while (frame := camera.grab_frame()) is not None:
    frame1 = pp1.process(frame)

camera.close()
cv.destroyAllWindows()
