"""
Helper functions and classes
"""
import cv2 as cv
import numpy as np

class Pipeline:
    """
    A Pipeline applies functions to an image one after another
    """

    def __init__(self, *args):
        """ 
        Create a pipeline with a list of functions
        EX: pipeline = Pipeline(red_mask, grayscale, show)
        """
        self.steps = [a for a in args]

    def process(self, image):
        image = image.copy()
        for func in self.steps:
            image = func(image)
        return image

class Camera:
    """
    Wrapper for opencv camera
    """
    
    def __init__(self, id):
        """
        Open camera with a given id
        """
        self.id = id
        self.capture = cv.VideoCapture(id)
        self.width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.calibrated = False
        if not self.capture.isOpened():
            raise Exception("Cannot open camera", id)

    def grab_frame(self):
        """
        Get the next frame and apply camera calibration
        """
        ret, frame = self.capture.read()
        if cv.waitKey(1) == ord('q'):
            print("Exit key pressed")
            return None
        if not ret:
            print("Failed to grab frame, stream closed?")
            return None
        return self.undistort(frame)

    def calibrate(self, from_mtx, dist):
        """
        Apply a camera calibration, handles calibration
        and grabbing frames
        """
        to_mtx, roi = cv.getOptimalNewCameraMatrix(from_mtx, dist, (self.width, self.height), 1, (self.width, self.height))
        self.from_mtx = from_mtx
        self.to_mtx = to_mtx
        self.dist = dist
        self.roi = roi
        self.calibrated = True
        return self

    def calibrate_from(self, file):
        """
        Apply a camera calibration from a file
        """
        data = np.load(file)
        from_mtx = data["arr_0"]
        dist = data["arr_1"]
        self.calibrate(from_mtx, dist)

    def undistort(self, frame):
        """
        Apply corrections based on camera calibration
        """
        # Exit early if camera not calibrated
        if not self.calibrated:
            return frame
        x, y, w, h = self.roi
        return cv.undistort(frame, self.from_mtx, self.dist, None, self.to_mtx)[y:y+h, x:x+w]

    def close(self):
        """
        Stop capturing the camera
        """
        self.capture.release()

def red_mask(image):
    """
    Extract red from image, output grayscale image
    https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
    """
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_1 = np.array([0, 100, 20])
    upper_1 = np.array([10, 255, 255])
    lower_2 = np.array([160,100,20])
    upper_2 = np.array([179,255,255])
     
    lower_mask = cv.inRange(image_hsv, lower_1, upper_1)
    upper_mask = cv.inRange(image_hsv, lower_2, upper_2)
     
    mask = lower_mask + upper_mask
    return cv.bitwise_and(image, image, mask=mask)

def blue_mask(image):
    """
    Extract blue from image, output grayscale image
    https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html 
    """
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower = np.array([110,100,20]) 
    upper = np.array([130,255,255])
    mask = cv.inRange(image_hsv, lower, upper)
    return cv.bitwise_and(image, image, mask=mask)

def grayscale(image):
    """
    Convert BRG to grayscale
    https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def gaussian_blur(image):
    """
    https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    """
    return cv.GaussianBlur(image, (5, 5), cv.BORDER_DEFAULT)

def canny_edge(image, min, max):
    """
    https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    """
    return cv.Canny(image, min, max)

def dilate(image):
    return cv.dilate(image, np.ones((3)), iterations=1)

def get_contours(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image, contours):
    return cv.drawContours(image, contours, -1, (255, 255, 255), 10)

def scale_image(image, scale):
    """
    Resize an image. < 1 is bigger > 1 is smaller
    https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
    """
    return cv.resize(image, None, fx=scale, fy=scale)
    
def show(image):
    """
    Show a frame from a video on screen. Needs to be called in a loop
    https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    """
    cv.imshow("Window", image)
    return image

def show_image(image):
    """
    Shows an image and waits for the user to press q
    https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
    """
    while True:
        show(image)
        if cv.waitKey(0) == ord("q"):
            cv.destroyAllWindows()
            return

def dbg_loop(camera, *args):
    """
    Quick and dirty way to create a camera and 
    pipeline and display its output on the screen
    """
    pipeline = Pipeline(*args)
    while (frame := camera.grab_frame()) is not None:
        pipeline.process(frame)     
    camera.close()
    cv.destroyAllWindows()

    
def list_ports():
    """
    Cameras get assigned weird id's sometimes so this tries
    every id from 0-100 and prints the ones that are connected
    """
    working = []
    for i in range(100):
        camera = cv.VideoCapture(i)
        if camera.isOpened():
            working.append(i)
    print("Working ports: ", working)
    return working
