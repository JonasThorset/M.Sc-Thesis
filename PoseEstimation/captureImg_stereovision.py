import cv2
import numpy as np
import threading
import time
import os.path
from os import path
from calibrateFisheye import findCorners, calibrate



"""
Creating an object of the CamCaptureImg class will create a new thread of video stream. To capture
an image, press the keyboard button 'c'. The image will be saved to the specified folder path. To 
terminate the session, press 'q' on all windows.
"""

class CamCaptureImg(threading.Thread):
    def __init__(self, ID, dim, savePath = "./Images/"):
        threading.Thread.__init__(self)
        self.ID = ID
        self.savePath = savePath
        self.dim = dim
       
    def run(self):
        captureImage(self.ID, dim, self.savePath)


def captureImage(ID, dim, folderPath = "./Images/"):
    vid = cv2.VideoCapture(ID)

    if not (vid.isOpened()):
        raise IOError("Cannot activate camera"+ str(ID+1))
    

    cv2.namedWindow("Camera" + str(ID+1))
    
    imgCount = 0
    while True:
        retval, img = vid.read()
        if not retval:
            print("Unable to read frame")
            break
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Camera" + str(ID+1), img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Press 'q' to terminate
            break
        elif key == ord('c'):
            # Press 'c' to capture image
            out = folderPath + "Cam" + str(ID+1)+ "_frame" + str(imgCount) + imgFormat
            while (path.exists(out)):
                imgCount +=1
                out = folderPath + "Cam" + str(ID+1)+ "_frame" + str(imgCount) + imgFormat
            cv2.imwrite(out, img)
            time.sleep(0.1)
           
            print("frame "+ str(imgCount) + " from camera "+ str(ID+1) + " saved...")
            imgCount += 1

                
            
    vid.release()
    cv2.destroyWindow("Camera" + str(ID+1))

#-------------------------------------------------------------------------------------
    


if __name__ == "__main__":
    imgFormat = ".jpg"

    im_width = 1920      #cv2 default: 640
    im_height = 1080     #cv2 default: 480
    dim = (im_width, im_height)
    savePath = "./BR/"

    camThread = CamCaptureImg(2, dim, savePath)
    

    camThread.start()
   