import cv2
import numpy as np
import threading
import time
from undistort import undistort
import os.path
from os import path
from calibrateFisheye import findCorners, calibrate



"""
Creating an object of the CamCaptureImg class will create a new thread of video stream. To capture
an image, press the keyboard button 'c'. The image will be saved to the specified folder path. To 
terminate the session, press 'q' on all windows.
"""

class CamCaptureImg(threading.Thread):
    def __init__(self, ID, calib = False, undistortMode ="", checkerboardDim = None, savePath = "./Images/"):
        threading.Thread.__init__(self)
        self.ID = ID
        self.calib = calib
        self.savePath = savePath
        self.checkerboardDim = checkerboardDim
        self.undistortMode = undistortMode
    def run(self):
        captureImage(self.ID, self.calib,self.undistortMode, self.checkerboardDim, self.savePath)


def captureImage(ID, calib = True, undistortMode = "", checkerboardDim = None, folderPath = "./Images/"):
    vid = cv2.VideoCapture(ID)

    if not (vid.isOpened()):
        raise IOError("Cannot activate camera"+ str(ID+1))
    if calib and checkerboardDim == None:
        raise ValueError("Specify correct checkerboard dimension")

    cv2.namedWindow("Camera" + str(ID+1))
    
    imgCount = 0
    deleteImage = False
    while True:
        retval, img = vid.read()
        if not retval:
            print("Unable to read frame")
            break
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if undistortMode == 'air':
            if calib:
                raise RuntimeError("Running both calibration and undistortion is not allowed")
            K = np.loadtxt("./Images/CalibImages/Cam" + str(ID+1) + "/" + "K_air.txt")
            D = np.loadtxt("./Images/CalibImages/Cam" + str(ID+1) + "/" + "D_air.txt")
            img = undistort(img, K, D)
        elif undistortMode == 'water':
            if calib:
                raise RuntimeError("Running both calibration and undistortion is not allowed")
            K = np.loadtxt("./Images/CalibImages/Cam" + str(ID+1) + "/" + "K_water.txt")
            D = np.loadtxt("./Images/CalibImages/Cam" + str(ID+1) + "/" + "D_water.txt")
            img = undistort(img, K, D)
       

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
            
            if calib:
                try:
                    deleteImage = False
                    grayShape, objpoints, imgpoints = findCorners([out], checkerboardDim, display = False)
                    _, _, _, _, _= calibrate(objpoints, imgpoints, grayShape)
                except:
                    print("CALIB_CHECK_COND - Ill-conditioned matrix during calibration")
                    deleteImage = True
            if deleteImage:
                os.remove(out)
            else:
                print("frame "+ str(imgCount) + " from camera "+ str(ID+1) + " saved...")
                imgCount += 1

                
            
    vid.release()
    cv2.destroyWindow("Camera" + str(ID+1))

#-------------------------------------------------------------------------------------
    


if __name__ == "__main__":
    imgFormat = ".jpg"

    im_width = 1920      #cv2 default: 640  1920
    im_height = 1080     #cv2 default: 480  1080
    dim = (im_width, im_height)

    #SET PARAMS BEFORE CAPTURING IMAGES. EXISTING IMAGES IN THE PATH MAY BE OVERWRITTEN!
    savePath = "./Images/"
    calibrateMode = True
    undistortMode = ""      #Can be either "", "air" or "water"
    CHECKERBOARD = (10,7)

    camThread = CamCaptureImg(0, calib = calibrateMode, undistortMode= undistortMode,
                                checkerboardDim = CHECKERBOARD, savePath =savePath)
    #camThread2 = CamCaptureImg(1, calib = calibrateMode, undistortMode= undistortMode,
     #                           checkerboardDim = CHECKERBOARD, savePath =savePath)
    #camThread3 = CamCaptureImg(2)

    camThread.start()
    #camThread2.start()
    #camThread3.start()

    time.sleep(3)
    print("\nActive threads", threading.activeCount())