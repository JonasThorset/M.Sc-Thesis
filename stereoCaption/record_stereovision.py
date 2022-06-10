from concurrent.futures import thread
from xml.etree.ElementTree import tostring
import cv2
import numpy as np
import threading
import time
import importlib.util
#from cameraCalibration import undistortImage
from undistort import undistort

"""
Creating an object of the CamRecord class will create a new thread that records video from a USB camera.
To start recording .start() has to be called(example in main). Press the keyboard number of the respective camera to terminate and save video.
"""

################# SET PARAMS ####################
saveVideo = True
folderSavePath = "./Videos/"
outputName = "test1"
vidFormat = ".mp4"
im_width = 640 #+ int(640*0.5)     #cv2 default: 640
im_height = 480 #+ int(480*0.5)    #cv2 default: 480
dim = (im_width, im_height)
#################################################


class CamRecord(threading.Thread):
    def __init__(self, ID):
        threading.Thread.__init__(self)
        self.ID = ID
    def run(self, K, D, save = False):
        print("Camera " + str(self.ID+1), " recording [PRESS '" +str(self.ID+1) +"' to quit]")
        recordVideo(self.ID, save)
        

def recordVideo(ID, save):
    cv2.namedWindow("Camera" + str(ID+1))
    vid = cv2.VideoCapture(ID)
    K = np.loadtxt("./Images/CalibImages/Cam" + str(ID+1) + "/K_air.txt")
    D = np.loadtxt("./Images/CalibImages/Cam" + str(ID+1) + "/D_air.txt")
    
    if not (vid.isOpened()):
        raise IOError("Cannot activate camera"+ str(ID+1))
    
    if(save):
        out = cv2.VideoWriter(folderSavePath + outputName+'_cam'+str(ID+1) + vidFormat, -1, 20.0, dim)

    retval = True
    while(retval):
        retval, img = vid.read()
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        print(img.shape)
        img = undistort(img, K, D)
        print(img.shape)
        if save:
            out.write(img)
        cv2.imshow("Camera" + str(ID+1), img)
        key = cv2.waitKey(1) & 0xFF
        if(key == ord(str(ID+1))):
            break
    vid.release()
    out.release()
    cv2.destroyWindow("Camera" + str(ID+1))






if __name__ == "__main__":
    
    video1 = CamRecord(0, saveVideo)
    #video2 = CamRecord(1, saveVideo)
    #video3 = CamRecord(2, saveVideo)
    video1.start()
    #video2.start()
    #video3.start()
    time.sleep(3)
    print("\nActive threads", threading.activeCount())