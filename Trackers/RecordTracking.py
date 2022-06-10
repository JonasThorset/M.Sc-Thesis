from concurrent.futures import thread
from msilib.schema import Error
from xml.etree.ElementTree import tostring
import cv2
import numpy as np
import threading
import time
import importlib.util
from undistort import undistort
from Tracker import*
from Detector import DarknetModel

class RecordTracking(threading.Thread):
    def __init__(self, ID, K, D, DIM, savePath):
        threading.Thread.__init__(self)
        self.ID = ID
        self.K = K
        self.D = D
        self.DIM = DIM
        self.savePath = savePath
    def run(self):
        print("Camera " + str(self.ID+1), " recording [PRESS '" +str(self.ID+1) +"' to quit]")
        record(self.ID, self.K, self.D, self.DIM, self.savePath)
        

def record(ID, K, D, dim, savePath):
    cv2.namedWindow("Camera" + str(ID+1))
    vidStream = cv2.VideoCapture(ID)
    retval, frame = vidStream.read()
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    frame = undistort(frame, K, D) 
    yolov4 = DarknetModel('./DetectorData/coco.names', "./DetectorData/yolov4-tiny.weights", "./DetectorData/yolov4-tiny.cfg")
    initBboxes = yolov4.estimateBboxes(frame, 0.5)
    
    while ((len(initBboxes) == 0) and retval):
        retval, frame = vidStream.read()
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        frame = undistort(frame, K, D) 
        initBboxes = yolov4.estimateBboxes(frame, 0.5)
        print("NUMBER OF INITIAL BBOXES:", len(initBboxes))

    objTrackers = []
    for init_bbox in initBboxes:
        objTrackers.append(ObjectTracking(frame, init_bbox, cv2.legacy.TrackerMOSSE_create()))

    if not (vidStream.isOpened()):
        raise ValueError("Cannot activate camera"+ str(ID+1))
    
    out = cv2.VideoWriter(savePath +'_cam'+str(ID+1) + ".mp4", -1, 20.0, dim)

    retval = True
    while(retval):
        timer = cv2.getTickCount()
        retval, frame = vidStream.read()
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        frame = undistort(frame, K, D)
        for objTracker in objTrackers:
            bbox_pair[ID].append(objTracker.bbox)
            objTracker.updateBbox(frame)
            
            if objTracker.retval:
                objTracker.bbox.drawBbox(frame)
                objTracker.bbox.drawCenter(frame)

        frameRate = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        cv2.putText(frame, "FPS: " + str(int(frameRate)), org = (0,20), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color = (255,0,0), thickness = 2)       
        out.write(frame)
        cv2.imshow("Camera" + str(ID+1), frame)
        key = cv2.waitKey(1) & 0xFF
        if(key == ord(str(ID+1))):
            break
    vidStream.release()
    out.release()
    cv2.destroyWindow("Camera" + str(ID+1))



if __name__ == "__main__":

    ################# SET PARAMS ####################
    folderSavePath = "./Results/"
    im_width = 640 #+ int(640*0.5)     #cv2 default: 640
    im_height = 480 #+ int(480*0.5)    #cv2 default: 480
    DIM = (im_width, im_height)

    K1 = np.loadtxt("./CalibData/Cam1/K_air.txt")
    D1 = np.loadtxt("./CalibData/Cam1/D_air.txt")
    K2 = np.loadtxt("./CalibData/Cam2/K_air.txt")
    D2 = np.loadtxt("./CalibData/Cam2/D_air.txt")
    #################################################

    global bbox_pair
    bbox_pair = [[], [], []]
    
    video1 = RecordTracking(0, K1, D1, DIM, folderSavePath)
    video1.start()
    time.sleep(10)
    video2 = RecordTracking(1, K2, D2, DIM, folderSavePath)
    video2.start()
    
    time.sleep(3)
    print(bbox_pair)
    print("\nActive threads", threading.activeCount())