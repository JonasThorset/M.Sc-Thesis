import cv2
from Bbox import*


""" 
openCV trackers to choose from:

BOOSTING: cv2.legacy.TrackerBOOSTING_create() 
MEDIANFLOW: cv2.legacy.TrackerMEDIANFLOW_create() 
TLD: cv2.legacy.TrackerTLD_create() 
KCF: cv2.legacy.TrackerKCF_create() 
GOTURN: cv2.legacy.TrackerGOTURN_create() 
MOSSE: cv2.legacy.TrackerMOSSE_create() 
 """

class ObjectTracking :
    def __init__(self,  frame, init_bbox, tracker):
        self.bbox = init_bbox
        self.tracker = tracker
        self.retval = False
        self.tracker.init(frame, (init_bbox.x, init_bbox.y, init_bbox.w, init_bbox.h))
 
    def updateBbox(self, frame):
        retval, bbox = self.tracker.update(frame)
        self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        self.retval = retval


if __name__ == "__main__":

    """
    The following mainly demonstrates the IoU between two bounding boxes.
    """
    vidStream = cv2.VideoCapture(0)                           
    retval, frame = vidStream.read()                          
    init_bbox1 = cv2.selectROI("CAM1", frame, False)          
    init_bbox1 = Bbox(init_bbox1[0], init_bbox1[1], init_bbox1[2], init_bbox1[3])
    init_bbox2 = cv2.selectROI("CAM1", frame, False)          
    init_bbox2 = Bbox(init_bbox2[0], init_bbox2[1], init_bbox2[2], init_bbox2[3])
    IoU = iou(init_bbox1, init_bbox2)
    print(IoU)

    objTracker = ObjectTracking(frame, init_bbox1, cv2.legacy.TrackerMOSSE_create())

    while True:
        timer = cv2.getTickCount()
        retval, frame = vidStream.read() 
        objTracker.updateBbox(frame)

        if objTracker.retval:
            init_bbox2.drawBbox(frame)
            objTracker.bbox.drawBbox(frame)
            objTracker.bbox.drawCenter(frame)
            IoU, iouBox = iou(init_bbox2, objTracker.bbox)
            iouBox.drawBbox(frame)
        else:
            cv2.putText(frame, "Lost tracking", org = (0,50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = (255,0,0), thickness = 2)

        frameRate = cv2.getTickFrequency()/(cv2.getTickCount()-timer)

        cv2.putText(frame, "FPS: " + str(int(frameRate)), org = (0,20), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color = (255,0,0), thickness = 2)
        cv2.imshow("CAM1", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
