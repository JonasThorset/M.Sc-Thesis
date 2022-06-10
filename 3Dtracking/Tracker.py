import cv2
from Detector import*
from Bbox import*


class ObjectTracking :
    def __init__(self,  frame, tracker_type, ROI_selection, detector_domination = True, bbox_slack = 0):
        classPath = './DetectorData/coco.names'
        weight_path = "./DetectorData/yolov4-tiny.weights"
        cfg_path = "./DetectorData/yolov4-tiny.cfg"
        yolov4 = DarknetModel(classPath, weight_path, cfg_path)
        self.detector = yolov4
        self.detector_domination = detector_domination
        self.detectionCounter = 0
        
        if ROI_selection == 'manually':
            bbox = cv2.selectROI("Draw bounding box", frame, False)
            self.bbox = Bbox(bbox[0], bbox[1], bbox[2], bbox[3])

        if ROI_selection == 'yolo':
            bboxes = yolov4.estimateBboxes(frame, 0.5)
            print(len(bboxes))
            if len(bboxes) == 1:
                print("hei")
                self.bbox = bboxes[0]
            if len(bboxes) > 1:
                self.bbox = bboxes[0]
                print("WARNING: Multiple bboxes detected!")
            if len(bboxes) == 0:
                raise Exception("No bboxes found!")

        self.bbox.addSlack(bbox_slack)
        self.retval = False
        
        self.tracker_type = tracker_type
        self.createNewTracker(frame)
    
    def createNewTracker(self, frame):
        if self.tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if self.tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if self.tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if self.tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if self.tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if self.tracker_type == 'GOTURN':
           self.tracker = cv2.TrackerGOTURN_create()
        if self.tracker_type == 'MOSSE':
            self.tracker = cv2.legacy.TrackerMOSSE_create()
        if self.tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, (self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h))
 
    def updateBbox(self, frame):
        detections = self.detector.estimateBboxes(frame, conf = 0.6)
        if len(detections) > 0 and self.detector_domination:
            self.detectionCounter += 1
            print("hei")
            self.bbox = detections[0]
            self.bbox.addSlack(10)
            self.createNewTracker(frame)
            
        else:
            retval, bbox = self.tracker.update(frame)
            self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h = bbox[0], bbox[1], bbox[2], bbox[3]
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
