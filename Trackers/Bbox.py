import cv2
from Tracker import*


class Bbox:
    def __init__(self, x, y, w, h, text = "", color = (0, 0, 255), classID = -1):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.text = text
        self.color = color
        self.classID = classID
        self.retval = True
       
    def drawBbox(self, frame):
        cv2.putText(frame, self.text, (self.x, self.y - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = self.color, thickness = 1)
        cv2.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y+self.h), color = self.color, thickness = 2, lineType = 1 )

    def drawCenter(self, frame):
        center = (int(self.x + self.w/2), int(self.y + self.h/2))
        cv2.circle(frame, center, radius = 1, color = self.color, thickness = 3)

    
    def getCenter(self):
        return (self.x + self.w/2, self.y + self.h/2)

def iou(bbox1, bbox2):

    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1.x, bbox2.x)
    y_top = max(bbox1.y, bbox2.y)
    x_right = min(bbox1.x +bbox1.w, bbox2.x + bbox2.w)
    y_bottom = min(bbox1.y + bbox1.h, bbox2.y + bbox2.h)
    
    iouBox = Bbox(x_left, y_top, x_right-x_left, y_bottom-y_top, text = "", color = (255,0,0))
    if x_right < x_left or y_bottom < y_top:
        iouBox.x, iouBox.y, iouBox.w ,iouBox.h = 0, 0, 0, 0
        return 0.0, iouBox

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = bbox1.w * bbox1.h
    bb2_area = bbox2.w * bbox2.h

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    
    return iou, iouBox