# YOLO object detection
import cv2
import numpy as np
import time
from Bbox import Bbox



class DarknetModel:
    def __init__(self, classnames_path, weight_path, cfg_path):
        self.classes = open(classnames_path).read().strip().split('\n')
        self.model = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        layerNames = self.model.getLayerNames()
        self.outLayerIdx = [layerNames[i - 1] for i in self.model.getUnconnectedOutLayers()]
        np.random.seed(42)
        self.classColors =  np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
    
    def estimateBboxes(self, frame, conf):
        frameH, frameW = frame.shape[:2]
        boxes = []
        confidences = []
        classIDs = []
        goodBoxes = []

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        t0 = time.time()
        outputs = self.model.forward(self.outLayerIdx)
        t = time.time() - t0
        #print("Forward prop time:", t)
        outputs = np.vstack(outputs)
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([frameW, frameH, frameW, frameH])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                color = [int(c) for c in self.classColors[classIDs[i]]]
                text = "{}: {:.4f}".format(self.classes[classIDs[i]], confidences[i])
                goodBoxes.append(Bbox(x,y,w,h, text, color, classIDs[i]))
        return goodBoxes


if __name__ == "__main__":
    
    classPath = './DetectorData/coco.names'
    weight_path = "./DetectorData/yolov4-tiny.weights"
    cfg_path = "./DetectorData/yolov4-tiny.cfg"

    yolov4 = DarknetModel(classPath, weight_path, cfg_path)
         

    cv2.namedWindow('window')
    retval = True
    vidStream = cv2.VideoCapture(0)
    while retval:
        retval, frame = vidStream.read()
        bboxes = yolov4.estimateBboxes(frame, 0.5)
        for bbox in bboxes:
            bbox.drawBbox(frame)
        cv2.imshow('window',frame)
        key = cv2.waitKey(1) & 0xFF
        if(key == ord(str('q'))):
            break
    vidStream.release()
    cv2.destroyWindow('window')
    
    """
    frame = cv2.imread("./DetectorData/Images/0001.png")
    bboxes = yolov4.estimateBboxes(frame, 0.5)
    for bbox in bboxes:
        print(bbox.x,bbox.y)
        bbox.drawBbox(frame)

    cv2.imshow('window',  frame)
    cv2.waitKey(0)
    """