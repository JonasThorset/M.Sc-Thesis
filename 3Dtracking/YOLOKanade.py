from Bbox import*
from Detector import*
import glob
import cv2


class YOLOKanade :
    def __init__(self,  init_frame, detector):
        self.frame_prev = init_frame
        self.bboxes = detector.estimateBboxes(init_frame, 0.5)
        self.detector = detector
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))
        self.frame_prev_gray = cv2.cvtColor(self.frame_prev, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.frame_prev_gray, mask = None, **self.feature_params)
        self.d_vecs = []
        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(self.frame_prev)
        
 
    def updateBboxSizes(self, frame_next, iou_thresh):
        new_bboxes = self.detector.estimateBboxes(frame_next, 0.5)

        for i in range(len(new_bboxes)):
            for j in range(len(self.bboxes)):
                if iou(new_bboxes[i], self.bboxes[j]) > iou_thresh:
                    self.bboxes[j].w, self.bboxes[j].h = int(new_bboxes[i].w), int(new_bboxes[i].h)
    
    def updateBboxCenters(self):
        for i in range(len(self.bboxes)):
            for j in range(self.d_vecs.shape[0]):
               pt =  self.bboxes[i].getCenter() + (self.d_vecs[j, 0], self.d_vecs[j, 1])
               if self.bboxes[i].isInBbox(pt):
                    self.bboxes[i].x = int(self.bboxes[i].x + self.d_vecs[j, 0])
                    self.bboxes[i].y =  int(self.bboxes[i].y + self.d_vecs[j, 1])
                    return

    
    def kanadeUpdate(self, frame_next):
        frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.frame_prev_gray, frame_next_gray, self.p0, None, **self.lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
        # draw the tracks
        d_vecs = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            frame_next = cv2.circle(frame_next, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            d_vecs.append(np.array([a-c, b-d]))
        
        self.frame_prev_gray = frame_next_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        

        self.d_vecs = np.array(d_vecs)
    
    def yoloUpdate(self, frame_next, iou_thresh):
        yolo_bboxes = self.detector.estimateBboxes(frame_next, 0.6)
        for i in range(len(yolo_bboxes)):
            for j in range(len(self.bboxes)):
                if iou(yolo_bboxes[i], self.bboxes[j])[0] > iou_thresh:
                    self.bboxes[j] = yolo_bboxes[i]
                if iou(yolo_bboxes[i], self.bboxes[j])[0] <= 0:
                    self.bboxes.append(yolo_bboxes[i])

    def removeKanadePtsOutsideBboxes(self):
        new_p0 = []
        for i in range(len(self.p0)):
            print(self.p0[i])
            for j in range(len(self.bboxes)):
                if self.bboxes[j].isInBbox(self.p0[i][0]):
                    new_p0.append(self.p0[i])
        
        new_p0 = np.array(new_p0).reshape(-1, 1, 2)
        self.p0 = new_p0
    
    def renewFeaturesInsideBboxes(self):
        p0_inside_bboxes = []
        p0 = cv2.goodFeaturesToTrack(self.frame_prev_gray, mask = None, **self.feature_params)
        for i in range(len(self.bboxes)):
            for j in range(p0.shape[0]):
                if self.bboxes[i].isInBbox(p0[j][0]):
                    p0_inside_bboxes.append(p0[j])
        
        p0_inside_bboxes = np.array(p0_inside_bboxes).reshape(-1,1,2)
        self.p0 = p0_inside_bboxes
        


if __name__ == "__main__":
    classPath = './DetectorData/coco.names'
    weight_path = "./DetectorData/yolov4-tiny.weights"
    cfg_path = "./DetectorData/yolov4-tiny.cfg"
    yolov4 = DarknetModel(classPath, weight_path, cfg_path)

    images_cam0 = [cv2.imread(file) for file in glob.glob('./Cam1/images/yrot/*.png')]
    images_cam0 += [cv2.imread(file) for file in glob.glob('./Cam1/images/xrot/*.png')]
    init_frame = images_cam0[0]

    tracker = YOLOKanade(init_frame, yolov4)

    for i in range(1, len(images_cam0)):
        dvecs = tracker.kanadeUpdate(images_cam0[i])
        tracker.updateBboxCenters()
        tracker.yoloUpdate(images_cam0[i], iou_thresh = 0.3)
        tracker.removeKanadePtsOutsideBboxes()
        tracker.renewFeaturesInsideBboxes()
       

        for j in range(len(tracker.bboxes)):
            tracker.bboxes[j].drawBbox(images_cam0[i])
            tracker.bboxes[j].drawCenter(images_cam0[i])
        
        #img = cv2.add(images_cam0[i], tracker.mask)
        #cv2.imshow("window", img)
        cv2.imshow("window", images_cam0[i])
        cv2.waitKey(100)
