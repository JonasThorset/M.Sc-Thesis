from Bbox import*
from Detector import*
from Kalman_filter import*
from Bbox import*
import glob
import cv2

class YOLOKalman :
    def __init__(self, detector, init_frame, T, sigma_x, sigma_y, sigma_vx, sigma_vy, sigma_a):
        self.detector = detector
        
        bboxes = detector.estimateBboxes(init_frame, conf = 0.6)
        self.KF_bboxes = []
        for bbox in bboxes:
            x_init = np.array([bbox.x, bbox.y, 0, 0])
            z_init = x_init.copy()
            self.KF_bboxes.append(KF_CV(bbox, x_init, z_init, T, sigma_x, sigma_y, sigma_vx, sigma_vy, sigma_a))

    def kalmanPredict(self):
        for i in range(len(self.KF_bboxes)):
            self.KF_bboxes[i].predict()
    
    def getYOLOMeasurements(self, frame_next, conf = 0.6):
        bboxes = self.detector.estimateBboxes(frame_next, conf)
        return bboxes

    def kalmanUpdate(self, yolo_bboxes, iou_thresh):
        for i in range(len(yolo_bboxes)):
            for j in range(len(self.KF_bboxes)):
                print("IOU", iou(yolo_bboxes[i], self.KF_bboxes[j].bbox)[0])
                if iou(yolo_bboxes[i], self.KF_bboxes[j].bbox)[0] > iou_thresh:
                    x = yolo_bboxes[i].x
                    y = yolo_bboxes[i].y
                    vx = yolo_bboxes[i].x - self.KF_bboxes[j].x_prev[0]
                    vy = yolo_bboxes[i].y - self.KF_bboxes[j].x_prev[1]
                    z = np.array([x, y, vx, vy])
                    self.KF_bboxes[j].update(z)
                    self.KF_bboxes[j].bbox.w = yolo_bboxes[i].w
                    self.KF_bboxes[j].bbox.h = yolo_bboxes[i].h


if __name__ == "__main__":

    classPath = './DetectorData/coco.names'
    weight_path = "./DetectorData/yolov4-tiny.weights"
    cfg_path = "./DetectorData/yolov4-tiny.cfg"
    yolov4 = DarknetModel(classPath, weight_path, cfg_path)

    images_cam0 = [cv2.imread(file) for file in glob.glob('./Cam0/images/yrot/*.png')]
    images_cam0 += [cv2.imread(file) for file in glob.glob('./Cam0/images/xrot/*.png')]
    init_frame = images_cam0[0]

    YKF = YOLOKalman(yolov4, init_frame, T=1, sigma_x=0.9, sigma_y=0.9, sigma_vx=0.05, sigma_vy=0.05, sigma_a=0.05)

    for i in range(1, (len(images_cam0))):
        YKF.kalmanPredict()
        bboxes = YKF.getYOLOMeasurements(images_cam0[i], conf = 0.6)
        YKF.kalmanUpdate(bboxes, iou_thresh=0.1)

        for KF_bbox in YKF.KF_bboxes:
            KF_bbox.bbox.drawBbox(images_cam0[i])
            KF_bbox.bbox.drawCenter(images_cam0[i])
        
        cv2.imshow("window", images_cam0[i])
        cv2.waitKey(100)

