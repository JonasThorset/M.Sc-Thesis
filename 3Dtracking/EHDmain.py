import numpy as np
import cv2
import glob
import os
from Detector import*
from Bbox import*
from Tracker import*
from plotResults import*
from triangulateSinglePair import*
from utils import euclideanDistance
from undistort import undistortPinHole


select_ROI = 'manually' #can be either 'yolo' or 'manually'
tracker_name = 'CSRT'# Can be 'KCF, 'BOOSTING', 'MIL', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE' or 'CSRT'
slack = 5 #How much you want to expand the yolobbox with (pixels)
yolo_domination = True

def frameNr(path):
    file = path.split("frame")[-1]
    fileNr = file.split("cam")[0]
    return int(fileNr)

T0 = np.loadtxt("./Cams/EHD/Cam1/data/T0_refined2.txt")
T1 = np.loadtxt("./Cams/EHD/Cam2/data/T1_refined2.txt")
T2 = np.loadtxt("./Cams/EHD/Cam3/data/T2_refined2.txt")

K0 = np.loadtxt("./Cams/EHD/Cam1/data/K1080_EHD1_water.txt")
K1 = np.loadtxt("./Cams/EHD/Cam2/data/K1080_EHD2_water.txt")
K2 = np.loadtxt("./Cams/EHD/Cam3/data/K1080_EHD3_water.txt")

D0 = np.loadtxt("./Cams/EHD/Cam1/data/D1080_EHD1_water.txt")
D1 = np.loadtxt("./Cams/EHD/Cam2/data/D1080_EHD2_water.txt")
D2 = np.loadtxt("./Cams/EHD/Cam3/data/D1080_EHD3_water.txt")

 
img_paths0 = glob.glob('./../PoolTest/Imgs/Tennis/1080/Cam1/*.jpg')
img_paths0.sort(key = frameNr)
images_cam0 = [cv2.imread(file) for file in img_paths0]

img_paths1 = glob.glob('./../PoolTest/Imgs/Tennis/1080/Cam2/*.jpg')
img_paths1.sort(key = frameNr)
images_cam1 = [cv2.imread(file) for file in img_paths1]

img_paths2 = glob.glob('./../PoolTest/Imgs/Tennis/1080/Cam3/*.jpg')
img_paths2.sort(key = frameNr)
images_cam2 = [cv2.imread(file) for file in img_paths2]

for i in range(len(img_paths0)):
    images_cam0[i] = undistortPinHole(images_cam0[i], K0, D0)
    images_cam1[i] = undistortPinHole(images_cam1[i], K1, D1)
    images_cam2[i] = undistortPinHole(images_cam2[i], K2, D2)
    
print(K1)

frame = images_cam0[0]
h, w = frame.shape[:2]
K0, _ = cv2.getOptimalNewCameraMatrix(K0, D0, (w,h), 1, (w,h))

objTracker = ObjectTracking(frame,tracker_name, select_ROI, yolo_domination, bbox_slack= slack)
uv1 = []
center = objTracker.bbox.getCenter()
uv1.append(np.array([center[0], center[1]]))
cv2.namedWindow('window')
for i in range(1, len(images_cam0)):
    frame = images_cam0[i]
    objTracker.updateBbox(frame)
    objTracker.bbox.drawBbox(frame)
    objTracker.bbox.drawCenter(frame)
    center = objTracker.bbox.getCenter()
    uv1.append(np.array([center[0], center[1]]))
    cv2.imshow('window',frame)
    cv2.waitKey(10)

frame = images_cam1[0]
h, w = frame.shape[:2]
K1, _ = cv2.getOptimalNewCameraMatrix(K1, D1, (w,h), 1, (w,h))

objTracker = ObjectTracking(frame,tracker_name, select_ROI, yolo_domination, bbox_slack= slack)
uv2 = []
center = objTracker.bbox.getCenter()
uv2.append(np.array([center[0], center[1]]))
cv2.namedWindow('window')
for i in range(1, len(images_cam1)):
    frame = images_cam1[i]
    objTracker.updateBbox(frame)
    objTracker.bbox.drawBbox(frame)
    center = objTracker.bbox.getCenter()
    uv2.append(np.array([center[0], center[1]]))
    cv2.imshow('window',frame)
    cv2.waitKey(10)


frame = images_cam2[0]
h, w = frame.shape[:2]
K2, _ = cv2.getOptimalNewCameraMatrix(K2, D2, (w,h), 1, (w,h))

objTracker = ObjectTracking(frame,tracker_name, select_ROI, yolo_domination, bbox_slack= slack)
uv3 = []
center = objTracker.bbox.getCenter()
uv3.append(np.array([center[0], center[1]]))
cv2.namedWindow('window')
for i in range(1, len(images_cam2)):
    frame = images_cam2[i]
    objTracker.updateBbox(frame)
    objTracker.bbox.drawBbox(frame)
    center = objTracker.bbox.getCenter()
    uv3.append(np.array([center[0], center[1]]))
    cv2.imshow('window',frame)
    cv2.waitKey(10)

X01 = []
X02 = []
X12 = []

for i in range(len(uv1)):
    uvw_cam01 =  triangulateSinglePair(uv1[i], uv2[i], K0, K1, T0, T1)
    uvw_cam02 =  triangulateSinglePair(uv1[i], uv3[i], K0, K2, T0, T2)
    uvw_cam12 =  triangulateSinglePair(uv2[i], uv3[i], K1, K2, T1, T2)
    X01.append(uvw_cam01)
    X02.append(uvw_cam02)
    X12.append(uvw_cam12)

X01 = np.array(X01)
X01 = X01.reshape((X01.shape[0], 4))
X01 = X01.T
X02 = np.array(X02)
X02 = X02.reshape((X02.shape[0], 4))
X02 = X02.T
X12 = np.array(X12)
X12 = X12.reshape((X12.shape[0], 4))
X12 = X12.T

Ts = [T0, T1, T2]
Xs = [X01, X02, X12]

plot_results2(Ts, Xs, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,-1], vec_scale=2)