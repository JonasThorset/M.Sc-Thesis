import numpy as np
import cv2
import glob
from Detector import*
from Bbox import*
from Tracker import*
from plotResults import*
from triangulateSinglePair import*
from utils import euclideanDistance

#Comments:
#CSRT worsk perfectly(with and without yolo updates)

select_ROI = 'manually' #can be either 'yolo' or 'manually'
tracker_name = 'CSRT'# Can be 'KCF, 'BOOSTING', 'MIL', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE' or 'CSRT'
slack = 10 #How much you want to expand the yolobbox with (pixels)
yolo_domination = False


#T0 = txt2np("./Cam0/data/T0_post_bundle.txt")
#T1 = txt2np("./Cam1/data/T1_post_bundle.txt")
#T2 = txt2np("./Cam2/data/T2_post_bundle.txt")
T0 = np.loadtxt("./Cams/Blender/Cam0/data/T0_refined.txt")
T1 = np.loadtxt("./Cams/Blender/Cam1/data/T1_refined.txt")
T2 = np.loadtxt("./Cams/Blender/Cam2/data/T2_refined.txt")

#GT cameras
T0_gt = np.loadtxt("./Cams/Blender/Cam0/data/T0b_gt.txt")
T1_gt = np.loadtxt("./Cams/Blender/Cam1/data/T1b_gt.txt")
T2_gt = np.loadtxt("./Cams/Blender/Cam2/data/T2b_gt.txt")
Ts_gt = [T0_gt, T1_gt, T2_gt] #Extracted ground truth

downscale =  1/10.0

scale = T1_gt[0,3] / T1[0,3] * downscale
T0[:3, 3] *= scale
T1[:3, 3] *= scale
T2[:3, 3] *= scale

K0 = np.loadtxt("./Cams/Blender/Cam0/data/K_B0.txt")
K1 = np.loadtxt("./Cams/Blender/Cam1/data/K_B1.txt")
K2 = np.loadtxt("./Cams/Blender/Cam2/data/K_B2.txt")

  
images_cam0 = [cv2.imread(file) for file in glob.glob('./Cams/Blender/Cam0/images/Pellet_05_density/*.png')]
images_cam1 = [cv2.imread(file) for file in glob.glob('./Cams/Blender/Cam1/images//Pellet_05_density/*.png')]
images_cam2 = [cv2.imread(file) for file in glob.glob('./Cams/Blender/Cam2/images/Pellet_05_density/*.png')]

#images_cam0 += [cv2.imread(file) for file in glob.glob('./Cam0/images/xrot/*.png')]
#images_cam1 += [cv2.imread(file) for file in glob.glob('./Cam1/images//xrot/*.png')]
#images_cam2 += [cv2.imread(file) for file in glob.glob('./Cam2/images/xrot/*.png')]

frame = images_cam0[0]
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
 
Xgt = np.loadtxt("gt_pellet_xzy.txt")[1:, :]

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

print(Xgt[0,:3])
Xgt = np.vstack((Xgt.T, np.ones(Xgt.shape[0])))

#Transformation of the extracted ground truth from w to c1
T_w2c = np.hstack((euler2R(np.array([deg2Rad(0.0), deg2Rad(0.0), deg2Rad(0.0)])), np.array([0.0, 25.0, 0.0]).reshape((3,1))))
T_w2c = np.vstack((T_w2c, np.array([0,0,0,1])))

Xgt = np.linalg.inv(T_w2c) @ Xgt

T_c2origin = np.hstack((euler2R(np.array([deg2Rad(0.0), deg2Rad(0.0), deg2Rad(0.0)])), np.array(-Xgt[:3,0]).reshape((3,1))))
T_c2origin = np.vstack((T_c2origin, np.array([0,0,0,1])))

Xgt = T_c2origin @ Xgt

T_rot = np.hstack((euler2R(np.array([deg2Rad(-10.0), deg2Rad(0.0), deg2Rad(0.0)])), np.array([0,0,0]).reshape((3,1))))
T_rot = np.vstack((T_rot, np.array([0,0,0,1])))

Xgt = np.linalg.inv(T_c2origin) @ T_rot @ Xgt

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
print(T1[:, -1])

print(Xgt.shape)
Xs = [Xgt, X01, X02, X12]

e1 = []
e2 = []
e3 = []



d_target_2_cam0 = euclideanDistance(np.array([0.0, -0.4, 3.3]), [0.0, 0.0, 0.0])
d_target_2_cam1 = euclideanDistance(np.array([0.0, 0.0, 3.9]), [1.0, 0.0, 1.5])
d_target_2_cam2 = euclideanDistance(np.array([0.0, 0.1, 4.1]), [-1.0, 0.0, 1.5])

print("Sphere pos to cam0:", d_target_2_cam0)
print("Sphere pos to cam1:", d_target_2_cam1)
print("Sphere pos to cam2:", d_target_2_cam2)

for i in range(Xgt.shape[1]):
    e1.append(euclideanDistance(Xgt[:, i], X01[:,i]))
    e2.append(euclideanDistance(Xgt[:, i], X02[:,i]))
    e3.append(euclideanDistance(Xgt[:, i], X12[:,i]))
lim_scale = scale

fig, ax = plt.subplots()
ax.plot(e1, color = "red", label = "Error Cam0-Cam1")
ax.plot(e2, color = "blue", label = "Error Cam0-Cam2")
ax.plot(e3, color = "green", label = "Error Cam1-Cam2")
ax.set_xlabel("Frame number")
ax.set_ylabel("Error -Euclidean Distance from ground truth [m]")
ax.legend()

plot_results(Ts, Ts_gt, Xs, xlim=[-1*lim_scale,+1*lim_scale], ylim=[-1*lim_scale,+1*lim_scale], zlim=[-1*lim_scale,+1*lim_scale], vec_scale=30)
