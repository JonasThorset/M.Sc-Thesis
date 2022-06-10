import numpy as np
import matplotlib.pyplot as plt
from rotationUtils import*
from utils import euclideanDistance


def plot_results(Ts, Ts_gt, Xs, xlim, ylim, zlim, vec_scale = 1):
    
    plt.figure('Tracking results', figsize=(10,10))
    ax = plt.axes(projection='3d')

    for i in range(len(Xs)):
        if i == 0:
            color = 'black'
            lbl = "Ground truth"
        if i == 1:
            color = "red"
            lbl = "Cam0-Cam1"
        if i == 2:
            color = "blue"
            lbl = "Cam0-Cam2"
        if i == 3:
            color = "green"
            lbl = "Cam1-Cam2"
        ax.scatter(Xs[i][0,:], -Xs[i][2,:], Xs[i][1,:], c= color, marker='.', depthshade=False, label = lbl)
    
    pt = 3
    gt_pt = Xs[0][:3, pt]
    cam1_pt = Xs[1][:3, pt]
    cam2_pt = Xs[2][:3, pt]
    x, y, z = [gt_pt[0], cam1_pt[0]], [gt_pt[1], cam1_pt[1]], [gt_pt[2], cam1_pt[2]]
    x2, y2, z2 = [gt_pt[0], cam2_pt[0]], [gt_pt[1], cam2_pt[1]], [gt_pt[2], cam2_pt[2]]
    #ax.plot(x,y,z, c = "purple")
    #x.plot(x2,y2,z2, c = "orange")

    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim(ylim)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.legend()
    
    plt.show()


def plot_results2(Ts, Xs, xlim, ylim, zlim, vec_scale = 1):

    plt.figure('Tracking results', figsize=(6,6))
    ax = plt.axes(projection='3d')

    for i in range(len(Xs)):
        if i == 0:
            color = "red"
            lbl = "Cam0-Cam1"
        if i == 1:
            color = "blue"
            lbl = "Cam0-Cam2"
        if i == 2:
            color = "green"
            lbl = "Cam1-Cam2"
        ax.scatter(Xs[i][0,:], Xs[i][2,:], Xs[i][1,:], c= color, marker='.', depthshade=False, label = lbl)
    
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim(ylim)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
        
    euler = np.array([16.0, 0.0, 0.0]) #The ground truth should be rotated according to the pitch tilt of the sensing rig
    print("Rad: ",deg2Rad(euler[0]))
    euler[0] = deg2Rad(euler[0])
    R_tilt = euler2R(euler)
    print(R_tilt)

    p0 = np.array([0.25, -0.12, 1.25])
    d0 = R_tilt @ np.array([-0.5, 0.0 , 0.0])
    p1 = p0 + d0
    d1 = R_tilt @ np.array([0.0, 0.0, 1.0])
    p2 = p1 + d1
    d2 = R_tilt @ np.array([0.0, -0.58, 0.0])

    p_gt0 = [p0.copy()]
    p_gt1 = [p1.copy()]
    p_gt2 = [p2.copy()]
    
    for j in range(100):
        p_gt0.append(p_gt0[-1] + d0/100.0)
        p_gt1.append(p_gt1[-1] + d1/100.0)
        p_gt2.append(p_gt2[-1] + d2/100.0)
            
    p_gt = np.array(p_gt0 + p_gt1 + p_gt2)
    
    dist = [[], [], []]
    
    for i in range(len(Xs)):
        for j in range(Xs[i].shape[1]):
            shortest_dist = 1000
            for k in range(p_gt.shape[0]):
                shortest_dist = min(shortest_dist, euclideanDistance(p_gt[k,:], Xs[i][:3, j]))
            dist[i].append(shortest_dist)
    
    e1 = np.array(dist[0])
    e2 = np.array(dist[1])
    e3 = np.array(dist[2])
    _, ax2 = plt.subplots()
    ax2.plot(e1, color = "red", label = "Error Cam0-Cam1")
    ax2.plot(e2, color = "blue", label = "Error Cam0-Cam2")
    ax2.plot(e3, color = "green", label = "Error Cam1-Cam2")
    ax2.set_xlabel("Frame number")
    ax2.set_ylabel("Error -Euclidean Distance from ground truth [m]")
    ax2.legend()


    print(p0, p1)

    ax.quiver(p0[0], p0[2], p0[1], d0[0], d0[2], d0[1],  color = "black")
    ax.quiver(p1[0], p1[2], p1[1], d1[0], d1[2], d1[1],  color = "black")
    ax.quiver(p2[0], p2[2], p2[1], d2[0], d2[2], d2[1],  color = "black", label = "Ground truth")
    
    ax.legend()
   
    plt.show()



