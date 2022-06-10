import numpy as np
from featureMatching import*
from generatePlots import*


def reprojection_error(uv_gt, uv_pred):
    error = uv_gt - uv_pred
    all_norms = []
    for i in range(error.shape[0]):
       norm =  np.sqrt(error[i,0]**2+error[i,1]**2)
       all_norms.append(norm)
    return np.sum(all_norms)/error.shape[0]

def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    A = np.empty((n, 9))
    for i in range(n):
        x1,y1 = xy1[:2,i]
        x2,y2 = xy2[:2,i]
        A[i,:] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

    _,_,VT = np.linalg.svd(A)
    return np.reshape(VT[-1,:], (3,3))

def get_F(E, K1, K2):
    K_inv1 = np.linalg.inv(K1)
    K_inv2 = np.linalg.inv(K2)
    F = K_inv2.T@E@K_inv1
    return F

def triangulate_all(xy1, xy2, P1, P2):
    
    n = xy1.shape[1]
    X = np.empty((4,n))
    for i in range(n):
        A = np.empty((4,4))
        A[0,:] = P1[0,:] - xy1[0,i]*P1[2,:]
        A[1,:] = P1[1,:] - xy1[1,i]*P1[2,:]
        A[2,:] = P2[0,:] - xy2[0,i]*P2[2,:]
        A[3,:] = P2[1,:] - xy2[1,i]*P2[2,:]
        U,s,VT = np.linalg.svd(A)
        X[:,i] = VT[3,:]/VT[3,3]
    return X

def R_t_2_T(R,t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def decompose(E):
    
    U,_,VT = np.linalg.svd(E)
    R90 = np.array([[0, -1, 0], [+1, 0, 0], [0, 0, 1]])
    R1 = U @ R90 @ VT
    R2 = U @ R90.T @ VT
    if np.linalg.det(R1) < 0: R1 = -R1
    if np.linalg.det(R2) < 0: R2 = -R2
    t1, t2 = U[:,2], -U[:,2]
    return [R_t_2_T(R1,t1), R_t_2_T(R1,t2), R_t_2_T(R2, t1), R_t_2_T(R2, t2)]

def determine_valid_decomposition(T4, xy1, xy2, metric_scale = 1):
    best_num_visible = 0
    for i, T in enumerate(T4):
        P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        P2 = T[:3,:]
        X1 = triangulate_all(xy1, xy2, P1, P2)*metric_scale
        X2 = T@X1
        num_visible = np.sum((X1[2,:] > 0) & (X2[2,:] > 0))
        if num_visible > best_num_visible:
            best_num_visible = num_visible
            best_T = T
            best_X1 = X1
    T = best_T
    X = best_X1
    T[:3, -1] *= metric_scale
    print('Best solution: %d/%d points visible' % (best_num_visible, xy1.shape[1]))
    return X, T

def eline(l, **args):
    
    lim = np.array([-1e8, +1e8])
    a,b,c = l
    if np.absolute(a) > np.absolute(b):
        x,y = -(c + b*lim)/a, lim
    else:
        x,y = lim, -(c + a*lim)/b
    plt.plot(x, y, **args)

def e_distance(F, uv1, uv2):
    
    n = uv1.shape[1]
    l2 = F@uv1
    l1 = F.T@uv2
    e = np.sum(uv2*l2, axis=0)
    norm1 = np.linalg.norm(l1[:2,:], axis=0)
    norm2 = np.linalg.norm(l2[:2,:], axis=0)
    return 0.5*e*(1/norm1 + 1/norm2)

def num_trials(sample_size, confidence, inlier_fraction):
    return int(np.log(1 - confidence)/np.log(1 - inlier_fraction**sample_size))

def RANSAC(xy1, xy2, K1, K2, distance_threshold, num_trials):
    uv1 = K1@xy1
    uv2 = K2@xy2
   
    best_num_inliers = -1
    print(num_trials)
    for i in range(num_trials):
        sample = np.random.choice(xy1.shape[1], size=8, replace=False)
        E_i = estimate_E(xy1[:,sample], xy2[:,sample])
        d_i = e_distance(get_F(E_i, K1, K2), uv1, uv2)
        inliers_i = np.absolute(d_i) < distance_threshold
        num_inliers_i = np.sum(inliers_i)
        if num_inliers_i > best_num_inliers:
            best_num_inliers = num_inliers_i
            E = E_i
            inliers = inliers_i
    return E, inliers

def findAllInitalPosesAnd3DPoints(images, frame_indices, Ks, allMatches, camIDs, metric_scale = 1, display = False):
   
    uv1s = []
    uv2s = []
    xy1s = []
    xy2s = []

    for i in range(len(allMatches)):
        uv1 = np.vstack([allMatches[i][:,:2].T, np.ones(allMatches[i].shape[0])])
        uv2 = np.vstack([allMatches[i][:,2:4].T, np.ones(allMatches[i].shape[0])])
        uv1s.append(uv1)
        uv2s.append(uv2)
        
        xy1 = np.linalg.inv(Ks[camIDs[i, 0]])@uv1
        xy2 = np.linalg.inv(Ks[camIDs[i, 1]])@uv2
        xy1s.append(xy1)
        xy2s.append(xy2)
      
  
    #Outlier rejection
    confidence = 0.99
    inlier_fraction = 0.50
    distance_threshold = 0.3
    num_trials = num_trials(8, confidence, inlier_fraction)
    #num_trials = 20000
    
    T0 = np.eye(4) #Pose of the base image
    allXs = []
    allTs = []
    allTs.append(T0)
    inlier_uv1s = []
    inlier_uv2s = []
    inlier_xy1s = []
    inlier_xy2s = []
    allInliers = []
    pointIDs = []

    for i in range(len(allMatches)):
        print("FINDING POSES AND 3D POINTS")
        E,inliers = RANSAC(xy1s[i], xy2s[i], Ks[camIDs[i,0]], Ks[camIDs[i,1]], distance_threshold, num_trials)
        uv1 = uv1s[i][:,inliers]
        uv2 = uv2s[i][:,inliers]
        xy1 = xy1s[i][:,inliers]
        xy2 = xy2s[i][:,inliers]
        
                                                                                            
        T4 = decompose(E)
        X, T = determine_valid_decomposition(T4, xy1, xy2, metric_scale)
        
        if display:
            print("Triangulating cam", frame_indices[i][0][0], " image",frame_indices[i][0][1], "and cam", frame_indices[i][1][0], "image", frame_indices[i][1][1] )
            I1 = images[frame_indices[i][0][0]][frame_indices[i][0][1]]
            I2 = images[frame_indices[i][1][0]][frame_indices[i][1][1]]
            draw_correspondences(I1, I2, uv1, uv2, get_F(E, Ks[camIDs[i,0]], Ks[camIDs[i,1]]), sample_size=8)
            draw_point_cloud(X, I2, uv2, xlim=[-4,+4], ylim=[-5,+2], zlim=[1,10])
            plt.show()
        
        X = X.T[:,:3]
        inlier_uv1s.append(uv1[:2].T)
        inlier_uv2s.append(uv2[:2].T)
        inlier_xy1s.append(xy1[:2].T)
        inlier_xy2s.append(xy2[:2].T)
        allTs.append(T)
        allXs.append(X)
        allInliers.append(inliers)
        pointIDs.append([np.arange(uv1[:2].T.shape[0]), np.arange(uv2[:2].T.shape[0])])

    allXs = np.array(allXs, dtype=object)
    allTs = np.array(allTs, dtype=object)
    inlier_uv1s = np.array(inlier_uv1s, dtype=object)
    inlier_uv2s = np.array(inlier_uv2s, dtype=object)
    pointIDs =  np.array(pointIDs, dtype = object)
    
    return allXs, allTs, inlier_uv1s, inlier_uv2s, pointIDs, allInliers