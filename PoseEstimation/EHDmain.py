import cv2
import numpy as np
import matplotlib.pyplot as plt
from generatePlots import*
import undistort as calib
from rotationUtils import*
from bundleAdjustment import*
from featureMatching import*
from triangulate import*
from generatePlots import*
from featureTracks import*
from random import randint

 



if __name__ == "__main__":
    plottingOnly = False
    
    K1 = np.loadtxt("./../CalibData/K1080_EHD1_air.txt")
    K2 = np.loadtxt("./../CalibData/K1080_EHD2_air.txt")
    K3 = np.loadtxt("./../CalibData/K1080_EHD3_air.txt")

    D1 = np.loadtxt("./../CalibData/D1080_EHD1_air.txt")
    D2 = np.loadtxt("./../CalibData/D1080_EHD2_air.txt")
    D3 = np.loadtxt("./../CalibData/D1080_EHD3_air.txt")
    
    Ks = [K1, K2, K3]
    Ds = [D1, D2, D3]
    
    frame11 = cv2.imread("./../PoolTest/Door/Cam1/21.jpg")
    frame21 = cv2.imread("./../PoolTest/Door/Cam2/22.jpg")
    frame31 = cv2.imread("./../PoolTest/Door/Cam3/23.jpg")
    
    images = [[frame11], [frame21], [frame31]]
    print("LOADING DONE")

    for i in range(len(images)):
        for j in range(len(images[i])):
            images[i][j] = calib.undistortFisheye(images[i][j], Ks[i], Ds[i])
            cv2.imwrite("./../PoolTest/Door/undistorted/" +str(i) +str(j) + ".jpg",images[i][j])
            images[i][j] = cv2.cvtColor(images[i][j], cv2.COLOR_BGR2RGB)
    print("UNDISTORTION DONE")

    if plottingOnly:
        for i in range(len(images)):
                for j in range(len(images[i])):
                    images[i][j] = images[i][j]/255.0

    if not plottingOnly:
        allMatches,  camIDs, frame_indices, kps, descriptors = matchAllAgainstReference(images, display = False)

        for i in range(len(images)):
                for j in range(len(images[i])):
                    images[i][j] = images[i][j]/255.0

        allXs, allTs, uv1s, uv2s, pointIDs, allInliers = findAllInitalPosesAnd3DPoints(images, frame_indices, Ks, allMatches, camIDs, metric_scale = 1, display = False)
        
        allXs, uv1s, uv2s = deleteDuplicatePoints(allXs, uv1s, uv2s)
        fts = makeAllPtsFeatureTracks(allXs, uv1s)
        print("Number of fts:", len(fts))
        fts = mergeAllCorrespondences(fts)
        print("Number of fts:", len(fts))
        fts = deleteNonFts(fts)
        print("Number of fts:", len(fts))
        fts = makeFtsUnique(fts)
        print("Number of fts: ", len(fts))

        scale_factors = findScalingFactors(fts, allXs)
        print(scale_factors)
        
        for i in range(len(allXs)):
            allXs[i]  = allXs[i] * scale_factors[i]
            allTs[i+1][:3, -1] = allTs[i+1][:3, -1] * scale_factors[i]
        
        uv1s_reshaped = uv1s[0].copy()
        allXs_initial = allXs[0].copy()

        metric_scale = 0
        T0 = allTs[0].copy()
        T1 = allTs[1].copy()
        T2 = allTs[2].copy()
        metric_scale += abs(0.90/T1[0, 3])
        metric_scale += abs(0.90/T2[0, 3])
        metric_scale = metric_scale/2
        
        T0[:3, :3] = np.eye(3)
        T0[:3, 3] = T0[:3, 3] *metric_scale
        T1[:3, 3] = T1[:3, 3] *metric_scale
        T2[:3, 3] = T2[:3, 3] *metric_scale
        allTs2 = [T0, T1, T2]
    
        Ts_initial = allTs2
        
        for i in range(1, len(uv1s)):
            uv1s_reshaped = np.vstack((uv1s_reshaped, uv1s[i].copy()))
            allXs_initial = np.vstack((allXs_initial, allXs[i].copy()))
    
        uv1s_reshaped = np.vstack((uv1s_reshaped.T, np.ones(uv1s_reshaped.shape[0])))
        allXs_initial = np.vstack((allXs_initial.T, np.ones(allXs_initial.shape[0])))
    
    draw_poses_and_cloud(Ts_initial, allXs_initial*(metric_scale/3.0), images[0][0], uv1s_reshaped,xlim=[-1*metric_scale,+1*metric_scale], ylim=[-1*metric_scale,+1*metric_scale], zlim=[-1*metric_scale,1*metric_scale], vec_scale=4*metric_scale)

    if not plottingOnly:
        camera_params, points_3d, camera_indices, point_indices, points_2d = prepareBundleAgainstFirstImage(allXs, allTs, uv1s, uv2s, camIDs, Ks, fts)   
        
        bundle = BA(camera_params, points_3d, points_2d, camera_indices, point_indices)

        uv_initial = bundle.project(points_3d[point_indices], camera_params[camera_indices])
        uv_gt = points_2d
        print("Initial reprojection error: ", reprojection_error(uv_gt,uv_initial))

        n_points = points_3d.shape[0]
        params_refined, X_refined = bundle.bundleAdjust()

        # 3D-2D projection back to the images to calulate the reprojection error
        uv_initial = bundle.project(points_3d[point_indices], camera_params[camera_indices])
        uv_hat = bundle.project(X_refined[point_indices], params_refined[camera_indices])

        print("Post-bundle reprojection error: ", reprojection_error(uv_gt,uv_hat))

    #Make homogenious
        X_refined = np.vstack((X_refined.T, np.ones(X_refined.shape[0])))
        
        extrinsics_initial = camera_params[:, :6]
        extrinsics_refined = params_refined[:, :6]
        
        Ts_refined = bundle_params2T(extrinsics_refined)

    Ts_refined = [0,0,0]
    Ts_refined[0] = np.loadtxt("./Results/EHD/T0_refined.txt")
    T0 = Ts_refined[0]
    Ts_refined[0] = np.linalg.inv(T0) @ Ts_refined[0]
    Ts_refined[1] = np.linalg.inv(T0) @ np.loadtxt("./Results/EHD/T1_refined.txt")
    Ts_refined[2] = np.linalg.inv(T0) @ np.loadtxt("./Results/EHD/T2_refined.txt")
    X_refined = np.loadtxt("./Results/EHD/X_refined.txt")
    

    T0 = Ts_refined[0].copy()
    T1 = Ts_refined[1].copy()
    T2 = Ts_refined[2].copy()
    metric_scale += abs(0.90/T1[0, 3])
    metric_scale += abs(0.90/T2[0, 3])
    metric_scale = metric_scale/2
    allTs_refined = [T0, T1, T2]
    print("##############################")
    printPose(allTs_refined[0])
    printPose(allTs_refined[1])
    printPose(allTs_refined[2])

    t0_post = allTs_refined[0][:3, -1]
    t1_post = allTs_refined[1][:3, -1]
    t2_post = allTs_refined[2][:3, -1]

    t1_gt = np.array([0.90, 0.00, 0.50])
    err_init1 = abs(t1_post) -t1_gt
    err_init2 = abs(t2_post) -t1_gt
    
    print(err_init1 , err_init2)

    draw_poses_and_cloud(allTs_refined, allXs_initial*metric_scale, images[0][0], uv1s_reshaped, xlim=[-1*metric_scale,+1*metric_scale], ylim=[-1*metric_scale,+1*metric_scale], zlim=[-1*metric_scale,1*metric_scale], vec_scale=4*metric_scale)
    
    Ts_refined[0] =  np.loadtxt("./Results/EHD/T0_initial.txt")
    Ts_refined[1] =  np.loadtxt("./Results/EHD/T1_initial.txt")
    Ts_refined[2] =  np.loadtxt("./Results/EHD/T2_initial.txt")
    X_refined = np.loadtxt("./Results/EHD/X_refined.txt")
    
    metric_scale2 = 0
    T0 = Ts_refined[0].copy()
    T1 = Ts_refined[1].copy()
    T2 = Ts_refined[2].copy()
    metric_scale2 += abs(0.90/T1[0, 3])
    metric_scale2 += abs(0.90/T2[0, 3])
    metric_scale2 = metric_scale2/2
    allTs_refined = [T0, T1, T2]
    
    t0_init = allTs_refined[0][:3, -1]
    t1_init = allTs_refined[1][:3, -1]
    t2_init = allTs_refined[2][:3, -1]

    t1_gt = np.array([0.90, 0.00, 0.50])
    err_init1 = abs(abs(t1_init) -t1_gt)
    err_init2 = abs(abs(t2_init) -t1_gt)
    
    print(err_init1 , err_init2)

    draw_poses_and_cloud(allTs_refined, X_refined*(metric_scale/3.0), images[0][0], uv1s_reshaped, xlim=[-1*metric_scale,+1*metric_scale], ylim=[-1*metric_scale,+1*metric_scale], zlim=[-1*metric_scale,1*metric_scale], vec_scale=4*metric_scale)
    