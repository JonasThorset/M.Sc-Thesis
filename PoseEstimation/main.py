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


if __name__ == "__main__":

    metric_scale = 1
    K = np.loadtxt("./BR/K_B1.txt")
    Ks = [K, K, K]
    D = np.zeros(4) #No distiortion in blender images apparently
    Ds = [D, D, D]

    frame11 = cv2.imread("./Blender/11.png")
    frame21 = cv2.imread("./Blender/21.png")
    frame31 = cv2.imread("./Blender/31.png")
    
    frame12 = cv2.imread("./Blender/12.png")
    frame22 = cv2.imread("./Blender/22.png")
    frame32 = cv2.imread("./Blender/32.png")
    
    #images = [[frame11, frame12], [frame21, frame22], [frame31, frame32]]
    images = [[frame12], [frame21], [frame31]]
    print("LOADING DONE")
    for i in range(len(images)):
        for j in range(len(images[i])):
            images[i][j] = cv2.cvtColor(images[i][j], cv2.COLOR_BGR2RGB)
            images[i][j] = calib.undistortPinHole(images[i][j], Ks[i], Ds[i])
    print("UNDISTORTION DONE")
      
    allMatches,  camIDs, frame_indices, kps, descriptors = matchAllAgainstReference(images, display = False)

    for i in range(len(images)):
            for j in range(len(images[i])):
                images[i][j] = images[i][j]/255.0
    allXs, allTs, uv1s, uv2s, pointIDs, allInliers = findAllInitalPosesAnd3DPoints(images, frame_indices, Ks, allMatches, camIDs, metric_scale, display = False)
     
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


    uv1s_reshaped = uv1s[0]
    allXs_initial = allXs[0]
    Ts_initial = allTs

    T0 = Ts_initial[0]
    T1 = Ts_initial[1]
    T2 = Ts_initial[2]

    metric_scale = 0
    metric_scale += abs(10.0/T1[0, 3])
    metric_scale += abs(15/T1[2, 3])
    metric_scale += abs(10.0/T2[0, 3])
    metric_scale += abs(15.0/T2[2, 3])
    metric_scale = metric_scale/4.0
    

    T0[:3, :3] = np.eye(3)
    T1[:3, :3] = T1[:3, :3].T
    T2[:3, :3] = T2[:3, :3].T
    T0[:3, 3] = T0[:3, 3] *metric_scale
    T1[:3, 3] = T1[:3, 3] *metric_scale
    T2[:3, 3] = T2[:3, 3] *metric_scale
    allTs2 = [T0, T1, T2]
    printPose(T0)
    printPose(T1)
    printPose(T2)

    Ts_initial = allTs2
    
    for i in range(1, len(uv1s)):
        uv1s_reshaped = np.vstack((uv1s_reshaped, uv1s[i]))
        allXs_initial = np.vstack((allXs_initial, allXs[i]))
  
    uv1s_reshaped = np.vstack((uv1s_reshaped.T, np.ones(uv1s_reshaped.shape[0])))
    allXs_initial = np.vstack((allXs_initial.T, np.ones(allXs_initial.shape[0])))

    #draw_poses_and_cloud_to_trix(Ts_initial, allXs_initial, images[0][0], uv1s_reshaped, xlim=[-1,+1], ylim=[-1,+1], zlim=[-1,+1], vec_scale=2)
    draw_poses_and_cloud_B(Ts_initial, allXs_initial*metric_scale, images[0][0], uv1s_reshaped,xlim=[-2*metric_scale,+2*metric_scale], ylim=[-2*metric_scale,+2*metric_scale], zlim=[-2*metric_scale,+2*metric_scale], vec_scale=2*metric_scale)

    #camera_params, points_3d, camera_indices, point_indices, points_2d = prepareBundleImageSequence(allXs, allTs, uv1s, uv2s, camIDs, frame_indices, Ks)
    camera_params, points_3d, camera_indices, point_indices, points_2d = prepareBundleAgainstFirstImage(allXs, allTs, uv1s, uv2s, camIDs, Ks, fts)   
    
    bundle = BA(camera_params, points_3d, points_2d, camera_indices, point_indices)

    uv_initial = bundle.project(points_3d[point_indices], camera_params[camera_indices])
    uv_gt = points_2d
    print("Initial reprojection error: ", reprojection_error(uv_gt,uv_initial))

    n_points = points_3d.shape[0]
    
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 6 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    params_refined, X_refined = bundle.bundleAdjust()

    #3D-2D projection back to the images to calulate the reprojection error
    uv_initial = bundle.project(points_3d[point_indices], camera_params[camera_indices])
    uv_hat = bundle.project(X_refined[point_indices], params_refined[camera_indices])

    print("Post-bundle reprojection error: ", reprojection_error(uv_gt,uv_hat))
        
   #Make homogenious
    X_refined = np.vstack((X_refined.T, np.ones(X_refined.shape[0])))
    allXs_refined = np.vstack((X_refined.T, np.ones(X_refined.shape[0])))

    extrinsics_initial = camera_params[:, :6]
    extrinsics_refined = params_refined[:, :6]
    
    Ts_refined = bundle_params2T(extrinsics_refined)

    
    np.savetxt("T0_refined.txt", Ts_refined[0])
    np.savetxt("T1_refined.txt", Ts_refined[1])
    np.savetxt("T2_refined.txt", Ts_refined[2])

    metric_scale = 0
    metric_scale += abs(10.0/T1[0, 3])
    metric_scale += abs(15.0/T1[2, 3])
    metric_scale += abs(10.0/T2[0, 3])
    metric_scale += abs(15.0/T2[2, 3])
    metric_scale = metric_scale/4.0
    
    T0 = Ts_refined[0]
    T1 = Ts_refined[1]
    T2 = Ts_refined[2]
    T0[:3, :3] = np.eye(3)
    T1[:3, :3] = T1[:3, :3].T
    T2[:3, :3] = T2[:3, :3].T
    T0[:3, 3] = T0[:3, 3] *metric_scale
    T1[:3, 3] = T1[:3, 3] *metric_scale
    T2[:3, 3] = T2[:3, 3] *metric_scale
    Ts_refined = [T0, T1, T2]
    printPose(T0)
    printPose(T1)
    printPose(T2)

    draw_poses_and_cloud_B(Ts_refined, X_refined*metric_scale, images[0][0], uv1s_reshaped, xlim=[-2*metric_scale,+2*metric_scale], ylim=[-2*metric_scale,+2*metric_scale], zlim=[-2*metric_scale,2*metric_scale], vec_scale=2*metric_scale)
    
        

    

    