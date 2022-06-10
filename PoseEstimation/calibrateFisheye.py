import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def findCorners(imgPaths, boardDim, display = False):
    assert(int(cv2.__version__[0]) >= 3) #'The fisheye module requires opencv version >= 3.0.0'
    #print(cv2.__version__)
    
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    objp = np.zeros((1, boardDim[0]*boardDim[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:boardDim[0], 0:boardDim[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in imgPaths:
        print(fname)
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, boardDim, flags = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            searchWindow = (3, 3)        #Tune this to achieve lower re-projection error
            corners2 = cv2.cornerSubPix(gray,corners, searchWindow,(-1,-1),subpix_criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, boardDim, corners2, ret)
            if display:
                cv2.imshow(fname, img)
                cv2.waitKey(100000)
    cv2.destroyAllWindows()
    shapeImages = gray.shape
    return shapeImages, objpoints, imgpoints

def calibrate(objpoints, imgpoints, grayShape):
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW +cv2.fisheye.CALIB_CHECK_COND
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, mtx, dist, rotvecs, transvecs = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            grayShape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    return rms, mtx, dist, rotvecs, transvecs


def reprojectionError(objpoints, imgpoints, rvecs, tvecs, K, D):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "Mean reprojection error all images: {}".format(mean_error/len(objpoints)))


def reprojectionErrorHistogram(imagePaths, objpoints, imgpoints, rvecs, tvecs, K, D):
    error = []
    frames = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error.append(cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2))
    for image in imagePaths:
        frameName = image.split("_frame")
        frameName = frameName[1].split(".")[0]
        frames.append(frameName)

       
    plt.bar(frames, error)
    plt.show()


def visualize_reprojection(imgDim, imgpoints, objpoints, rvecs, tvecs, mtx, dist, coverage = False):
    imgpoints = np.array(imgpoints)
    print("imgpoints.shape: ", imgpoints.shape)
    num_imgs = imgpoints.shape[0]
    imgpoints = imgpoints.reshape((imgpoints.shape[0],imgpoints.shape[1],2))
    uv_pred = []
    for i, (rvec,tvec) in enumerate(zip(rvecs,tvecs)):
        uv,_ = cv2.projectPoints(objpoints[0], rvec, tvec, mtx, dist)
        uv_pred.append(uv)
    uv_pred = np.array(uv_pred).reshape((imgpoints.shape[0],imgpoints.shape[1],2))

    rep_error = []

    fig, axs = plt.subplots(2)

    for i, uv_p in enumerate (uv_pred):
        #error = imgpoints[i]-uv_p
        #error_norm = np.sqrt(error[:,0]**2+error[:,1]**2)
        error_norm = cv2.norm(imgpoints[i], uv_p, cv2.NORM_L2)/len(uv_p)
        rep_error.append(np.mean(error_norm))

        axs[1].scatter(imgpoints[i][:,0]-uv_p[:,0],imgpoints[i][:,1]-uv_p[:,1],c="b",marker="x")
    axs[1].set_title("Reprojection error")


    axs[0].bar(np.arange(num_imgs), rep_error)
    axs[0].set_xticks(np.arange(0,num_imgs,5))

    axs[0].set_title("Mean reprojection error per image")
    plt.show()
    
    if (coverage):
        for x in imgpoints:
            plt.scatter(x[:,0], x[:,1])
            plt.title("Checkerboard corner coverage")
            plt.xlim([0,imgDim[1]])
            plt.ylim(([0,imgDim[0]]))
            plt.gca().invert_yaxis()
        plt.show()




if __name__ == '__main__':
    #SET PARAMS BEFORE CALIBRATING. EXISTING CAMERA INTRINSICS MAY BE OVERWRITTEN!!!
    savePath = './Results/'
    images = glob.glob('*.jpg')
    medium = "air"          #Either air or water
    imgDim = cv2.imread(images[0]).shape[:2]
    CHECKERBOARD = (10,7)

    grayShape, objpoints, imgpoints = findCorners(images, CHECKERBOARD, display = False)
    rms, K, D, rvecs, tvecs= calibrate(objpoints, imgpoints, grayShape)
    
    reprojectionError(objpoints, imgpoints, rvecs, tvecs, K, D)
    reprojectionErrorHistogram(images, objpoints, imgpoints, rvecs, tvecs, K, D)
    visualize_reprojection(imgDim, imgpoints, objpoints, rvecs, tvecs, K, D, coverage = True)

    #Save data to files
    np.savetxt(savePath + "K_"+medium+".txt", K)
    np.savetxt(savePath + "D_"+medium+".txt", D)

    rvecs = np.asarray(rvecs)
    tvecs = np.asarray(tvecs)
    objpoints = np.asarray(objpoints)
    imgpoints = np.asarray(imgpoints)
    print(objpoints.shape)
    print(imgpoints.shape)
    objpoints.reshape(objpoints.shape[0], objpoints.shape[2], objpoints.shape[3]) #shape(n_images, 1, n_points, xyz_point)
    imgpoints.reshape(imgpoints.shape[0], imgpoints.shape[1], imgpoints.shape[3]) #shape(n_images, n_points, 1, xy_point) yes it's weird that it doesn't fit the shape of objpoints
    print(objpoints.shape)
    print(imgpoints.shape)
    np.save(savePath + "objpoints.npy", objpoints)
    np.save(savePath + "rvecs.npy", rvecs.reshape(rvecs.shape[0], rvecs.shape[1]))
    np.save(savePath + "tvecs.npy", tvecs.reshape(tvecs.shape[0], tvecs.shape[1]))
    np.save(savePath + "objpoints.npy", objpoints)
    np.save(savePath + "imgpoints.npy", imgpoints)