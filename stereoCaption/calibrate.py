import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.rcParams.update({'font.size': 10})



def findCheckerboardCorners(imgFolderPath, boardDim, display = False):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((boardDim[0] * boardDim[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:boardDim[0],0:boardDim[1]].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(imgFolderPath + '*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, ChessboardDimension, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, ChessboardDimension, corners2, ret)
            if display:
                cv2.imshow('img', img)
                cv2.waitKey(100000)
    cv2.destroyAllWindows()
    shapeImages = gray.shape
    return shapeImages, objpoints, imgpoints



def calibrate(objpoints, imgpoints, grayShape):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayShape[::-1], None, None)
    print("\n\nOverall RMS re-projection error: \n", ret)
    return  mtx, dist, rvecs, tvecs

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
    plt.xlabel("Image Number")
    plt.ylabel("Mean Reprojection Error")
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
    
   
    plt.show()
    
    if (coverage):
        for x in imgpoints:
            plt.scatter(x[:,0], x[:,1])
            plt.xlim([0,imgDim[0]])
            plt.ylim(([0,imgDim[1]]))
            plt.gca().invert_yaxis()
            plt.xlabel("Image width")
            plt.ylabel("Image height")
        plt.show()

def undistort(img, K, D):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    print("new K: \n ", newcameramtx)
    print("D: \n",D)
    # undistort
    dst = cv2.undistort(img, K, D, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst


def reprojectionError(objpoints, imgpoints, rvecs, tvecs, K, D):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)))

if __name__ == "__main__":
    
    imgFolderPath = './Images/CalibImages2/Cam3/Water/1080/'
    images = glob.glob(imgFolderPath + '*.jpg')
    ChessboardDimension = (16, 7)   #(Width, height) remember zero indexing, i.e. first squre is the 0'th square
    testImg = cv2.imread(imgFolderPath+ "Cam3_frame0.jpg")
    frameH, frameW = testImg.shape[:2]
    frameDim = (frameW, frameH)
    print(frameDim)

    shapeImages, objpoints, imgpoints= findCheckerboardCorners(imgFolderPath, ChessboardDimension, display = False)
    K, D, rvecs, tvecs = calibrate(objpoints, imgpoints, shapeImages)
    reprojectionErrorHistogram(images, objpoints, imgpoints, rvecs, tvecs, K, D)
    visualize_reprojection(frameDim, imgpoints, objpoints, rvecs, tvecs, K, D, coverage = True)
    undistortedImg = undistort(testImg, K, D)
    reprojectionError(objpoints, imgpoints, rvecs, tvecs, K, D)
    
    print("New Shape: ", undistortedImg.shape)
    cv2.imwrite('distimage_water.png', undistortedImg)
    cv2.imwrite('calibresult_water.png', undistortedImg)
    #np.savetxt("./Images/CalibImages/Iphone/K_Iph.txt", K)
    #np.savetxt("./Images/CalibImages/Iphone/D_Iph.txt", D)
 