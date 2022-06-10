import numpy as np
import cv2
import sys




def undistort(img, K, D):
    h, w = img.shape[:2]
    DIM = (w, h)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def undistortFisheye(img, K, D):
    h, w = img.shape[:2]
    DIM = (w, h)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

if __name__ == '__main__':
    frame = cv2.imread('./Images/CalibImages/Cam1/Cam1_frame4.jpg')
    K = np.loadtxt('./Images/CalibImages/Cam1/K_air.txt')
    D = np.loadtxt('./Images/CalibImages/Cam1/D_air.txt')
    dst = undistort(frame, K, D)

    cv2.imshow("Testframe", dst)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()                                                                                     