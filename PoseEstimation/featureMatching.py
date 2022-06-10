import matplotlib.pyplot as plt
import numpy as np
import cv2


def SIFT_match_with_reference(frame_ref, des_ref, kp_ref,    frame2, displayMatches = False):
    des1 = des_ref
    kp1 = kp_ref
    frame1 = frame_ref  #only used for displaying matches
    #Convert to grayscale
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(frame2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    #Apply ratio test
    good = []
    match_pt = []
    des_img1 = []
    des_img2 = []
    kp_img1 = []
    kp_img2 = []
    
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            match_pt.append([kp1[m.queryIdx].pt, kp2[m.trainIdx].pt])
            des_img1.append(des1[m.queryIdx])
            des_img2.append(des2[m.trainIdx])
            kp_img1.append(kp1[m.queryIdx])
            kp_img2.append(kp2[m.trainIdx])
            
    match_pt = np.array(match_pt).reshape(len(match_pt),4)
    
    if displayMatches:
        frame3 = cv2.drawMatchesKnn(frame1,kp1,frame2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(frame3),plt.show()
    des1 = np.array(des_img1)
    des2 = np.array(des_img2)
    kp1 = np.array(kp_img1)
    kp2 = np.array(kp_img2)

    return match_pt, kp1, des1, kp2, des2



def matchAllAgainstReference(images, display = False):
    allMatches = []
    camIDs = []
    frame_indices = []
    descriptors = []
    kps = []
    img2Idx = 1
    
    frame_ref = cv2.cvtColor(images[0][0], cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(frame_ref,None)
    for i in range(len(images)):
        for j in range(len(images[i])):
            if i == 0 and j == 0: continue    #To avoid matching the reference image with itself
            print("FINDING MATCHES CAM ", 0, " img", 0, ", AND CAM", i , " img" , j)
            matches, kp1, des1, kp2, des2 = SIFT_match_with_reference(frame_ref, des_ref, kp_ref, images[i][j], displayMatches=display)
            allMatches.append(matches)
            camIDs.append([0, i])
            frame_indices.append([0, img2Idx])
            if img2Idx == 1:
                kps.append(kp1)
                descriptors.append(des1)
            kps.append(kp2)
            descriptors.append(des2)
            img2Idx += 1
    camIDs = np.array(camIDs)
    frame_indices = np.array(frame_indices)
 
    return allMatches, camIDs, frame_indices, kps, descriptors



def findMatchesImageSequence(images, display = False):
    
    allMatches = []
    camIDs = []

    frame1Idx = 0
    frame2Idx = 1
    frame_indices = []

    query_img = images[0][0]
    sift = cv2.SIFT_create()
    kp_query, des_query = sift.detectAndCompute(query_img,None)
    for i in range(len(images)):
        for j in range(len(images[i])):
            cam1Idx = i
            cam2Idx = i
            img1Idx = j
            img2Idx = j+1
            if j == len(images[i]) - 1 and i == len(images) - 1:
                break
            elif j == len(images[i]) -1 and i < len(images) -1:
                cam2Idx = i+1
                img2Idx = 0
            print("FINDING MATCHES BETWEEN", cam1Idx, img1Idx, "AND", cam2Idx, img2Idx, " (frame idx", frame1Idx, "and", frame2Idx,")")
            matches, kp2, des2 = SIFT_match_with_reference(query_img, des_query, kp_query,   images[cam2Idx][img2Idx], displayMatches= display)
            kp_query, des_query = kp2, des2
            allMatches.append(matches)
            camIDs.append([cam1Idx, cam2Idx])
            frame_indices.append([frame1Idx, frame2Idx])
            frame1Idx +=1
            frame2Idx +=1
    return allMatches, np.array(camIDs), np.array(frame_indices)




def getInlierPtsAndIndices(descriptors, kps, allInliers):
    
    descriptors[0] = descriptors[0][allInliers[0], :]
    kps[0] = kps[0][allInliers[0]]
    for i in range(1, len(descriptors)):
        descriptors[i] = descriptors[i][allInliers[i-1], :]
        kps[i] = kps[i][allInliers[i-1]]
    
    allPts = []
    allIndices = []
    kp_query = kps[0]
    des_query = descriptors[0]
    for i in range(1, len(descriptors)):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_query,descriptors[i],k=2)
        kp2 = kps[i]
        pts = [] 
        idx = [] 
        j=0
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                pts.append(kp2[m.trainIdx].pt)
                idx.append([j, m.trainIdx])
                j+=1
        pts = np.array(pts).reshape(len(pts),2)
        idx = np.array(idx).reshape((len(idx), 2))
        allPts.append(pts)
        allIndices.append(idx)

    return allPts, allIndices