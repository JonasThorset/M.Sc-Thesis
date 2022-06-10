import numpy as np
from random import randint

class featureTrack:
    def __init__(self, uv1, X, match_idx, pt_idx):
        self.uv1 = uv1
        self.X = X
        self.pt_idx = pt_idx
        self.match_idx = match_idx
        self.ID = (match_idx, pt_idx)
        self.duplicate_match_indices = []
        self.duplicate_pt_indices = []
        self.duplicate_IDs = []
        self.duplicate_match_indices.append(match_idx)
        self.duplicate_pt_indices.append(pt_idx)
        self.duplicate_IDs.append(self.ID)

    def isFeatureTrack(self):
        return len(self.duplicate_pt_indices) > 1
    
    def mergeCorrespondences(self, featureTracks):
        for i in range(len(featureTracks)):
            if (self.uv1 == featureTracks[i].uv1).all() and (self.ID != featureTracks[i].ID):
                self.duplicate_match_indices.append(featureTracks[i].match_idx)
                self.duplicate_pt_indices.append(featureTracks[i].pt_idx)
                self.duplicate_IDs.append((featureTracks[i].match_idx, featureTracks[i].pt_idx))
        return self

    def printSelf(self):
        print("")
        print("uv1: ", self.uv1)
        print("X: ", self.X)
        print("ID: ", self.ID)
        print("IDs of duplicates: ", self.duplicate_IDs)

def deleteDuplicatePoints(Xs, uv1s, uv2s):
    newXs = []
    newUv1s = []
    newUv2s = []
    for i in range(len(uv1s)):
        uv1, indices = np.unique(uv1s[i], axis = 0, return_index=True)
        newUv1s.append(uv1)
        newUv2s.append(uv2s[i][indices])
        newXs.append(Xs[i][indices])
    return newXs, newUv1s, newUv2s
        


def makeAllPtsFeatureTracks(Xs, uv1s):
    featureTracks = []
    for i in range(len(Xs)):
        for j in range(len(Xs[i])):
            ft = featureTrack(uv1s[i][j,:], Xs[i][j,:], i, j)
            featureTracks.append(ft)
    return featureTracks

def mergeAllCorrespondences(featureTracks):
    mergedFeatureTracks = []
    for ft in featureTracks:
        mergedFeatureTracks.append(ft.mergeCorrespondences(featureTracks))
    return featureTracks

def deleteNonFts(featureTracks):
    newFeatureTracks = []
    for i in range(len(featureTracks)):
        if featureTracks[i].isFeatureTrack():
            newFeatureTracks.append(featureTracks[i])
    return newFeatureTracks

def makeFtsUnique(fts):
    for ft in fts:
        ft.duplicate_IDs.sort(key = lambda y: y[0]) #sort by match_indices
    
    for i in range(len(fts) - 1):
        for j in range(i+1, len(fts)):
            if fts[i].duplicate_IDs == fts[j].duplicate_IDs:
                fts[j].duplicate_pt_indices = []
    
    newFts = deleteNonFts(fts)
    return newFts


def findScalingFactors(fts, allXs):
    equalPts1 = np.zeros((2, 3))
    equalPts2 = np.zeros((2, 3))
    scale_factors = []
    
    for i in range(len(allXs)):
        rand_idx1 = randint(0, len(fts) -1)
        rand_idx2 = randint(0, len(fts) -1)
        notFoundEqualPts1 = True
        notFoundEqualPts2 = True
        #Find random pair of equal points with matchID 0 and matchID i
        while notFoundEqualPts1:
            rand_idx1 = randint(0, len(fts) -1)
            match_ID1, match_ID2 = (-1, -1), (-1, -1)
            
            for j in range(len(fts[rand_idx1].duplicate_IDs)):
                if fts[rand_idx1].duplicate_IDs[j][0] == 0:
                    match_ID1 = fts[rand_idx1].duplicate_IDs[j]
                if fts[rand_idx1].duplicate_IDs[j][0] == i:
                    match_ID2 = fts[rand_idx1].duplicate_IDs[j]
                if match_ID1[0] == 0 and match_ID2[0] == i:
                    notFoundEqualPts1 = False
                    equalPts1[0, :] = allXs[match_ID1[0]][match_ID1[1]]
                    equalPts1[1, :] = allXs[match_ID2[0]][match_ID2[1]]
        #Find another random pair of equal points with matchID 0 and matchID i
        while notFoundEqualPts2:
            rand_idx2 = randint(0, len(fts) -1)
            match_ID1, match_ID2 = (-1, -1), (-1, -1)
            for j in range(len(fts[rand_idx2].duplicate_IDs)):
                if fts[rand_idx2].duplicate_IDs[j][0] == 0:
                    match_ID1 = fts[rand_idx2].duplicate_IDs[j]
                if fts[rand_idx2].duplicate_IDs[j][0] == i:
                    match_ID2 = fts[rand_idx2].duplicate_IDs[j]
                if match_ID1[0] == 0 and match_ID2[0] == i:
                    notFoundEqualPts2 = False
                    equalPts2[0, :] = allXs[match_ID1[0]][match_ID1[1]]
                    equalPts2[1, :] = allXs[match_ID2[0]][match_ID2[1]]
       
        scale_factor = euclideanDistance(equalPts1[0,:], equalPts2[0, :]) / euclideanDistance(equalPts1[1,:], equalPts2[1, :])
        scale_factors.append(scale_factor)
    return scale_factors


def euclideanDistance(pt1, pt2):
    """
    Returns the euclidean distance of 2 points in 3D pt1 and pt2
    """
    return np.linalg.norm(pt1 - pt2)




