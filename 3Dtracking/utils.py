import numpy as np

def euclideanDistance(pt1, pt2):
    """
    Returns the euclidean distance of 2 points in 3D pt1 and pt2
    """
    return np.linalg.norm(pt1 - pt2)