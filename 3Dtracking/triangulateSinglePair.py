import numpy as np

def triangulateSinglePair(uv1, uv2, K1, K2, T1, T2):
    
    uv1 = np.array([uv1[0], uv1[1], 1]).reshape((3, 1))
    uv2 = np.array([uv2[0], uv2[1], 1]).reshape((3, 1))
    P1 = T1[:3,:]
    P2 = T2[:3,:]
    xy1 = np.linalg.inv(K1) @ uv1
    xy2 = np.linalg.inv(K2) @ uv2
    
    X = np.empty((4,1))
    
    A = np.empty((4,4))
    A[0,:] = P1[0,:] - xy1[0]*P1[2,:]
    A[1,:] = P1[1,:] - xy1[1]*P1[2,:]
    A[2,:] = P2[0,:] - xy2[0]*P2[2,:]
    A[3,:] = P2[1,:] - xy2[1]*P2[2,:] 
    
    
    _,_,VT = np.linalg.svd(A)
    X = VT[3,:]/VT[3,3]
    X = X.reshape((4,1))
    return X


