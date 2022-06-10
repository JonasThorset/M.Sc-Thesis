import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from rotationUtils import*
import matplotlib.pylab as plt2
import scipy.sparse as sparse
import cv2

#This class is inspired by https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

class BA:
    """Bundle Adjustment Class"""

    def __init__(self, cam_params, Xs, uvs, cam_indices, uv_indices):
        
        self.cam_params = cam_params
        self.Xs = Xs
        self.uvs = uvs
        self.uv_indices = uv_indices
        self.cam_indices = cam_indices

    def sparsity(self, num_cams, optParamsPerCam, num_points, cam_indices, uv_indices):
        m = cam_indices.size * 2
        n = num_cams * optParamsPerCam + num_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cam_indices.size)
        for s in range(optParamsPerCam):
            A[2 * i, cam_indices * optParamsPerCam + s] = 1
            A[2 * i + 1, cam_indices * optParamsPerCam + s] = 1

        for s in range(3):
            A[2 * i, num_cams * optParamsPerCam + uv_indices * 3 + s] = 1
            A[2 * i + 1, num_cams * optParamsPerCam + uv_indices * 3 + s] = 1

        return A

    def project(self, Xs, camera_params):
        points_proj = []
        Xs_hom = np.c_[Xs, np.ones(Xs.shape[0])]

        for i in range(camera_params.shape[0]):
            fx = camera_params[i, 6]
            fy = camera_params[i, 7]
            cx = camera_params[i, 8]
            cy = camera_params[i, 9]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            R = euler2R(camera_params[i, :3])
            t = camera_params[i, 3:6]
            T = np.c_[R, t]
            T = np.vstack((T, np.array([0,0,0,1])))
            uvw = T@ Xs_hom[i, :]
            uv_norm = (uvw / uvw[2])[:-1]
            uv_norm = K@uv_norm
            uv = (uv_norm/uv_norm[2])[:2]
            points_proj.append(uv)
            

        return np.array(points_proj)

            
    def fun(self, params, n_cameras, optParamsPerCam, n_points, camera_indices, point_indices, points_2d, intrinsics):
        extrinsics = params[:n_cameras * optParamsPerCam].reshape((n_cameras, optParamsPerCam))
        camera_params = np.hstack((extrinsics, intrinsics))
        Xs = params[n_cameras * optParamsPerCam:].reshape((n_points, 3))
        points_proj = self.project(Xs[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()
    
    def optimizedParams(self, params, n_cameras, n_paramsPerCam, n_points, intrinsics):
        camera_params = params[:n_cameras * n_paramsPerCam].reshape((n_cameras, n_paramsPerCam))
        camera_params = np.hstack((camera_params, intrinsics))
        Xs = params[n_cameras * n_paramsPerCam:].reshape((n_points, 3))

        return camera_params, Xs

    def bundleAdjust(self):
        num_cams = self.cam_params.shape[0]
        optParamsPerCam = self.cam_params[:,:6].shape[1] #extrinsics only
        num_points = self.Xs.shape[0]
        intrinsics = self.cam_params[:, 6:]
        params0 = np.hstack((self.cam_params[:,:6].ravel(), self.Xs.ravel()))
        f0 = self.fun(params0, num_cams, optParamsPerCam, num_points, self.cam_indices, self.uv_indices, self.uvs, intrinsics)
        
        A = self.sparsity(num_cams,optParamsPerCam, num_points, self.cam_indices, self.uv_indices)
        #self.displaySparsity(A)
        res = least_squares(self.fun, params0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(num_cams, optParamsPerCam, num_points, self.cam_indices, self.uv_indices, self.uvs, intrinsics))

        cam_params_optimized, Xs_optimized = self.optimizedParams(res.x, num_cams, optParamsPerCam, num_points, intrinsics)

        
        plt.plot(f0)
        plt.show()
        
        plt.plot(res.fun)
        plt.show()

        return cam_params_optimized, Xs_optimized

    def displaySparsity(self, A):
        plt2.spy(A)
        plt2.show()
        

def prepareBundleAgainstFirstImage(allXs, allTs, uv1s, uv2s, camIDs, Ks, fts):
    """
    This function assumes all images are matched with the first frame
    """
    camera_params = np.zeros((allTs.shape[0], 10))
    for i in range(allTs.shape[0]):
        camera_params[i, :3] = R2Euler(allTs[i, :3,:3])
        camera_params[i, 3:6] = allTs[i, :3, -1]

    camera_params[0, 6] = Ks[camIDs[0,0]][0,0]               #fx
    camera_params[0, 7] = Ks[camIDs[0,0]][1,1]               #fy
    camera_params[0, 8] = Ks[camIDs[0,0]][0,2]               #cx
    camera_params[0, 9] = Ks[camIDs[0,0]][1,2]               #cy
    for i in range(len(allXs)):
        camera_params[i+1, 6] = Ks[camIDs[i,1]][0,0]               
        camera_params[i+1, 7] = Ks[camIDs[i,1]][1,1]               
        camera_params[i+1, 8] = Ks[camIDs[i,1]][0,2]               
        camera_params[i+1, 9] = Ks[camIDs[i,1]][1,2] 

    points_3d = []
    points_2d = []
    point_indices = []
    camera_indices = []
    pt_idx = 0

    #address fts first
    for i in range(len(fts)):
        points_3d.append(fts[i].X)
        matchIDs = []
        ptIDs = []
        
        for j in range(len(fts[i].duplicate_IDs)):
            matchID = fts[i].duplicate_IDs[j][0]
            ptID = fts[i].duplicate_IDs[j][1]

            points_2d.append(uv1s[matchID][ptID])
            points_2d.append(uv2s[matchID][ptID])

            camera_indices.append(0)
            camera_indices.append(matchID + 1)
            point_indices.append(pt_idx)
            point_indices.append(pt_idx)
           
        pt_idx += 1

    #adress non-fts
    for i in range(len(allXs)):
        for j in range(len(allXs[i])):
            if allXs[i][j] not in np.array(points_3d):
                points_3d.append(allXs[i][j,:])
                points_2d.append(uv1s[i][j,:])
                points_2d.append(uv2s[i][j,:])
                camera_indices.append(0)
                camera_indices.append(i+1)
                point_indices.append(pt_idx)
                point_indices.append(pt_idx)
                pt_idx += 1
                    
    points_2d = np.array(points_2d)
    points_3d = np.array(points_3d)
    point_indices = np.array(point_indices)
    camera_indices = np.array(camera_indices)
   
    return camera_params, points_3d, camera_indices.astype(int), point_indices.astype(int), points_2d