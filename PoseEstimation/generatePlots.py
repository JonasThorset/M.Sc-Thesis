import matplotlib.pyplot as plt
import numpy as np
from rotationUtils import*

def project(K, X):
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

def project_camera_frame(K, T, scale, color = 'black'):
    """
    Draw the axes of T and a pyramid, representing the camera.
    """
    s = scale
    X = []
    X.append(np.array([0,0,0,1]))
    X.append(np.array([-s,-s,1.5*s,1]))
    X.append(np.array([+s,-s,1.5*s,1]))
    X.append(np.array([+s,+s,1.5*s,1]))
    X.append(np.array([-s,+s,1.5*s,1]))
    X.append(np.array([5.0*s,0,0,1]))
    X.append(np.array([0,5.0*s,0,1]))
    X.append(np.array([0,0,5.0*s,1]))
    X = np.array(X).T
    u,v = project(K, T@X)
    lines = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
    plt.plot([u[0], u[5]], [v[0], v[5]], color='#ff5555', linewidth=2)
    plt.plot([u[0], u[6]], [v[0], v[6]], color='#33cc55', linewidth=2)
    plt.plot([u[0], u[7]], [v[0], v[7]], color='#44aaff', linewidth=2)
    for (i,j) in lines:
        plt.plot([u[i], u[j]], [v[i], v[j]], color)






def draw_correspondences(I1, I2, uv1, uv2, F, sample_size=8):
    """
    Draws a random subset of point correspondences and their epipolar lines.
    """
    assert uv1.shape[0] == 3 and uv2.shape[0] == 3, 'uv1 and uv2 must be 3 x n arrays of homogeneous 2D coordinates.'
    sample = np.random.choice(range(uv1.shape[1]), size=sample_size, replace=False)
    uv1 = uv1[:,sample]
    uv2 = uv2[:,sample]
    n = uv1.shape[1]
    uv1 /= uv1[2,:]
    uv2 /= uv2[2,:]

    l1 = F.T@uv2
    l2 = F@uv1

    colors = plt.cm.get_cmap('Set2', n).colors
    plt.figure('Correspondences', figsize=(10,4))
    plt.subplot(121)
    plt.imshow(I1)
    plt.xlabel('Image 1')
    plt.scatter(*uv1[:2,:], s=100, marker='x', c=colors)
    for i in range(n):
        hline(l1[:,i], linewidth=1, color=colors[i], linestyle='--')
    plt.xlim([0, I1.shape[1]])
    plt.ylim([I1.shape[0], 0])

    plt.subplot(122)
    plt.imshow(I2)
    plt.xlabel('Image 2')
    plt.scatter(*uv2[:2,:], s=100, marker='o', zorder=10, facecolor='none', edgecolors=colors, linewidths=2)
    for i in range(n):
        hline(l2[:,i], linewidth=1, color=colors[i], linestyle='--')
    plt.xlim([0, I2.shape[1]])
    plt.ylim([I2.shape[0], 0])
    plt.tight_layout()
    plt.suptitle('Point correspondences and associated epipolar lines (showing %d randomly drawn pairs)' % sample_size)

def draw_point_cloud(X, I1, uv1, xlim, ylim, zlim):
    assert uv1.shape[1] == X.shape[1], 'If you get this error message in Task 4, it probably means that you did not extract the inliers of all the arrays (uv1,uv2,xy1,xy2) before calling draw_point_cloud.'

    # We take I1 and uv1 as arguments in order to assign a color to each
    # 3D point, based on its pixel coordinates in one of the images.
    c = I1[uv1[1,:].astype(np.int32), uv1[0,:].astype(np.int32), :]

    # Matplotlib doesn't let you easily change the up-axis to match the
    # convention we use in the course (it assumes Z is upward). So this
    # code does a silly rearrangement of the Y and Z arguments.
    plt.figure('3D point cloud', figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.scatter(X[0,:], X[2,:], X[1,:], c=c, marker='.', depthshade=False)
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.title('[Click, hold and drag with the mouse to rotate the view]')

def hline(l, **args):
    """
    Draws a homogeneous 2D line.
    You must explicitly set the figure xlim, ylim before or after using this.
    """

    lim = np.array([-1e8, +1e8]) # Surely you don't have a figure bigger than this!
    a,b,c = l
    if np.absolute(a) > np.absolute(b):
        x,y = -(c + b*lim)/a, lim
    else:
        x,y = lim, -(c + a*lim)/b
    plt.plot(x, y, **args)




def draw_poses_and_cloud(Ts, X, frame1, uv1, xlim, ylim, zlim, vec_scale = 1):

    assert uv1.shape[1] == X.shape[1]

    c = frame1[uv1[1,:].astype(np.int32), uv1[0,:].astype(np.int32), :]

    # Matplotlib doesn't let you easily change the up-axis to match the
    # convention we use in the course (it assumes Z is upward). So this
    # code does a silly rearrangement of the Y and Z arguments.
    plt.figure('3D point cloud', figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.scatter(X[0,:], X[2,:], -X[1,:], c=c, marker='.', depthshade=False)
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim(ylim)
    #ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    
    #Origin
    p0 = np.array([0,0,0])
    x0 = np.array([1,0,0])
    y0 = np.array([0,1,0])
    z0 = np.array([0,0,1])

    #Reference cam
    R = Ts[0][:3, :3].copy()
    t = Ts[0][:3, -1].copy()
    p = p0 + t
    x = R @ x0
    y = R @ y0
    z = R @ z0

    ax.quiver(p[0],p[2], p[1], x[0], x[2], -x[1], length = 0.1*vec_scale, color = "red", label = "x-axis")
    ax.quiver(p[0],p[2], p[1], y[0], y[2], -y[1], length = 0.1*vec_scale, color = "green" , label = "y-axis")
    ax.quiver(p[0],p[2], p[1], z[0], z[2], -z[1], length = 0.1*vec_scale, color = "blue" , label = "z-axis")
    ax.legend()
    ax.text(p[0],p[2]-0.5,p[1], "Cam0" )
    for i in range(1, len(Ts)):
        R = Ts[i][:3, :3].T.copy()
        t = Ts[i][:3, -1].copy()
        t_y = t[1]
        #t[1] = t[2]
        #t[2] = t_y

        p_i = p - t
        x_i = R @ x
        y_i = R @ y
        z_i = R @ z
        ax.quiver(p_i[0],p_i[2], p_i[1], x_i[0], x_i[2], -x_i[1], length = 0.1*vec_scale, color = "red")
        ax.quiver(p_i[0],p_i[2], p_i[1], y_i[0], y_i[2], -y_i[1], length = 0.1*vec_scale, color = "green")
        ax.quiver(p_i[0],p_i[2], p_i[1], z_i[0], z_i[2], -z_i[1], length = 0.1*vec_scale, color = "blue")
        ax.text(p_i[0],p_i[2]+0.5,p_i[1], "Cam" + str(i))
   
    
    """ #Plot GT cameras
    T0_gt = np.loadtxt("T0b_gt.txt")
    T1_gt = np.loadtxt("T1b_gt.txt")
    T2_gt = np.loadtxt("T2b_gt.txt")
    printPose(T0_gt)
    printPose(T1_gt)
    printPose(T2_gt)
    Ts_gt = [T0_gt,T1_gt,T2_gt]
    for i in range(len(Ts_gt)):
        R_i = Ts_gt[i][:3, :3]
        t_i = Ts_gt[i][:3, -1]
        p_i = p + t_i
        x_i = R_i @ x
        y_i = R_i @ y
        z_i = R_i @ z
        ax.quiver(p_i[0],p_i[2], p_i[1], x_i[0], x_i[2], x_i[1], length = 0.1*vec_scale, color = "red")
        ax.quiver(p_i[0],p_i[2], p_i[1], y_i[0], y_i[2], y_i[1], length = 0.1*vec_scale, color = "green")
        ax.quiver(p_i[0],p_i[2], p_i[1], z_i[0], z_i[2], z_i[1], length = 0.1*vec_scale, color = "blue")
        ax.text(p_i[0],p_i[2],p_i[1], "Camb" + str(i))

    """
    plt.show()



def draw_poses_and_cloud_B(Ts, X, frame1, uv1, xlim, ylim, zlim, vec_scale = 1):

    assert uv1.shape[1] == X.shape[1]

    c = frame1[uv1[1,:].astype(np.int32), uv1[0,:].astype(np.int32), :]

    #This code does a silly rearrangement of the Y and Z arguments since matplotlib sucks and dont support rearranging in simpler ways.
    plt.figure('3D point cloud', figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.scatter(X[0,:], X[2,:], -X[1,:], c=c, marker='.', depthshade=False)
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim(ylim)
    #ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    
    #Origin
    p0 = np.array([0,0,0])
    x0 = np.array([1,0,0])
    y0 = np.array([0,1,0])
    z0 = np.array([0,0,1])

    #Reference cam
    R = Ts[0][:3, :3].copy()
    t = Ts[0][:3, -1].copy()
    p = p0 + t
    x = R @ x0
    y = R @ y0
    z = R @ z0

    ax.quiver(p[0],p[2], p[1], x[0], x[2], -x[1], length = 0.1*vec_scale, color = "red", label = "x-axis")
    ax.quiver(p[0],p[2], p[1], y[0], y[2], -y[1], length = 0.1*vec_scale, color = "green" , label = "y-axis")
    ax.quiver(p[0],p[2], p[1], z[0], z[2], -z[1], length = 0.1*vec_scale, color = "blue" , label = "z-axis")
    ax.legend()
    ax.text(p[0],p[2]-1,p[1], "Cam0" )
    for i in range(1, len(Ts)):
        R = Ts[i][:3, :3].copy()
        t = Ts[i][:3, -1].copy()
        p_i = p + t
        x_i = R @ x
        y_i = R @ y
        z_i = R @ z
        ax.quiver(p_i[0],p_i[2], p_i[1], x_i[0], x_i[2], -x_i[1], length = 0.1*vec_scale, color = "red")
        ax.quiver(p_i[0],p_i[2], p_i[1], y_i[0], y_i[2], -y_i[1], length = 0.1*vec_scale, color = "green")
        ax.quiver(p_i[0],p_i[2], p_i[1], z_i[0], z_i[2], -z_i[1], length = 0.1*vec_scale, color = "blue")
        ax.text(p_i[0],p_i[2]+1,p_i[1], "Cam" + str(i))
   
    
    """ #Plot GT cameras
    T0_gt = np.loadtxt("T0b_gt.txt")
    T1_gt = np.loadtxt("T1b_gt.txt")
    T2_gt = np.loadtxt("T2b_gt.txt")
    printPose(T0_gt)
    printPose(T1_gt)
    printPose(T2_gt)
    Ts_gt = [T0_gt,T1_gt,T2_gt]
    for i in range(len(Ts_gt)):
        R_i = Ts_gt[i][:3, :3]
        t_i = Ts_gt[i][:3, -1]
        p_i = p + t_i
        x_i = R_i @ x
        y_i = R_i @ y
        z_i = R_i @ z
        ax.quiver(p_i[0],p_i[2], p_i[1], x_i[0], x_i[2], x_i[1], length = 0.1*vec_scale, color = "red")
        ax.quiver(p_i[0],p_i[2], p_i[1], y_i[0], y_i[2], y_i[1], length = 0.1*vec_scale, color = "green")
        ax.quiver(p_i[0],p_i[2], p_i[1], z_i[0], z_i[2], z_i[1], length = 0.1*vec_scale, color = "blue")
        ax.text(p_i[0],p_i[2],p_i[1], "Camb" + str(i))

    """
    plt.show()



