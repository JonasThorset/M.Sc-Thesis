import math
import numpy as np

# Checks if a matrix is in SO3.
def isRotMat(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles (xyz)
def R2Euler(R) :
    assert(isRotMat(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

# Calculates Rotation Matrix given euler angles.
def euler2R(theta) :
    print(np.cos(theta[0]))
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])  ],
                    [0, 1, 0 ],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = R_z @ R_y @ R_x
    return R

def deg2Rad(deg):
    return math.radians(deg)

def rad2Deg(rad):
    return math.degrees(rad)

def printPose(T):
    euler_angles = R2Euler(T[:3, :3])
    euler_angles[0] = rad2Deg(euler_angles[0])
    euler_angles[1] = rad2Deg(euler_angles[1])
    euler_angles[2] = rad2Deg(euler_angles[2])
    print("x [m]: ", T[0, -1])
    print("y [m]: ", T[1, -1])
    print("z [m]: ", T[2, -1])
    print("rot_x [deg]: ", euler_angles[0])
    print("rot_y [deg]: ", euler_angles[1])
    print("rot_z [deg]: ", euler_angles[2])


def bundle_params2T(bundle_params):
    """
    Transforms n camera parameter arrays from bundle form
    to T = [R,t]
    """
    n = bundle_params.shape[0]
    Ts = []
    for i in range(n):
        R = euler2R(bundle_params[i,:3])
        t = bundle_params[i,3:6]
        t = t.reshape((t.shape[0]), 1)
        T = np.hstack([R, t])
        T = np.vstack([T, np.array([0,0,0,1])])
        Ts.append(T)
    return Ts
