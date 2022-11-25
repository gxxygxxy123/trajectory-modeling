import cv2
import numpy as np
from math import cos, sin, pi

class H2Pose(object):
    """docstring for H2Pose"""
    def __init__(self, K, H):
        super(H2Pose, self).__init__()
        self.K = K
        self.H = H # H: ccs -> wcs

        self.R = np.zeros((3,3))
        self.t = np.zeros(3)
        self.Rt = np.zeros((3,4))

        self.c2w = np.zeros((3,3))
        self.Cam_wcs = np.zeros((3,1)) # camera position in wcs
        self.Ori_ccs = np.zeros((3,1)) # world center in ccs

        self.buildProjection()
        self.buildTransform()

    def buildProjection(self):
        K_inv = np.linalg.inv(self.K)
        H_inv = np.linalg.inv(self.H) # H_inv: wcs -> ccs

        lamda1 = np.linalg.norm(K_inv@H_inv[:,0], ord=None, axis=None, keepdims=False)
        lamda2 = np.linalg.norm(K_inv@H_inv[:,1], ord=None, axis=None, keepdims=False)
        lamda3 = (lamda1+lamda2)/2
        
        self.R[:,0] = (K_inv@H_inv[:,0])/lamda1
        self.R[:,1] = (K_inv@H_inv[:,1])/lamda2
        self.R[:,2] = np.cross(self.R[:,0], self.R[:,1])
        self.t = np.array((K_inv@H_inv[:,2])/lamda3)

        self.Rt[:,:3] = self.R
        self.Rt[:,3] = self.t

    def buildTransform(self):
        self.Ori_ccs = (self.Rt @ [[0],[0],[0],[1]])
        cir_pose_i_ccs = (self.Rt @ [[1],[0],[0],[1]]) - self.Ori_ccs
        cir_pose_j_ccs = (self.Rt @ [[0],[1],[0],[1]]) - self.Ori_ccs
        cir_pose_k_ccs = (self.Rt @ [[0],[0],[1],[1]]) - self.Ori_ccs
        self.c2w = np.array(
            [cir_pose_i_ccs.reshape(-1),
             cir_pose_j_ccs.reshape(-1),
             cir_pose_k_ccs.reshape(-1)]
        )
        self.Cam_wcs = (self.c2w @ -self.Ori_ccs)

        # Check Fake Pose
        if self.Ori_ccs[2,0] < 0:
            # Fake pose correction
            R_fix = np.array([[ cos(pi), sin(pi), 0],
                              [-sin(pi), cos(pi), 0],
                              [ 0,             0, 1]])
            t_fix = R_fix @ self.Cam_wcs
            Rt_fix = np.array([[ cos(pi), sin(pi), 0, -2*t_fix[0,0]],
                              [-sin(pi), cos(pi), 0, -2*t_fix[1,0]],
                              [ 0,             0, 1, +2*t_fix[2,0]],
                              [ 0,             0, 0,             1]])

            self.Rt = (np.concatenate((self.Rt,np.array([[0,0,0,1]])),axis=0)@Rt_fix)[:3,:]
            self.Ori_ccs = (self.Rt @ [[0],[0],[0],[1]])
            cir_pose_i_ccs = (self.Rt @ [[1],[0],[0],[1]]) - self.Ori_ccs
            cir_pose_j_ccs = (self.Rt @ [[0],[1],[0],[1]]) - self.Ori_ccs
            cir_pose_k_ccs = (self.Rt @ [[0],[0],[1],[1]]) - self.Ori_ccs
            self.c2w = np.array(
                [cir_pose_i_ccs.reshape(-1),
                 cir_pose_j_ccs.reshape(-1),
                 cir_pose_k_ccs.reshape(-1)]
            )
            self.Cam_wcs = (self.c2w @ -self.Ori_ccs)

            '''
            print('Cam_wcs:', Cam_wcs)

            print('c2w:')
            print(c2w)

            print('wcsO in ccs:',Ori_ccs.reshape(3))
            print('wcsX in ccs:',cir_pose_i_ccs.reshape(3))
            print('wcsY in ccs:',cir_pose_j_ccs.reshape(3))
            print('wcsZ in ccs:',cir_pose_k_ccs.reshape(3))
            '''

    def getRt(self):
        return self.Rt

    def getP(self):
        return self.K @ self.Rt

    def getC2W(self):
        return self.c2w

    def getCamera(self):
        return self.Cam_wcs

    def getCenter(self):
        return self.Ori_ccs