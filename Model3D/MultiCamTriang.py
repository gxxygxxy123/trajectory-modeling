# %%
import numpy as np
import cv2
import os, argparse
import logging
from math import sqrt, pi, sin, cos, tan

class MultiCamTriang(object):
    """docstring for MultiCamTriang"""
    def __init__(self, poses, eye, Ks):
        super(MultiCamTriang, self).__init__()
        self.poses = poses                   # shape:(num_cam, c2w(3, 3)) transform matrix from ccs to wcs
        self.eye = eye                       # shape:(num_cam, 1, xyz(3)) camera position in wcs
        self.Ks = Ks                         # shape:(num_cam, K(3,3)) intrinsic matrix
        self.f = (Ks[:,0,0] + Ks[:,1,1]) / 2 # shape:(num_cam) focal length
        self.p = Ks[:,0:2,2]                 # shape:(num_cam, xy(2)) principal point
        # self.projection_mat = projection_mat

    def setTrack2Ds(self, newtrack2Ds):     # must set, then run calculate3D
        #logging.debug('setting track2Ds:{newtrack2Ds}')
        self.track2Ds = newtrack2Ds          # shape:(num_cam, num_frame, xy(2)) 2D track from TrackNetV2

    def calculate3D(self):
        self.num_cam, self.num_frame, _ = self.track2Ds.shape
        self.backProject()
        self.getApprox3D()
        return self.track3D

    def setProjectionMats(self, ProjectionMats):
        self.projection_mat = ProjectionMats

    def setPoses(self, poses):
        self.poses = poses

    def setEye(self, eye):
        self.eye = eye

    def setKs(self, ks):
        self.Ks = ks
        self.f = (ks[:,0,0] + ks[:,1,1]) / 2
        self.p = ks[:,0:2,2]

    def rain_calculate3D(self):
        total_track3Ds = np.zeros((1,3))
        track_3Ds = []
        for i in range(0,len(self.track2Ds)-1):
            for j in range(i+1,len(self.track2Ds)):
                self.track3D_homo = cv2.triangulatePoints(self.projection_mat[i],self.projection_mat[j],self.track2Ds[i][0],self.track2Ds[j][0]) # shape:(4,num_frame), num_frame=1
                self.track3D = self.track3D_homo[:3] / self.track3D_homo[3] # shape:(3,num_frame), num_frame=1
                self.track3D = np.stack(self.track3D, axis=1) # shape:(num_frame,3), num_frame=1
                track_3Ds.append(self.track3D)
                # logging.debug('i:{}, j:{}, self.track3D:{}'.format(i, j, self.track3D))

        track_3Ds = np.array(track_3Ds)
        n=1.5
        #IQR = Q3-Q1
        for i in range(0,3):
            IQR = np.percentile(track_3Ds[:,:,i],75) - np.percentile(track_3Ds[:,:,i],25)
            track_3Ds = track_3Ds[track_3Ds[:,:,i] <= np.percentile(track_3Ds[:,:,i],75)+n*IQR]
            track_3Ds = track_3Ds[:,np.newaxis,:]
        for i in track_3Ds:
            total_track3Ds += i
        return total_track3Ds/len(track_3Ds)
        # self.track3D_homo = cv2.triangulatePoints(self.projection_mat[0],self.projection_mat[1],self.track2Ds[0][0],self.track2Ds[1][0]) # shape:(4,num_frame), num_frame=1
        # self.track3D = self.track3D_homo[:3] / self.track3D_homo[3] # shape:(3,num_frame), num_frame=1
        # self.track3D = np.stack(self.track3D, axis=1) # shape:(num_frame,3), num_frame=1
        # return self.track3D

    def backProject(self):
        # Back project the 2d points of all frames to the 3d ray in world coordinate system

        # Shift origin to principal point
        self.track2Ds_ccs = self.track2Ds - self.p[:,None,:]

        # Back project 2D track to the CCS
        self.track2Ds_ccs = self.track2Ds_ccs / self.f[:,None,None]
        track_d = np.ones((self.num_cam, self.num_frame, 1))
        self.track2Ds_ccs = np.concatenate((self.track2Ds_ccs, track_d), axis=2)

        # 2D track described in WCS
        self.track2D_wcs = self.poses @ np.transpose(self.track2Ds_ccs, (0,2,1)) # shape:(num_cam, 3, num_frame)
        self.track2D_wcs = np.transpose(self.track2D_wcs, (0,2,1)) # shape:(num_cam, num_frame, 3)
        self.track2D_wcs = self.track2D_wcs / np.linalg.norm(self.track2D_wcs, axis=2)[:,:,None]

    def getApprox3D(self):
        # Calculate the approximate solution of the ball postition by the least square method
        # n-lines intersection == 2n-planes intersection

        planeA = np.copy(self.track2D_wcs)
        planeA[:,:,0] = 0
        planeA[:,:,1] = -self.track2D_wcs[:,:,2]
        planeA[:,:,2] = self.track2D_wcs[:,:,1]

        # check norm == 0
        planeA_tmp = np.copy(self.track2D_wcs)
        planeA_tmp[:,:,0] = -self.track2D_wcs[:,:,2]
        planeA_tmp[:,:,1] = 0
        planeA_tmp[:,:,2] = self.track2D_wcs[:,:,0]
        mask = np.linalg.norm(planeA, axis=2)==0
        planeA[mask] = planeA_tmp[mask]

        # # check norm == 0
        # planeA_tmp = np.copy(self.track2D_wcs)
        # planeA_tmp[:,:,0] = -self.track2D_wcs[:,:,1]
        # planeA_tmp[:,:,1] = self.track2D_wcs[:,:,0]
        # planeA_tmp[:,:,2] = 0
        # mask = np.linalg.norm(planeA, axis=2)==0
        # planeA[mask] = planeA_tmp[mask]

        planeB = np.cross(self.track2D_wcs, planeA)

        Amtx = np.concatenate((planeA, planeB), axis=0) # shape:(2num_cam, num_frame, 3)
        b = np.concatenate((self.eye*planeA, self.eye*planeB), axis=0).sum(-1)[:,:,None] # shape:(2num_cam, num_frame, 1)

        Amtx = np.transpose(Amtx, (1,0,2)) # shape:(num_frame, 2num_cam, 3)
        b = np.transpose(b, (1,0,2)) # shape:(num_frame, 2num_cam, 1)

        left = np.transpose(Amtx, (0,2,1)) @ Amtx # shape:(num_frame, 3, 3)
        right = np.transpose(Amtx, (0,2,1)) @ b # shape:(num_frame, 3, 1)

        self.track3D = np.linalg.pinv(left) @ right # shape:(num_frame, 3, 1)
        self.track3D = self.track3D.reshape(-1,3)
        '''
        [[-1.52680479,  2.06202114,  2.04934252],
        [-1.34739384,  1.67499119,  2.49134506],
        [-1.16905542,  1.29068153,  2.88204759],
        [-0.9924781,   0.91104616,  3.22891526],
        [-0.81207975,  0.5231891,   3.53126069],
        [-0.63397124,  0.14283066,  3.78343654],
        [-0.46045112, -0.23837983,  3.99178557],
        [-0.28176591, -0.62200495,  4.15098372],
        [-0.10533186, -1.00410534,  4.26820641],
        [ 0.07375752, -1.38452603,  4.337159  ],
        [ 0.25253153, -1.77248105,  4.36025826],
        [ 0.43151949, -2.15530492,  4.33638467],
        [ 0.61271096, -2.53986812,  4.26825079],
        [ 0.78954231, -2.91919479,  4.14984721],
        [ 0.96996572, -3.30069187,  3.9908667 ],
        [ 1.14671596, -3.68684962,  3.78039425],
        [ 1.32658648, -4.07093151,  3.52737657],
        [ 1.5030584,  -4.45345251,  3.22569203],
        [ 1.68413826, -4.8356843,   2.88023936],
        [ 1.86211489, -5.22080128,  2.48178244],
        [ 2.0396313,  -5.60385936,  2.04277793]]
        '''
'''
    mct = MultiCamTriang(
        track2Ds=np.array(track2Ds), 
        poses=np.array(poses), 
        eye=np.array(eye), 
        Ks=np.array(Ks)
'''
