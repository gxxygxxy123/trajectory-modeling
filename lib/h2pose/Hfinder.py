import logging
import cv2
import numpy as np
from math import cos, sin, pi

# Testing Code
import sys
import os
import json
sys.path.append("../../lib")
from common import loadConfig
from common import saveConfig
from h2pose.H2Pose import H2Pose
from numpy.linalg import inv
###########

np.set_printoptions(suppress=True)

class Hfinder(object):
    """docstring for Hfinder"""
    def __init__(self, camera_ks, dist, nmtx, img, court3D, window_width=800):
        super(Hfinder, self).__init__()
        self.camera_ks = camera_ks
        self.dist = dist
        self.nmtx = nmtx
        self.img = img
        self.window_width = window_width
        self.court2D = []

        self.zooming_img = np.copy(self.img)
        self.zooming_center = {} # {x:?, y:?}

        self.ZOOMING_OFFSET_X = 50
        self.ZOOMING_OFFSET_Y = 50
        self.ZOOMING_FACTOR = 4

        self.court3D = court3D
        self.H = np.zeros((3,3)) # mapping 2D pixel to wcs 3D plane
        self.calculateH(self.img)

    def getH(self):
        return self.H

    def mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if not self.zooming_center: # Not Zooming
                if len(self.court2D) < len(self.court3D):
                    self.court2D.append([x, y])
                else:
                    idx = np.linalg.norm(np.array(self.court2D) - np.array([[x, y]]), axis=1).argmin()
                    self.court2D[idx] = [x, y]
                logging.debug("You pressed at ({},{}),including padding".format(x,y))
            else: # Zooming
                new_x = self.zooming_center['x']-self.ZOOMING_OFFSET_X + int(x/self.ZOOMING_FACTOR)
                new_y = self.zooming_center['y']-self.ZOOMING_OFFSET_Y + int(y/self.ZOOMING_FACTOR)
                if len(self.court2D) < len(self.court3D):
                    self.court2D.append([new_x, new_y])
                    self.zooming_center = {}
                else:
                    idx = np.linalg.norm(np.array(self.court2D) - np.array([[new_x, new_y]]), axis=1).argmin()
                    self.court2D[idx] = [new_x, new_y]
                logging.debug("You pressed at ({},{}),including padding".format(new_x,new_y))
        if event == cv2.EVENT_RBUTTONUP:
            if not self.zooming_center: # Not Zooming
                if x-self.ZOOMING_OFFSET_X >= 0 and x+self.ZOOMING_OFFSET_X <= self.img.shape[1] and y-self.ZOOMING_OFFSET_Y >= 0 and y+self.ZOOMING_OFFSET_Y <= self.img.shape[0]:
                    self.zooming_center['x'] = x
                    self.zooming_center['y'] = y
            else:
                self.zooming_center = {}


    def calculateH(self, img):

        cv2.namedWindow("Please pick {} point of court".format(len(self.court3D)), cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Please pick {} point of court".format(len(self.court3D)), (self.window_width,self.window_width*img.shape[0]//img.shape[1]))
        cv2.setMouseCallback("Please pick {} point of court".format(len(self.court3D)), self.mouseEvent)
        while True:
            show = np.copy(img)
            for idx,c in enumerate(self.court2D):
                color = (0,0,0)
                if idx % 4 == 0:
                    color = (0,230,181) # Green
                elif idx % 4 == 1:
                    color = (0,20,229) # Red
                elif idx % 4 == 2:
                    color = (239,80,0) # Blue
                elif idx % 4 == 3:
                    color = (0,200,227) # Yellow
                cv2.putText(show, str(idx+1), (c[0]-10, c[1]-6), cv2.FONT_HERSHEY_TRIPLEX,0.6, color, 1, cv2.LINE_AA)
                cv2.circle(show, (c[0], c[1]), 4, color, -1)
            #if len(self.court2D) > 1:
            #    cv2.drawContours(show, [np.array(self.court2D)], 0, (38, 28, 235), 2)
            if self.zooming_center:
                x,y = self.zooming_center['x'], self.zooming_center['y']
                show = show[y-self.ZOOMING_OFFSET_Y:y+self.ZOOMING_OFFSET_Y,x-self.ZOOMING_OFFSET_X:x+self.ZOOMING_OFFSET_X]
                show = cv2.resize(show,(self.ZOOMING_OFFSET_Y*2*self.ZOOMING_FACTOR,self.ZOOMING_OFFSET_X*2*self.ZOOMING_FACTOR),interpolation=cv2.INTER_AREA)
            cv2.imshow("Please pick {} point of court".format(len(self.court3D)), show)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
        self.court2D = np.array(self.court2D)

        self.court3D = np.array(self.court3D)

        logging.debug("{} points: {}\np.s. excluding padding\n".format(len(self.court3D),self.court2D))

        undistort_track2D = cv2.undistortPoints(np.array(np.expand_dims(self.court2D, 1), np.float32),
                                                np.array(self.camera_ks,np.float32),
                                                np.array(self.dist,np.float32),
                                                None,
                                                np.array(self.nmtx,np.float32))

        logging.debug('undistort {} points: {}\np.s. excluding padding\n'.format(len(self.court3D), undistort_track2D))
        mycourt3D = np.pad(self.court3D,((0,0),(0,1)))

        ## use solvePnP to calculate pose and eye
        ret, rvec, tvec = cv2.solvePnP(np.array(mycourt3D,np.float32), np.array(self.court2D,np.float32), np.array(self.camera_ks,np.float32), np.array(self.dist,np.float32),flags = cv2.SOLVEPNP_ITERATIVE)
        #logging.debug('rain tvec:{}'.format(tvec))
        T = np.array(tvec,np.float32)
        R = np.array(cv2.Rodrigues(rvec)[0],np.float32)
        self.R = R
        self.T = T
        #logging.debug('rain rvec:\n{}'.format(R))
        #logging.debug('rain t:\n{}'.format(T))
        #logging.debug('eye:\n{}'.format(-R.T.dot(T)))
        #pose_inv = np.concatenate((R[:,0:2],T),axis=1)
        #pose = inv(pose_inv)
        #logging.debug('pose_inv:\n{}'.format(pose_inv))
        #logging.debug('pose:\n{}'.format(pose))
        self.projection_mat = self.nmtx@np.concatenate((R,T),axis=1)
        #logging.debug('projection_mat:\n{}'.format(self.projection_mat))
        self.H, status = cv2.findHomography(np.squeeze(undistort_track2D, axis=1), self.court3D)

    def getProjection_mat(self):
        return self.projection_mat

    def getExtrinsic_mat(self):
        return np.concatenate((self.R,self.T),axis=1)

if __name__ == '__main__': # Debug: Finding Eye bugs
    f = sys.argv[1] # Video
    camera_config = sys.argv[2] # 12345678.cfg

    video = cv2.VideoCapture(f)
    success, image = video.read()
    if success:
        # cv2.imwrite('test.png',image)
        c_config =  loadConfig(camera_config)

        camera_ks = np.array(json.loads(c_config['Other']['ks']))
        dist = np.array(json.loads(c_config['Other']['dist']))
        nmtx = np.array(json.loads(c_config['Other']['newcameramtx']))

        # You should enter the 3d points coordinate (on the ground)
        # court3D = [[0.02, 6.66], [2.55, 6.66], [2.55, 5.94], [0.02, 5.94]]
        # court3D = [[-3.01, 6.66], [-2.55, 6.66], [-0.02, 6.66], [0.02, 6.66], [2.55, 6.66], [3.01, 6.66], [-3.01, 5.9], [-2.55, 5.9], [-0.02, 5.9], [0.02, 5.9], [2.55, 5.9], [3.01, 5.9], [0.02, 2.02]]

        hf = Hfinder(camera_ks=camera_ks, dist=dist, nmtx=nmtx, img=image, court3D=court3D)
        Hmtx = hf.getH()
        Kmtx = nmtx
        projection_mat = hf.getProjection_mat()
        extrinsic_mat = hf.getExtrinsic_mat()
        h2p = H2Pose(Kmtx, Hmtx)

        poses = h2p.getC2W()
        eye = h2p.getCamera().T
        eye[0][2] = abs(eye[0][2])

        c_config['Other']['poses'] = str(poses.tolist())
        c_config['Other']['eye'] = str(eye.tolist())
        c_config['Other']['hmtx'] = str(Hmtx.tolist())
        c_config['Other']['projection_mat'] = str(projection_mat.tolist())
        c_config['Other']['extrinsic_mat'] = str(extrinsic_mat.tolist())

        saveConfig(camera_config, c_config)

