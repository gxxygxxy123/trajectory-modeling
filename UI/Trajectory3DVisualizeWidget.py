from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import pandas as pd
import os
import sys
import cv2
import argparse, csv, json

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import nan, sqrt, pi, sin, cos, tan

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ICONDIR = f"{ROOTDIR}/UI/icon"

sys.path.append(f"{ROOTDIR}/Pseudo3D")
from generator import toRad, drawCourt, drawNet
sys.path.append(f"{ROOTDIR}/RNN") # no push to git
from threeDprojectTo2D import FitVerticalPlaneTo2D


np.set_printoptions(precision=14)
np.set_printoptions(suppress=True)

class Transform(object):
    def __init__(self):
        super(Transform, self).__init__()
        self.gt = False

        # Opengl param
        self.eye = np.zeros(3)
        self.obj = np.zeros(3)
        self.up = np.zeros(3)
        self.quadric = gluNewQuadric()

        # Curve param
        self._f = 0 # guess focal offset
        self.rad_z = 0
        self.rad_y = 0
        self.rad_x = 0

        # OpenGL Init camera pose parameters for gluLookAt() function
        init_pose = (
            np.array([0., -15.5885, 9.]),
            np.array([0., 0., 0.]),
            np.array([0. , 0.5 , 0.866])
        )
        self.setupCam(init_pose)

    def setupCam(self, pose):
        self.eye = pose[0]
        self.obj = pose[1]
        self.up = pose[2]

    def rotate(self, zAngle, yAngle, xAngle):
        rot_z = np.array([[cos(zAngle), -sin(zAngle), 0],
                        [sin(zAngle),  cos(zAngle), 0],
                        [0, 0, 1]])
        rot_y = np.array([[cos(yAngle), 0, sin(yAngle)],
                        [0, 1, 0],
                        [-sin(yAngle), 0, cos(yAngle)]])

        rot_x = np.array([[1, 0, 0],
                        [0, cos(xAngle), -sin(xAngle)],
                        [0, sin(xAngle), cos(xAngle)]])

        _eye = rot_x @ rot_y @ rot_z @ self.eye.reshape(-1,1)

        _up = rot_x @ rot_y @ rot_z @ self.up.reshape(-1,1)

        return _eye.reshape(-1), self.obj, _up.reshape(-1)

class Trajectory3DVisualizeWidget(QOpenGLWidget):
    def __init__(self, parent=None, fovy=40, height=1060, width=1920):
        QOpenGLWidget.__init__(self, parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        self.setFocusPolicy(Qt.StrongFocus)

        # initialize
        self.tracks = {}
        # 0: all, 1: First shot, 2: Second shot, ...
        self.current_shot = 0

        # set opengl gluPerspective
        self.fovy = fovy
        self.height = height
        self.width = width

        self.x = 0
        self.y = 0
        self.ctl = False
        self.translate = 0
        self.translate_z = 0
        self.show_axis = False
        self.show_location = False

        self.reset_btn = QPushButton('', self) # mode = 0
        self.reset_btn.move(850, 10)
        self.reset_btn.clicked.connect(self.onClickOrigin)
        self.reset_btn.resize(70, 70)
        size = QSize(62, 62)
        self.reset_btn.setIcon(QIcon(f'{ICONDIR}/reset.png'))
        self.reset_btn.setIconSize(size)

        self.top_btn = QPushButton('', self) # mode = 1
        self.top_btn.move(850, 90)
        self.top_btn.clicked.connect(self.onClickTop)
        self.top_btn.resize(70, 70)
        self.top_btn.setIcon(QIcon(f'{ICONDIR}/top.png'))
        self.top_btn.setIconSize(size)

        self.side_btn = QPushButton('', self) # mode = 2
        self.side_btn.move(850, 170)
        self.side_btn.clicked.connect(self.onClickSide)
        self.side_btn.resize(70, 70)
        self.side_btn.setIcon(QIcon(f'{ICONDIR}/side.png'))
        self.side_btn.setIconSize(size)

        self.show_axis_btn = QPushButton('', self)
        self.show_axis_btn.move(830, 520)
        self.show_axis_btn.clicked.connect(self.onClickShowAxis)
        self.show_axis_btn.resize(50, 50)
        self.show_axis_btn.setIcon(QIcon(f'{ICONDIR}/xyz.png'))
        self.show_axis_btn.setIconSize(size)

        self.show_location_btn = QPushButton('', self)
        self.show_location_btn.move(780, 520)
        self.show_location_btn.clicked.connect(self.onClickShowLocation)
        self.show_location_btn.resize(50, 50)
        self.show_location_btn.setIcon(QIcon(f'{ICONDIR}/location.png'))
        self.show_location_btn.setIconSize(size)

        self.ball = QLabel(self)
        self.ball.move(30, 20)
        self.ball.resize(100, 50)
        self.ball.setStyleSheet("color: white; background-color: #191919")
        self.ball.setFont(QFont('Times', 19))

        self.tf = Transform()
        self.scaled = 1
        self.mode = 0 # origin, top, side
        self.top_pt = 0
        self.side_pt = 0
        self.top_angle = [60, 0, -60, 0]
        self.side_angle = [-30, 0, 30, 0]
        # self.color_list = [
        #     [1,0,0],
        #     [0,1,0],
        #     [0,0,1],
        #     [1,1,0],
        #     [0,1,1],
        #     [1,0,1]
        # ]

    def initializeGL(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.1, 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Tell the Opengl, we are going to set PROJECTION function
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()  # always call this before gluPerspective
        # set intrinsic parameters (fovy, aspect, zNear, zFar)
        gluPerspective(self.fovy, self.width/self.height, 0.1, 100000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        _eye, _obj, _up = self.tf.rotate(self.tf.rad_z, self.tf.rad_y, self.tf.rad_x)
        gluLookAt(_eye[0], _eye[1], _eye[2],
                _obj[0], _obj[1], _obj[2],
                _up[0], _up[1], _up[2])
        glScalef(self.scaled, self.scaled, self.scaled)
        translate_x = (abs(_eye[1])/(abs(_eye[0])+abs(_eye[1])))*self.translate
        translate_y = (abs(_eye[0])/(abs(_eye[0])+abs(_eye[1])))*self.translate
        glTranslatef(translate_x, translate_y, self.translate_z)

        if self.show_axis:
            line_len = 1
            grid_color = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            glLineWidth(3) # set line width thicker
            origin = [0.0, 0.0, 0.0]
            for i in range(3):
                tmp = [0.0, 0.0, 0.0]
                tmp[i] = line_len*1.02
                glColor3f(*grid_color[i])
                glBegin(GL_LINES)
                glVertex3f(*origin)
                glVertex3f(*tmp)
                glEnd()
            glColor4ub(255, 0, 0, 255)
            glRasterPos3f(1.1, 0.0, 0.1)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord('x'))
            glColor4ub(0, 255, 0, 255)
            glRasterPos3f(0.0, 1.1, 0.1)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord('y'))
            glColor4ub(0, 0, 255, 255)
            glRasterPos3f(0.0, 0.0, 1.7)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord('z'))
            glLineWidth(1) # set line width back to default
        # Draw Model3D track

        total_tracks = len(self.tracks)
        if total_tracks > 0:
            if self.current_shot == 0:
                for shot, track in self.tracks.items():
                    self.drawTrack(track, shot)
                self.ball.setText('Total: {}'.format(total_tracks))
            elif self.current_shot <= total_tracks:
                track = self.tracks[self.current_shot]
                self.drawTrack(track, 0)
                self.ball.setText(f"shot: {self.current_shot}")
        else:
            self.ball.setText('Total: 0')

        # Draw badminton court
        drawCourt()
        drawNet()

        # Draw Plane (not pushed to git)
        if self.current_shot > 0 and self.current_shot <= total_tracks:
            np_points = np.stack([p.toXYZT() for p in self.tracks[self.current_shot] if p.visibility == 1], axis=0)
            _, curve_3d2d, slope, intercept = FitVerticalPlaneTo2D(np_points, smooth_2d=True, smooth_2d_x_accel=True)

            glBegin(GL_QUADS)
            glColor4ub(255, 0, 0, 60)
            glVertex3f(curve_3d2d[0][0], curve_3d2d[0][1], 0)
            glVertex3f(curve_3d2d[0][0], curve_3d2d[0][1], 5)
            glVertex3f(curve_3d2d[-1][0], curve_3d2d[-1][1], 5)
            glVertex3f(curve_3d2d[-1][0], curve_3d2d[-1][1], 0)
            glEnd()



    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fovy, w / h, 0.1, 100000.0)
        # gluPerspective(40, w / h, 0.1, 100000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def drawTrack(self, track, shot):
        pre_points = []
        MAX_TIME_DELAY = 0.5
        # color = shot % len(self.color_list)
        for point in track:
            size = 0.02
            self.sphere(point.fid, point.x, point.y, point.z, color=point.color, size = size)
            if len(pre_points) != 0:
                pre_point = pre_points.pop(0)
                if abs(pre_point.timestamp - point.timestamp) <= MAX_TIME_DELAY:
                    glBegin(GL_LINES)
                    glVertex3f(pre_point.x, pre_point.y, pre_point.z)
                    glVertex3f(point.x, point.y, point.z)
                    glEnd()
            pre_points.append(point)

    def sphere(self, fid, x, y, z, color, size=0.02):
        if color == 'red':
            color = (255, 0, 0)
        if color == 'orange':
            color = (255, 165, 0)
        elif color == 'yellow':
            color = (255, 255, 0)
        elif color == 'green':
            color = (0, 255, 0)
        elif color == 'light blue':
            color = (173, 216, 230)
        elif color == 'cyan':
            color = (0, 255, 255)
        elif color == 'blue':
            color = (0, 0, 255)
        elif color == 'indigo':
            color = (75, 0, 130)
        elif color == 'purple':
            color = (128, 0, 128)
        elif color == 'violet':
            color = (238, 130, 238)
        elif color == 'magenta':
            color = (255, 0, 255)
        elif color == 'pink':
            color = (255, 192, 203)
        elif color == 'white':
            color = (255, 255, 255)
        elif color == 'gray' or color == 'grey':
            color = (128, 128, 128)
        elif color == 'black':
            color = (0, 0, 0)
        glColor3ub(color[0], color[1], color[2])
        glTranslatef(x, y, z)
        gluSphere(self.tf.quadric, size, 32, 32)
        glTranslatef(-x, -y, -z)
        if self.show_location == True:
            glColor4ub(color[0], color[1], color[2], 255)
            glRasterPos3f(x+0.3 , y, z)
            self.DrawText('{}:({},{},{})'.format(fid, round(x,2), round(y,2), round(z,2)))

    def DrawText(self, string):
        for c in string:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(c))

    def addPointByTrackID(self, point, id):
        if id not in self.tracks.keys():
            self.tracks[id] = []
        self.tracks[id].append(point)

    def clearAll(self):
        self.tracks.clear()
        self.current_shot = 0
        # self.setRad(0, 0, 0)
        self.mode = 0
        self.scaled = 1

    def setShot(self, shot):
        if len(self.tracks) == 0:
            self.current_shot = 0
            return
        if shot < 0:
            shot = len(self.tracks) - 1
        if shot > len(self.tracks):
            shot = 0
        self.current_shot = shot

    def getShot(self):
        return self.current_shot

    def onClickOrigin(self):
        self.mode = 0
        self.scaled = 1
        self.setRad(0, 0, 0)
        self.update
        self.translate = 0
        self.translate_z = 0
    def onClickTop(self):
        self.mode = 1
        self.scaled = 1
        self.top_pt = 0
        self.setRad(0, toRad(60), toRad(-90))
        self.update

    def onClickSide(self):
        self.mode = 2
        self.scaled = 1
        self.side_pt = 0
        self.setRad(0, toRad(-28), toRad(-86)) # Slightly tilt, to see the trajectory pass above the net or not
        self.update
    def onClickShowAxis(self):
        if self.show_axis:
            self.show_axis = False
        else:
            self.show_axis = True
    def onClickShowLocation(self):
        if self.show_location:
            self.show_location = False
        else:
            self.show_location = True
    def mousePressEvent (self, event):      #click mouse
        self.x = event.x()
        self.y = event.y()
    def mouseMoveEvent(self, event):        #click and move mouse
        if event.buttons() and Qt.LeftButton:
            if not self.ctl:
                if int(event.x()) > int(self.x):
                    if self.mode == 0 or self.mode == 2: # origin
                        self.addRad(0, 0, toRad(-5))
                    elif self.mode == 1: # top
                        self.scaled = 1
                        self.top_pt -= 1
                        self.top_pt %= 4
                        y = self.top_angle[self.top_pt]
                        x = self.top_angle[(self.top_pt + 1) % 4]
                        self.setRad(toRad(x), toRad(y), self.tf.rad_z + toRad(-90))
                    self.x = event.x()
                    self.y = event.y()
                    self.update
                else:
                    if self.mode == 0 or self.mode == 2: # origin
                        self.addRad(0, 0, toRad(5))
                    elif self.mode == 1: # top
                        self.scaled = 1
                        self.top_pt += 1
                        self.top_pt %= 4
                        y = self.top_angle[self.top_pt]
                        x = self.top_angle[(self.top_pt + 1) % 4]
                        self.setRad(toRad(x), toRad(y), self.tf.rad_z + toRad(90))
                    self.x = event.x()
                    self.y = event.y()
                    self.update
            else:
                self.translate = (event.x()-self.x)/10
                self.translate_z = (event.y()-self.y)/10
    def wheelEvent(self, event):
        angle = event.angleDelta() / 8
        angleX = angle.x()
        angleY = angle.y()
        if angleY > 0:
            if self.scaled < 4:
                self.scaled += 0.1
                self.update
        else:
            if self.scaled > 0.5:
                self.scaled -= 0.1
                self.update
    # delete when finish
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_C:
            print('exit...')
            os._exit(0)
        elif e.key() == Qt.Key_D:
            self.addRad(0, 0, toRad(5))
            self.update
        elif e.key() == Qt.Key_A:
            self.addRad(0, 0, toRad(-5))
            self.update
        elif e.key() == Qt.Key_W:
            self.setRad(0, toRad(60), toRad(-90))
            self.update
        elif e.key() == Qt.Key_S:
            self.setRad(0, toRad(-30), toRad(-90))
            self.update
        elif e.key() == Qt.Key_R:
            self.setRad(0, 0, 0)
            self.update
        elif e.key() == Qt.Key_Q:
            print(self.tf.rad_x)
            print(self.tf.rad_y)
            print(self.tf.rad_z)
            self.update
        elif e.key() == Qt.Key_K:
            if self.scaled < 2:
                self.scaled += 0.1
                self.update
        elif e.key() == Qt.Key_L:
            if self.scaled > 0.5:
                self.scaled -= 0.1
                self.update
        elif e.key() == Qt.Key_Control:
            self.ctl = True
        elif e.key() == Qt.Key_Z:
            current = self.getShot()
            self.setShot(current-1)
        elif e.key() == Qt.Key_X:
            current = self.getShot()
            self.setShot(current+1)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.ctl = False

    def setRad(self, x, y, z):
        self.tf.rad_x = x
        self.tf.rad_x %= 2*pi
        self.tf.rad_y = y
        self.tf.rad_y %= 2*pi
        self.tf.rad_z = z
        self.tf.rad_z %= 2*pi

    def addRad(self, x, y, z):
        self.tf.rad_x += x
        self.tf.rad_x %= 2*pi
        self.tf.rad_y += y
        self.tf.rad_y %= 2*pi
        self.tf.rad_z += z
        self.tf.rad_z %= 2*pi
