import os
import sys
import numpy as np
import logging
import csv
import shutil
import torch
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QComboBox, QGroupBox, QPushButton, QGridLayout, QCheckBox
from PyQt5.QtCore import Qt, pyqtSignal, QSize
'''
Our common function
'''
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ICONDIR = f"{ROOTDIR}/UI/icon"
sys.path.append(f"{ROOTDIR}/lib")
from point import Point, load_points_from_csv
from writer import CSVWriter

'''
RNN
'''
sys.path.append(f"{ROOTDIR}/RNN")
from blstm import Blstm
from threeDprojectTo2D import FitVerticalPlaneTo2D
'''
other UI component
'''
from Trajectory3DVisualizeWidget import Trajectory3DVisualizeWidget

class VisualizeReplayPage(QGroupBox):
    def __init__(self, replay_path):
        super().__init__()
        # setup UI
        self.setFixedSize(1400,800)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # content
        self.replay_path = replay_path
        # initialize layout
        layout =  QGridLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(10, 10, 10, 10)

        # component
        # Setup virtual court for trajectory visualized
        self.widget_court = Trajectory3DVisualizeWidget()
        self.widget_court.setMinimumSize(950, 700)
        # Combobox for choose the tracks of list in "replay" folder
        self.cbo_replay = QComboBox()
        self.cbo_replay.setFixedSize(950,50)

        self.cbo_replay.currentTextChanged.connect(self.cbo_replay_changed)

        self.date_choose = self.cbo_replay.currentText()

        # change the court to show next shot
        self.lastbtn = QPushButton(self)
        self.lastbtn.clicked.connect(self.onClickLast)
        self.lastbtn.setFixedSize(QSize(100,160))
        self.lastbtn.setIcon(QIcon(f"{ICONDIR}/left_arrow.png"))
        self.lastbtn.setIconSize(QSize(95,95))
        # change the court to show last shot
        self.nextbtn = QPushButton(self)
        self.nextbtn.clicked.connect(self.onClickNext)
        self.nextbtn.setFixedSize(QSize(100,160))
        self.nextbtn.setIcon(QIcon(f"{ICONDIR}/right_arrow.png"))
        self.nextbtn.setIconSize(QSize(95,95))

        ### Task
        self.tasks = [(QCheckBox('Model3D.csv (white)')       , 'white','Model3D.csv'),
                      (QCheckBox('vicon.csv (red)')          , 'red'   ,'vicon.csv'),
                      (QCheckBox('smooth_3d2d.csv (orange)') ,'orange','smooth_3d2d.csv'),
                      (QCheckBox('nosmooth_3d2d.csv (blue)'), 'blue','nosmooth_3d2d.csv'),
                      (QCheckBox('physics_3d2d.csv (cyan)'), 'cyan','physics_3d2d.csv'),
                      (QCheckBox('smooth_3d.csv (black)'), 'black','smooth_3d.csv'),
                      (QCheckBox('Predict3d.csv (gray)'), 'gray','Predict3d.csv')]
        vlayout = QVBoxLayout()

        for t in self.tasks:
            t[0].setChecked(True)
            vlayout.addWidget(t[0])
            t[0].clicked.connect(self.onCheckBoxClick)
        # setup Layout
        layout.addWidget(self.cbo_replay, 0, 1)
        layout.addLayout(vlayout, 0, 3)
        layout.addWidget(self.lastbtn, 1, 0)
        layout.addWidget(self.widget_court, 1, 1)
        layout.addWidget(self.nextbtn, 1, 2)

        layout.setAlignment(Qt.AlignCenter)
        # set layout
        self.setLayout(layout)
        self.refreshReplaylist()


    def onCheckBoxClick(self):
        self.showTrajectories()

    def cbo_replay_changed(self):
        self.date_choose = self.cbo_replay.currentText()
        self.showTrajectories()

    def refreshReplaylist(self):
        # replay list
        self.cbo_replay.clear()
        replay_dir = [f for f in os.listdir(self.replay_path) ] # if file is exist then add to cbo
        replay_dir = sorted(replay_dir)
        self.cbo_replay.addItem('')
        self.cbo_replay.addItems(replay_dir)

    def showTrajectories(self):
        self.widget_court.clearAll()
        if self.date_choose == '':
            return

        tracksID = 1
        for t in self.tasks:
            if not t[0].isChecked():
                continue
            f = os.path.join(self.replay_path, self.date_choose, t[2])
            if not os.path.exists(f):
                continue
            points = load_points_from_csv(f)
            for p in points:
                p.color = t[1]
                if p.visibility == 1:
                    self.widget_court.addPointByTrackID(p, tracksID)

            tracksID += 1

    def onClickNext(self):
        shot = self.widget_court.getShot()
        self.widget_court.setShot(shot + 1)

    def onClickLast(self):
        shot = self.widget_court.getShot()
        self.widget_court.setShot(shot - 1)

    def showEvent(self, event):
        self.refreshReplaylist()
