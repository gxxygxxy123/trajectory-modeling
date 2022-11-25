import os
import sys
import threading
import cv2
import logging
import configparser
import paho.mqtt.client as mqtt
import numpy as np
import json
import time
import shutil
import math
import paho.mqtt.client as mqtt

from PyQt5.QtGui import QIcon, QPalette, QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow,QApplication, QWidget, QPushButton, QMessageBox, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QStyle \
, QSizePolicy, QFileDialog, QCheckBox, QComboBox, QGridLayout, QScrollArea, QFormLayout, QProgressBar, QDialog,QSpinBox,QDoubleSpinBox,QAction \
, QToolBar, QGroupBox
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QSize, QProcess, QThread, QFileInfo

from functools import partial
from datetime import datetime
'''
Our common function
'''
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ICONDIR = f"{ROOTDIR}/UI/icon"
sys.path.append(f"{ROOTDIR}/lib")
from nodes import setupCameras
sys.path.append(f"{ROOTDIR}/UI")
from common import loadConfig
from camera import findCameraByProject
'''
other UI component
'''
from CameraPreviewPage import CameraPreviewPage
from CameraSettingPage import CameraSettingPage
from UISettings import UIFont, UIState
from VideoReplayPage import VideoReplayPage
'''
Inheritance our Application
'''
from Application import Application

class Recorder(Application):
    def __init__(self, cfg_file):
        super().__init__()
        # Content
        self.setWindowTitle(self.__class__.__name__)

        self.root_replay_path = f"{ROOTDIR}/replay"
        os.makedirs(self.root_replay_path, exist_ok=True)
        self.state = UIState.STOP
        # Load system config to APP's settings

        cfg = loadConfig(cfg_file)
        # connect MQTT Broker
        # self.connectMQTTBroker(self.settings['Default']['mqtt_broker'])
        '''
        Layout Setup : Page setup
        '''
        # Toolbar
        self.createToolbar()
        # setup Cameras of Project
        self.cameras = findCameraByProject(cfg_file)
        # Page : Preview Page
        self.pageRecord = CameraPreviewPage(self.cameras, self.slotPreviewClicked)
        # Page : Camera Setting Page default: first camera
        if len(self.cameras) > 0:
            self.pageCameraSetting = CameraSettingPage(self.cameras[0], self.slotSettingPreviewClicked)
        # Page : Replay
        self.pageReplay = VideoReplayPage(self)
        # start homepage
        self.addPage(self.pageRecord)
        self.addPage(self.pageCameraSetting)
        self.addPage(self.pageReplay)
        self.showPage(self.pageRecord)
        self.state = UIState.READY

    def createToolbar(self):
        # Start to Record
        self.btn_onStartRecord = self.add2Menu(self.onStartRecord,f"{ICONDIR}/startrecord.png")
        # Stop to Record
        self.btn_onStopRecord = self.add2Menu(self.onStopRecord,f"{ICONDIR}/stoprecord.png")
        self.btn_onStopRecord.setEnabled(False)
        # Replay older record file
        self.btn_replay = self.add2Menu(self.onReplay,f"{ICONDIR}/replay.png")
         # Performance
        #self.addPerformanceButton()
        #btnPerformance = createPerformanceButton()
        #self.add2Menu(btnPerformance)

    def onStartRecord(self):
        # change Page
        if self.pageRecord.isHidden():
            self.showPage(self.pageRecord)
            return
        # create folder
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        video_path = f"{self.root_replay_path}/{dt_string}"
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
        # send control to start recording
        self.startRecording(video_path)
        # UI settings
        self.state = UIState.PLAY
        self.btn_onStartRecord.setEnabled(False)
        self.btn_onStopRecord.setEnabled(True)
        self.btn_replay.setEnabled(False)

    def onStopRecord(self):
        # change Page
        if self.pageRecord.isHidden():
            self.showPage(self.pageRecord)
            return
        # send control to stop reocrding
        self.stopRecording()

        # UI settings
        self.state = UIState.READY
        self.btn_onStartRecord.setEnabled(True)
        self.btn_onStopRecord.setEnabled(False)
        self.btn_replay.setEnabled(True)

    def onReplay(self):
        # change Page
        if self.pageReplay.isHidden():
            self.showPage(self.pageReplay)
            self.pageReplay.refresh_cbo()
            return

    def closeEvent(self, event):
        super().closeEvent(event)
        self.stopStreaming()
        for c in self.cameras:
            c.onStop()

    # Preview clicked
    def slotPreviewClicked(self, camera_name):
        logging.debug("{} clicked".format(camera_name))
        if self.state == UIState.READY:
            # setup camera of Camera Settings Page
            for cam in self.cameras:
                if cam.name == camera_name:
                    self.pageCameraSetting.setCamera(cam)
            # change Camera Settings Page
            if self.pageCameraSetting.isHidden():
                self.showPage(self.pageCameraSetting)
    # Camera Setting Preview Clicked
    def slotSettingPreviewClicked(self, camera_name):
        logging.debug("{} clicked".format(camera_name))
        # change Page
        if self.state == UIState.READY:
            if self.pageRecord.isHidden():
                self.showPage(self.pageRecord)

if __name__ == '__main__':
    pass
