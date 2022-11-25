import os
import sys
import logging
import configparser
from functools import partial

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, \
                            QSizePolicy, QDialog, QGroupBox, QMessageBox, QGridLayout
from PyQt5.QtCore import Qt, QUrl, QSize


DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ICONDIR = f"{ROOTDIR}/UI/icon"
sys.path.append(f"{ROOTDIR}/lib")
from common import updateLastExeDate, loadConfig
from nodes import setupCameras


sys.path.append(f"{ROOTDIR}/UI")
from Services import SystemService, MsgContract


from Recorder import Recorder

#####################

PAGE_SIZE = QSize(1800,1000)
PROJECT = '../projects/otuspect.cfg'

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # set desktop icon
        self.setWindowIcon(QIcon(f"{ICONDIR}/desktop.png"))

        # initialize
        self.pages = {}
        self.shown_page_name = ''
        self.myService = None

        # loading project config
        cfg_file = f"../projects/otuspect.cfg"
        cfg = loadConfig(cfg_file)

        # setup UI
        self.setupUI(cfg)

    def closeEvent(self, event):
        if self.myService.isRunning():
            logging.debug("stop system service")
            self.myService.stop()
        logging.debug("closeEvent")
        QMessageBox.warning(None, 'warning', '即將離開程式')

    def setupUI(self, cfg):
        self.layout_main = QGridLayout()
        mainWidget = QWidget()
        mainWidget.setLayout(self.layout_main)
        self.setCentralWidget(mainWidget)

        # setup Cameras Node of Project
        cameras = setupCameras('coachbox', cfg)
        # start Camera Node
        self.addNodes(cameras)
        # Page :: Index
        self.recorder_page = Recorder()
        self.addPage("Recorder", self.recorder_page)

    def addPage(self, name, page:QGroupBox):
        page.hide()
        page.setBackgroundService(self.myService)
        page.setFixedSize(PAGE_SIZE)

        self.layout_main.addWidget(page, 0, 1)
        if name in self.pages:
            del self.pages[name]
        self.pages[name] = page

    def showPage(self, name):
        logging.debug(f"{self.__class__.__name__}: showPage -> {name}")
        if self.pages[name].isHidden():
            if name in self.pages:
                if self.shown_page_name != '':
                    self.pages[self.shown_page_name].hide()
                self.pages[name].setStyleSheet("background-color: #DDD9C3")
                self.pages[name].show()
                self.shown_page_name = name
            else:
                logging.warrning(f"Page {name} is not exist.")


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
    updateLastExeDate()
    app = QApplication(sys.argv)
    window = Recorder()
    window.showMaximized()
    sys.exit(app.exec_())
