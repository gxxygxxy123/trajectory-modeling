import json
import logging
import os
import sys
import cv2
import numpy as np
from PyQt5.QtCore import QLine, QPoint, QRectF, QSize, Qt
from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QPushButton, QWidget, QComboBox)

'''
Our common function
'''
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ICONDIR = f"{ROOTDIR}/UI/icon"
sys.path.append(f"{ROOTDIR}/lib")
from common import loadConfig ,saveConfig
from h2pose.H2Pose import H2Pose
from h2pose.Hfinder import Hfinder

sys.path.append(f"{ROOTDIR}/UI")
from UISettings import UIStyleSheet

class CameraExtrinsic(QDialog):

    # std_court.png : (400,852)
    std_corner_x=[18, 48, 200, 351, 382] # corner pixel on court picture
    std_corner_y=[20, 70, 306, 548, 784, 831]

    court_picture_corner_x = [] # corner position on picture (to put point1~4.png)
    court_picture_corner_y = []

    actual_std_corner_x = [-3.05,-3.01,-2.59,-2.55,-0.02,0.02,2.55,2.59,3.01,3.05] # corner position in reality (unit: meter)
    actual_std_corner_y = [6.7,6.66,5.94,5.9,2.02,1.98,-1.98,-2.02,-5.9,-5.94,-6.66,-6.7]


    def __init__(self, cfg_path, camera_img):
        super().__init__()
        # Style
        self.setGeometry(0,0,620,852)
        self.setWindowTitle("Camera Extrinsic")

        # setup parameter
        #self.cameraUI = cameraUI
        self.cfg_path = cfg_path
        self.cameraCfg = loadConfig(cfg_path)
        self.camera_ks = np.array(json.loads(self.cameraCfg['Other']['ks']))
        self.dist = np.array(json.loads(self.cameraCfg['Other']['dist']))
        self.nmtx = np.array(json.loads(self.cameraCfg['Other']['newcameramtx']))
        self.camera_img = camera_img

        # Layout
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.court3D = [] # corner position in reality (unit: meter)

        self.mousePressEvent=self.dialogPressEvent

        self.show_star = True
        self.show_star_on_zoom = False

        self.zoom_court_picture = QLabel(self)
        self.zoom_width = 200
        self.zoom_height = 200
        self.zoom_court_picture.setGeometry(410,326,self.zoom_width,self.zoom_height)
        self.zoom_court_picture.setStyleSheet("border: 1px solid black;")
        self.zoom_court_picture.mousePressEvent=self.zoomPicturePressEvent

        self.zoom_option = [False,False,False,False] # left-top right-top left-bot right-bot

        self.x_idx = -1
        self.y_idx = -1

        self.MAX_POINTS = 80
        self.POINTS_ICON_WIDTH = 15
        self.POINTS_ICON_HEIGHT = 15
        self.points_icon = [QLabel(self) for i in range(self.MAX_POINTS)]
        for idx,p in enumerate(self.points_icon):
            p.hide()
            self.set_image(cv2.imread(f"{ICONDIR}/court/point{idx+1}.png"),p,target_width=self.POINTS_ICON_WIDTH,target_height=self.POINTS_ICON_HEIGHT)

        self.qline1 = QLine()
        self.qline2 = QLine()
        self.qline3 = QLine()
        self.qline4 = QLine()
        self.reset4Lines()

        self.btn_clear = QPushButton('重來',self)
        self.btn_clear.setFixedSize(50,30)
        self.btn_clear.move(560,702)
        self.btn_clear.clicked.connect(self.clear_all)

        self.btn_confirm = QPushButton('確認',self)
        self.btn_confirm.setStyleSheet("background-color: green")
        self.btn_confirm.setFont(QFont('Times', 39))
        self.btn_confirm.setFixedSize(200,100)
        self.btn_confirm.move(410,742)
        self.btn_confirm.clicked.connect(self.click_coordiate)
        self.btn_confirm.setEnabled(False)

    def paintEvent(self,e):
        painter = QPainter()
        painter.begin(self)
        painter.drawPixmap(QRectF(0,0,400,852),QPixmap(f"{ICONDIR}/court/std_court.png"),QRectF(0,0,400,852))
        if self.show_star:
            for x in self.std_corner_x:
                for y in self.std_corner_y:
                    painter.drawPixmap(QRectF(x-15,y-15,30,30),QPixmap(f"{ICONDIR}/court/star.png"),QRectF(0,0,640,640))
            painter.drawPixmap(QRectF(410,116,200,200),QPixmap(f"{ICONDIR}/court/hint_click_court.png"),QRectF(0,0,256,256))
        if self.show_star_on_zoom:
            painter.drawPixmap(QRectF(480,396,20,20),QPixmap(f"{ICONDIR}/court/star.png"),QRectF(0,0,640,640))
            painter.drawPixmap(QRectF(520,396,20,20),QPixmap(f"{ICONDIR}/court/star.png"),QRectF(0,0,640,640))
            painter.drawPixmap(QRectF(480,436,20,20),QPixmap(f"{ICONDIR}/court/star.png"),QRectF(0,0,640,640))
            painter.drawPixmap(QRectF(520,436,20,20),QPixmap(f"{ICONDIR}/court/star.png"),QRectF(0,0,640,640))
            painter.drawPixmap(QRectF(410,116,200,200),QPixmap(f"{ICONDIR}/court/hint_click_zoom.png"),QRectF(0,0,256,256))
        pen = QPen(Qt.black,1,Qt.DashLine)
        painter.setPen(pen)
        if self.y_idx >= 0 and self.y_idx <= 2 and self.show_star_on_zoom:
            painter.drawLine(self.qline2)
            painter.drawLine(self.qline3)
        if self.y_idx >= 3 and self.y_idx <= 5 and self.show_star_on_zoom:
            painter.drawLine(self.qline1)
            painter.drawLine(self.qline4)
        painter.end()

    def reset4Lines(self):
        self.qline1.setP1(QPoint(410,326))
        self.qline1.setP2(QPoint(410,326))
        self.qline2.setP1(QPoint(410,526))
        self.qline2.setP2(QPoint(410,526))
        self.qline3.setP1(QPoint(610,326))
        self.qline3.setP2(QPoint(610,326))
        self.qline4.setP1(QPoint(610,526))
        self.qline4.setP2(QPoint(610,526))

    def dialogPressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            court_img = cv2.imread(f"{ICONDIR}/court/std_court.png")
            self.mouse_x_left = event.pos().x()
            self.mouse_y_left = event.pos().y()

            if self.mouse_x_left > 400 or self.mouse_y_left > 852:
                return

            x_array = np.asarray(self.std_corner_x)
            y_array = np.asarray(self.std_corner_y)
            self.x_idx = (np.abs(x_array-self.mouse_x_left)).argmin()
            self.y_idx = (np.abs(y_array-self.mouse_y_left)).argmin()

            zoom_img = None

            if abs(self.std_corner_x[self.x_idx]-self.mouse_x_left) <= 16 and abs(self.std_corner_y[self.y_idx]-self.mouse_y_left) <= 16:
                if self.x_idx == 0:
                    if self.y_idx == 0:
                        zoom_img = cv2.imread(f"{ICONDIR}/corner/corner8_star.png")
                        self.zoom_option = [True,False,False,True]
                    elif self.y_idx == 5:
                        zoom_img = cv2.imread(f"{ICONDIR}/corner/corner7_star.png")
                        self.zoom_option = [False,True,True,False]
                    else:
                        zoom_img = cv2.imread(f"{ICONDIR}/corner/corner2_star.png")
                        self.zoom_option = [False,True,False,True]
                elif self.x_idx == 4:
                    if self.y_idx == 0:
                        zoom_img = cv2.imread(f"{ICONDIR}/corner/corner9_star.png")
                        self.zoom_option = [False,True,True,False]
                    elif self.y_idx == 5:
                        zoom_img = cv2.imread(f"{ICONDIR}/corner/corner6_star.png")
                        self.zoom_option = [True,False,False,True]
                    else:
                        zoom_img = cv2.imread(f"{ICONDIR}/corner/corner4_star.png")
                        self.zoom_option = [True,False,True,False]
                else:
                    if self.y_idx == 0 or (self.x_idx == 2 and self.y_idx == 3):
                        zoom_img = cv2.imread(f"{ICONDIR}/corner/corner3_star.png")
                        self.zoom_option = [False,False,True,True]
                    elif self.y_idx == 5 or (self.x_idx == 2 and self.y_idx == 2):
                        zoom_img = cv2.imread(f"{ICONDIR}/corner/corner5_star.png")
                        self.zoom_option = [True,True,False,False]
                    else:
                        zoom_img = cv2.imread(f"{ICONDIR}/corner/corner1_star.png")
                        self.zoom_option = [True,True,True,True]

                self.set_image(zoom_img, self.zoom_court_picture,target_width=self.zoom_width,target_height=self.zoom_height)
                self.show_star_on_zoom = True
            else:
                self.zoom_option = [False,False,False,False]

            self.qline1.setP1(QPoint(self.std_corner_x[self.x_idx],self.std_corner_y[self.y_idx]))
            self.qline2.setP1(QPoint(self.std_corner_x[self.x_idx],self.std_corner_y[self.y_idx]))
            self.qline3.setP1(QPoint(self.std_corner_x[self.x_idx],self.std_corner_y[self.y_idx]))
            self.qline4.setP1(QPoint(self.std_corner_x[self.x_idx],self.std_corner_y[self.y_idx]))
            self.update()
        elif event.buttons() == Qt.RightButton:
            pass

    def zoomPicturePressEvent(self,event):
        if event.buttons() == Qt.LeftButton:
            if self.show_star_on_zoom == False:
                return
            if len(self.court3D) < self.MAX_POINTS:
                court_img = cv2.imread(f"{ICONDIR}/court/std_court.png")
                self.mouse_x = event.pos().x()
                self.mouse_y = event.pos().y()

                pos_x = self.std_corner_x[self.x_idx] # corner pixel position
                pos_y = self.std_corner_y[self.y_idx]

                x_right = 1
                y_bot = 1

                if self.mouse_x < self.zoom_width/2 and self.mouse_y < self.zoom_height/2 and self.zoom_option[0]:   # Top Left
                    pos_x -= self.POINTS_ICON_WIDTH
                    pos_y -= self.POINTS_ICON_HEIGHT
                    x_right = 0
                    y_bot = 0
                elif self.mouse_x > self.zoom_width/2 and self.mouse_y < self.zoom_height/2 and self.zoom_option[1]: # Top Right
                    pos_y -= self.POINTS_ICON_HEIGHT
                    y_bot = 0
                elif self.mouse_x < self.zoom_width/2 and self.mouse_y > self.zoom_height/2 and self.zoom_option[2]: # Bottom Left
                    pos_x -= self.POINTS_ICON_WIDTH
                    x_right = 0

                elif self.mouse_x > self.zoom_width/2 and self.mouse_y > self.zoom_height/2 and self.zoom_option[3]: # Bottom Right
                    pass
                else:
                    return

                actual_x = self.actual_std_corner_x[self.x_idx*2+x_right]
                actual_y = self.actual_std_corner_y[self.y_idx*2+y_bot]
                if [actual_x,actual_y] in self.court3D:
                    reply = QMessageBox.information(self, 'Info', '這個點你已經選過了！', QMessageBox.Ok, QMessageBox.Ok)
                else:
                    self.points_icon[len(self.court3D)].show()
                    self.points_icon[len(self.court3D)].setGeometry(pos_x,pos_y,self.POINTS_ICON_WIDTH,self.POINTS_ICON_HEIGHT)

                    self.court3D.append([actual_x,actual_y])

                    logging.debug("The {}th actual court position: ({},{})".format(len(self.court3D),actual_x,actual_y))

                    self.x_idx = -1
                    self.y_idx = -1
                    self.zoom_court_picture.clear()

                    self.show_star_on_zoom = False

                    self.update()

                    if len(self.court3D) >= 4:
                        self.btn_confirm.setEnabled(True)

        elif event.buttons() == Qt.RightButton:
            pass

    def click_coordiate(self):
        print("Actual Points Coordinate\n{}\n".format(self.court3D))
        ''' [TODO] Pseudo
        if(self.project_info['Model3D']['node_type']=='Pseudo3D'):
            court2D = []
            Kmtx = self.cameras_ks[0]

            hf = Hfinder(self.cameras_img[0], court2D=court2D, court3D=self.court3D)
            Hmtx = hf.getH()
            self.project_info['Model3D']['Hmtx'] = str(Hmtx.tolist())

        else:
        '''
        print("===== Old =====")
        print(f"Poses: {json.loads(self.cameraCfg['Other']['poses'])}")
        print(f"Eye: {json.loads(self.cameraCfg['Other']['eye'])}")
        print(f"Hmtx: {json.loads(self.cameraCfg['Other']['hmtx'])}")
        print(f"projection_mat: {json.loads(self.cameraCfg['Other']['projection_mat'])}")
        print(f"extrinsic_mat: {json.loads(self.cameraCfg['Other']['extrinsic_mat'])}")
        print("\n")

        hf = Hfinder(camera_ks=self.camera_ks, dist=self.dist, nmtx=self.nmtx, img=self.camera_img, court3D=self.court3D)
        Hmtx = hf.getH()
        Kmtx = self.nmtx
        projection_mat = hf.getProjection_mat()
        extrinsic_mat = hf.getExtrinsic_mat()
        h2p = H2Pose(Kmtx, Hmtx)
        # poses = np.array(json.loads(self.cameraCfg['Other']['poses']))
        # eye = np.array(json.loads(self.cameraCfg['Other']['eye']))
        poses = h2p.getC2W()
        eye = h2p.getCamera().T
        eye[0][2] = abs(eye[0][2])

        # [TODO] Check if keys not in [Other] Section

        self.cameraCfg['Other']['poses'] = str(poses.tolist())
        self.cameraCfg['Other']['eye'] = str(eye.tolist())
        self.cameraCfg['Other']['hmtx'] = str(Hmtx.tolist())
        self.cameraCfg['Other']['projection_mat'] = str(projection_mat.tolist())
        self.cameraCfg['Other']['extrinsic_mat'] = str(extrinsic_mat.tolist())


        print("===== New =====")
        print(f"Poses: {json.loads(self.cameraCfg['Other']['poses'])}")
        print(f"Eye: {json.loads(self.cameraCfg['Other']['eye'])}")
        print(f"Hmtx: {json.loads(self.cameraCfg['Other']['hmtx'])}")
        print(f"projection_mat: {json.loads(self.cameraCfg['Other']['projection_mat'])}")
        print(f"extrinsic_mat: {json.loads(self.cameraCfg['Other']['extrinsic_mat'])}")
        print("\n")

    def clear_all(self):
        self.court3D.clear()
        self.btn_confirm.setEnabled(False)
        for p in self.points_icon:
            p.hide()

    def set_image(self, image, label, target_width, target_height):
        height, width, channel = image.shape

        bytesPerLine = 3 * width
        qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped().scaled(QSize(target_width,target_height),Qt.KeepAspectRatio)


        label.setPixmap(QPixmap(qimage))

    def closeEvent(self, event):
        pass
        # Save Poses and Eye and Hmtx to origin camera config file [Other]['poses'], [Other]['eye']
        saveConfig(self.cfg_path, self.cameraCfg)
        print(f"Save Config: {self.cfg_path} Successfully.\n")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.root_replay_path = f'{ROOTDIR}/replay'
        self.refreshReplaylist()

    def setupUI(self):
        layout_main = QGridLayout()
        mainWidget = QWidget()
        mainWidget.setLayout(layout_main)
        self.setCentralWidget(mainWidget)

        self.cbo_replay_date = QComboBox()
        self.cbo_replay_date.setFixedSize(700,50)
        self.cbo_replay_date.currentTextChanged.connect(self.cbo_replay_date_changed)

        self.cbo_replay_avi = QComboBox()
        self.cbo_replay_avi.setFixedSize(300,50)
        self.cbo_replay_avi.currentTextChanged.connect(self.cbo_replay_avi_changed)

        self.labelScreenShot = QLabel()
        pixmap = QPixmap(f'{ICONDIR}/no_camera.png')
        scaled_pixmap = QPixmap()
        scaled_pixmap = pixmap.scaled(1000, 750)
        self.labelScreenShot.setPixmap(scaled_pixmap)

        self.buttonStart = QPushButton('Start')
        self.buttonStart.setFixedSize(QSize(200, 100))
        self.buttonStart.setStyleSheet("background-color: #808080; color: #FFFFFF ; border-style: outset ; font: bold 35px")
        self.buttonStart.setEnabled(False)
        self.buttonStart.clicked.connect(self.startExtrinctic)

        layout_main.addWidget(self.cbo_replay_date, 0, 0, 1, 7)
        layout_main.addWidget(self.cbo_replay_avi, 0, 7, 1, 3)
        layout_main.addWidget(self.labelScreenShot, 1, 4, 1, 2)
        layout_main.addWidget(self.buttonStart, 1, 8, 1, 2, Qt.AlignBottom)

    def startExtrinctic(self):
        print(f"Config: {self.cfg_path}\n")
        court_img = cv2.imread(self.screenshot_path)
        dialog = CameraExtrinsic(self.cfg_path, court_img)
        dialog.exec()

    def cbo_replay_date_changed(self):
        if self.cbo_replay_date.currentText() == "":
            return
        if os.path.exists(os.path.join(self.root_replay_path, self.cbo_replay_date.currentText(), 'config')):
            cfg_file = os.path.join(self.root_replay_path, self.cbo_replay_date.currentText(), 'config')
        elif os.path.exists(os.path.join(self.root_replay_path, self.cbo_replay_date.currentText(), 'coachbox.cfg')):
            cfg_file = os.path.join(self.root_replay_path, self.cbo_replay_date.currentText(), 'coachbox.cfg')
        elif os.path.exists(os.path.join(self.root_replay_path, self.cbo_replay_date.currentText(), 'otuspect.cfg')):
            cfg_file = os.path.join(self.root_replay_path, self.cbo_replay_date.currentText(), 'otuspect.cfg')
        else:
            print("No config/coachbox.cfg/otuspect.cfg in folder")
            sys.exit(1)
        self.cfg = loadConfig(cfg_file)
        self.cbo_replay_avi.clear()
        replay_dir = []
        for f in os.listdir(os.path.join(self.root_replay_path, self.cbo_replay_date.currentText())):
            if 'avi' in f:
                replay_dir.append(f)
        self.cbo_replay_avi.addItem("")
        self.cbo_replay_avi.addItems(replay_dir)


    def cbo_replay_avi_changed(self):
        if self.cbo_replay_avi.currentText() == "":
            return
        video_path = os.path.join(self.root_replay_path, self.cbo_replay_date.currentText(), self.cbo_replay_avi.currentText())
        camera_name = self.cbo_replay_avi.currentText().split('.')[0]
        self.screenshot_path = os.path.join(self.root_replay_path, self.cbo_replay_date.currentText(), camera_name+'.jpg')
        self.cfg_path = os.path.join(self.root_replay_path, self.cbo_replay_date.currentText(),
                                            self.cfg[camera_name]['hw_id']+'.cfg')

        if not os.path.exists(self.screenshot_path):
            cap = cv2.VideoCapture(video_path)
            success, img = cap.read()
            if success:
                cv2.imwrite(self.screenshot_path, img)
                cap.release()

        pixmap = QPixmap(self.screenshot_path)
        scaled_pixmap = QPixmap()
        scaled_pixmap = pixmap.scaled(1000, 750)
        self.labelScreenShot.setPixmap(scaled_pixmap)

        self.buttonStart.setStyleSheet("background-color: #2894FF; color: #FFFFFF ; border-style: outset ; font: bold 35px")
        self.buttonStart.setEnabled(True)

    def refreshReplaylist(self):
        # replay list
        self.cbo_replay_date.clear()
        replay_dir = []
        for (root, dirs, files) in os.walk(self.root_replay_path):
            if 'config' in files:
                replay_dir.append(root)

        replay_dir = sorted(replay_dir, reverse=True)
        self.cbo_replay_date.addItem("")
        self.cbo_replay_date.addItems(replay_dir)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showNormal()
    sys.exit(app.exec_())
