import os
import sys
import argparse
import cv2
import configparser
import numpy as np
import json
import time
import shutil
import math

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *

from PyQt5.QtGui import QIcon, QPalette, QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow,QApplication, QWidget, QPushButton, QMessageBox, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QStyle \
, QSizePolicy, QFileDialog, QCheckBox, QComboBox, QGridLayout, QScrollArea, QFormLayout, QProgressBar, QDialog,QSpinBox,QDoubleSpinBox,QAction \
, QToolBar, QGroupBox

from PyQt5.QtCore import Qt, QUrl, QSize, QProcess, QThread, QFileInfo

'''
Our common function
'''
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)

sys.path.append(f"{ROOTDIR}/lib")
sys.path.append(f"{ROOTDIR}/UI")

from VisualizeReplayPage import VisualizeReplayPage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'See Trajectoies in OpenGL')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    args = parser.parse_args()
    app = QApplication(sys.argv)
    page = VisualizeReplayPage(args.folder)
    page.show()
    sys.exit(app.exec_())
