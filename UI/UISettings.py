import os
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QUrl, QSize
from PyQt5.QtWidgets import QPushButton, QLabel
from enum import Enum, auto

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ICONDIR = f"{ROOTDIR}/UI/icon"

class UIPageSize:
    Level1_WIDTH = 1800
    Level1_HEIGHT = 900
    Level2_WIDTH = 1600
    Level2_HEIGHT = 800

PAGE_SIZE = QSize(UIPageSize.Level1_WIDTH, UIPageSize.Level1_HEIGHT)

class UIComponentSize:
    ConfirmButton = QSize(160, 60)
    BackButton = QSize(160, 80)
    SettingButton = QSize(200, 100)
    SmallButton = QSize(50, 50)
    LineEdit = QSize(320, 60)
    LoingButton = QSize(600, 60)
    HistoryIconButton = QSize(50, 50)
    ActionButton = QSize(75, 50)

class UIStyleSheet:

    InfoFrame = "QGroupBox {border-style: solid; border-width: 2px; border-radius: 3px; border-color: rgb(0, 0, 0); background-color:transparent;}"
    CheckBox = 'QCheckBox { font: 15pt;} QCheckBox::indicator { width: 15px; height: 15px;};'
    InPageBox = "QGroupBox { border:none; background-color:transparent;}"
    LineEdit = "QLineEdit { background-color:white; font: 24px 'Times';}"
    ToolBar = "QToolBar { background-color:transparent; color : black; font: italic 16px 'Times'; }"

    RadioButton = 'QRadioButton { font: 15pt;} QRadioButton::indicator { width: 15px; height: 15px;};'
    HistoryIconButton = "QPushButton { background-color: transparent; color : white; font: bold 24px;  border-style: outset; \
        border-width: 2px; border-color: black; border-radius: 15px;}"
    ConfirmButton = "QPushButton { background-color: rgb(147,196,125); color : white; font: bold 24px;  border-style: outset; \
        border-width: 2px; border-color: black; border-radius: 15px;} \
        QPushButton::hover { background-color: rgb(167,216,145); color: black; }"
    CancelButton = "QPushButton { background-color: rgb(234,153,153); color : white; font: bold 24px;  border-style: outset; \
        border-width: 2px; border-color: black; border-radius: 15px;} \
        QPushButton::hover { background-color: rgb(214,133,133); color: black; }"
    RegisterButton = "QPushButton { background-color:  #DDD9C3; color : blue; font: bold 18px;  border: none; text-decoration: underline}"
    LoginButton = "QPushButton {  background-color: rgb(40,148,255); color: #FFFFFF; font: bold 30px;  border-style: outset; \
        border-width: 2px; border-color: black; border-radius: 15px;} \
        QPushButton::hover { background-color: rgb(60,168,255); color: black; }"
    ResetButton = "QPushButton { background-color: transparent; color : red; font: bold 18px;  border: none; text-decoration: underline}"
    CalculateButton = "QPushButton { background-color: transparent; color : blue; font: bold 18px;  border: none; text-decoration: underline}"
    MoreInfoButton = "QPushButton { background-color: rgb(132,215,252); color: #FFFFFF; font: bold 30px;  border-style: outset; \
        border-width: 2px; border-color: black; border-radius: 15px;} \
        QPushButton::hover { background-color: rgb(152,235,255); color: black; }"

    HSlider = "QSlider::groove:horizontal { border: 1px solid #bbb; background: white; height: 10px; border-radius: 4px;} \
                        QSlider::sub-page:horizontal { background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1, stop: 0 #66e, stop: 1 #bbf); \
                         background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1, stop: 0 #bbf, stop: 1 #55f); border: 1px solid #777; \
                         height: 10px; border-radius: 4px; } \
                        QSlider::add-page:horizontal { background: #fff; border: 1px solid #777; height: 10px; border-radius: 4px;} \
                        QSlider::handle:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc); \
                         border: 1px solid #777; width: 13px; margin-top: -2px; margin-bottom: -2px; border-radius: 4px; } \
                        QSlider::handle:horizontal:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fff, stop:1 #ddd); \
                         border: 1px solid #444; border-radius: 4px; } \
                        QSlider::sub-page:horizontal:disabled { background: #bbb; border-color: #999; } \
                        QSlider::add-page:horizontal:disabled { background: #eee; border-color: #999; } \
                        QSlider::handle:horizontal:disabled { background: #eee; border: 1px solid #aaa; border-radius: 4px; }"
    VSpinBox = "QSpinBox {background-color: rgb(245,245,245); font: 24px 'Times';}"

    LabelText       = "QLabel { background-color:transparent; color : gray; font: 16px 'Times'; }"
    HintText        = "QLabel { background-color:transparent; color : green; font: 16px 'Times'; }"
    WrongText       = "QLabel { background-color:transparent; color : red; font: 16px 'Times'; }"
    ContentText     = "QLabel { background-color:transparent; color : black; font: 24px 'Times'; }"
    HitoryTitleText = "QLabel { background-color:transparent; color : black; font: 42px 'Times'; border-style: solid; border-width: 2px; border-radius: 3px;}"
    SubtitleText    = "QLabel { background-color:transparent; color : black; font: 32px 'Times'; }"
    TitleText       = "QLabel { background-color:transparent; color : black; font: 60px 'Times'; }"
    WarningText     = "QLabel { background-color:transparent; color : red; font: bold 100px 'Times'; }"
    WelcomeText     = "QLabel { background-color:transparent; color : #ea9999; font: bold 130px 'Times'; }"

    MainPage = "background-color: #DDD9C3"

class UIFont:
    Combobox = QFont('Times', 24)
    SpinBox = QFont('Times', 24)

class UIThreadState(Enum):
    STOP = auto()
    READY = auto()
    PAUSE = auto()
    RUNNING = auto()

class UIState(Enum):
    STOP = auto()
    READY = auto()
    PLAY = auto()


def createHomeButton():
    btn_home = QPushButton()
    icon=f"{ICONDIR}/home.png"
    btn_home.setIcon(QIcon(icon))
    btn_home.setFixedSize(QSize(80, 80))
    btn_home.setIconSize(QSize(50,50))
    btn_home.setStyleSheet("QPushButton { border-radius : 40; border : 2px solid black; background-color: rgb(40,148,255)} \
                            QPushButton::hover { background-color: rgb(60,168,255); color: black; }")
    return btn_home

def createCopyright():
    copyright = QLabel()
    copyright.setFixedHeight(40)
    copyright.setText("版權所有 © 2022 運動特區科技有限公司")
    copyright.setStyleSheet("QLabel { background-color:transparent; color : black; font: italic 16px 'Times'; }")
    return copyright