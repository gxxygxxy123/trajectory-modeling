from enum import Enum, auto

class Account():
    def __init__(self, account="Frank", password="", email=""):
        self.account = account
        self.password = password
        self.email = email

class MsgContract():
    class ID(Enum):
        STOP = auto()
        # System Control
        SYSTEM_CLOSE = auto()
        # Account Manager
        LOGIN = auto()
        LOGOUT = auto()
        REGISTER = auto()
        UPDATE_PASSWORD = auto()
        IMPORT_ACCOUNT = auto()
        GET_ACCOUNTS = auto()
        DELETE_ACCOUNT = auto()
        ADD_STUDENTS = auto()
        GET_STUDENTS = auto()
        # History Table
        ADD_HISTORY = auto()
        GET_HISTORY = auto()
        # Camera Control
        CAMERA_START = auto()
        CAMERA_STOP = auto()
        CAMERA_RESTART = auto()
        CAMERA_STREAM = auto()
        CAMERA_RECORD = auto()
        CAMERA_READY_NUM = auto()
        CAMERA_SCREENSHOT = auto()
        CAMERA_GAIN = auto()
        CALCULATE_CAMERA_EXTRINSIC = auto()
        # Preview
        CAMERA_PREVIEW = auto()
        CAMERA_SETTING = auto()
        CAMERA_INTRINSIC = auto()
        EXPOSURE_VALUE = auto()
        SAVE_CAMERA_CONFIG = auto()
        # Training / Evaluate
        QUERY_PLAYER_NAME = auto()
        TRAINING_START = auto()
         # Analyst
        TRACKING_START = auto()
        TRACKING_STOP = auto()
        IS_ALL_READY = auto()
        # Serve Machine
        SERVE_MACHINE_START = auto()
        SERVE_MACHINE_REQUEST = auto()
        SERVE_MACHINE_STOP = auto()
        # Tracknet
        TRACKNET_DONE = auto()
        # Model3D
        MODEL3D_DONE = auto()
        # Action Analyzer
        ANALYZE_DONE = auto()
        # RNN Predictor
        RNN_DONE = auto()
        # MQTT SUBSCRIBE MESSAGE
        ON_MESSAGE = auto()
        # Court settings
        START_COURT_SETTINGS = auto()
        SET_REPLAY_DIR = auto()
        # UI updates state
        UI_PROGRESS = auto()
        UI_DONE = auto()
        PAGE_CHANGE = auto()
        PAGE_FINISH = auto()

    def __init__(self, id, arg:bool=False, value=None, reply=None, request=None):
        self.id = id
        self.arg = arg
        self.value = value
        self.reply = reply
        self.request = request

class MachineContract():
    class ID(Enum):
        SERVER = 19
    def __init__(self, id, times=1, delay=100):
        self.id = id
        self.times = times
        self.delay = 100

class MqttContract():
    class ID(Enum):
        STOP = auto()
        PUBLISH = auto()
        SUBSCRIBE = auto()

    def __init__(self, id, topic, payload):
        self.id = id
        self.topic = topic
        self.payload = payload

class ServeContract():
    class ID(Enum):
        START = auto()
        REQUEST = auto()
        REPLY = auto()
        STOP = auto()

    class BUTTON(int):
        TURN_DOWN = 1
        TURN_RIGHT = 3
        TURN_UP = 4
        POWER = 5
        TURN_LEFT = 7
        SPEED_UP = 9
        SPEED_DOWN = 10
        FREQ_DOWN = 11
        START_SERVE = 12
        FREQ_UP = 13
        PAUSE_SERVE = 19

    def __init__(self, id, button=None, times=1, interval=None):
        self.id = id
        self.button = button
        self.times = times
        self.interval = interval
