'''
    Application Background Service
'''
import json
import logging
import os
import queue
import sys
import threading
import time
import numpy as np

import cv2
import paho.mqtt.client as mqtt
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal, QRectF, QLineF

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
ICONDIR = f"{ROOTDIR}/UI/icon"
sys.path.append(f"{ROOTDIR}/lib")
from common import loadConfig, saveConfig
from h2pose.H2Pose import H2Pose
from h2pose.Hfinder2 import Hfinder2
from common import insertById
from frame import Frame
from message import *
from nodes import CameraReader, Node
from common import loadConfig, saveConfig
from ServeMachine import ServeMachineCommunicator
from DbAdapter import DbAdapter

# between UI and MQTTService
class SystemService(QThread):
    callback = pyqtSignal(MsgContract)
    def __init__(self, cfg):
        super().__init__()

        # Event Queue
        self.waitEvent = threading.Event()
        self.messageQueue = queue.Queue(maxsize=10)
        # MQTT Service
        self.mqttService = MQTTService(cfg['Project']['mqtt_broker'])
        self.mqttService.callback.connect(self.on_message)
        # Serve Machine Communicator
        self.serveMachineCommuniator = ServeMachineCommunicator(cfg)
        # nodes
        self.nodes = {}

    def stop(self):
        self.sendMessage(MsgContract(id=MsgContract.ID.STOP))
        for node in self.nodes.values():
            node.stop()

    def on_message(self, payload):
        self.sendMessage(MsgContract(MsgContract.ID.ON_MESSAGE, value=payload))

    def sendMessage(self, msg):
        self.messageQueue.put_nowait(msg)
        self.waitEvent.set()

    def handleMessage(self, msg:MsgContract):
        if msg.id == MsgContract.ID.SYSTEM_CLOSE:
            payload = {"stop": True}
            self.mqttService.sendMessage(MqttContract.ID.PUBLISH, 'system_control', payload)
        elif msg.id == MsgContract.ID.ON_MESSAGE:
            data = json.loads(msg.value)
            for node in self.nodes.values():
                if node.name in data:
                    if data[node.name] == 'ready':
                        node.state = Node.State.READY
                    elif data[node.name] == 'terminated':
                        node.state = Node.State.TERMINATED
                    else:
                        logging.error(f"Not supported node state. {node.name}: {data[node.name]}")
            if 'TrackNetL' in data: # check if both TrackNet Node are finished
                if data['TrackNetL'] == 'terminated':
                    for node in self.nodes.values():
                        if node.name == 'TrackNetR' and node.state == Node.State.TERMINATED:
                            self.delNode("TrackNetL")
                            self.delNode("TrackNetR")
                            msg_emit = MsgContract(id = MsgContract.ID.TRACKNET_DONE)
                            self.callback.emit(msg_emit)
                            #payload = {"stop": True}
                            #self.mqttService.sendMessage(MqttContract.ID.PUBLISH, 'Model3D', payload)
                            break
            elif 'TrackNetR' in data: # check if both TrackNet Node are finished
                if data['TrackNetR'] == 'terminated':
                    for node in self.nodes.values():
                        if node.name == 'TrackNetL' and node.state == Node.State.TERMINATED:
                            self.delNode("TrackNetL")
                            self.delNode("TrackNetR")
                            msg_emit = MsgContract(id = MsgContract.ID.TRACKNET_DONE)
                            self.callback.emit(msg_emit)
                            #payload = {"stop": True}
                            #self.mqttService.sendMessage(MqttContract.ID.PUBLISH, 'Model3D', payload)
                            break
            elif 'Model3D' in data:
                if data['Model3D'] == 'terminated':
                    self.delNode("Model3D")
                    msg_emit = MsgContract(id = MsgContract.ID.MODEL3D_DONE)
                    self.callback.emit(msg_emit)
            elif 'RnnPredictor' in data:
                if data['RnnPredictor'] == 'terminated':
                    self.delNode("RnnPredictor")
                    msg_emit = MsgContract(id = MsgContract.ID.RNN_DONE)
                    self.callback.emit(msg_emit)
            elif 'Analyzer' in data:
                if data['Analyzer'] == 'terminated':
                    self.delNode("Analyzer")
                    msg_emit = MsgContract(id = MsgContract.ID.ANALYZE_DONE)
                    self.callback.emit(msg_emit)
            elif 'progress' in data:
                msg_emit = MsgContract(id = MsgContract.ID.UI_PROGRESS)
                msg_emit.value = int(data['progress'])
                self.callback.emit(msg_emit)
        elif msg.id == MsgContract.ID.PAGE_CHANGE or msg.id == MsgContract.ID.PAGE_FINISH:
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.START_COURT_SETTINGS:
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.LOGIN:
            user:Account = msg.value
            self.coachname = user.account
            db = DbAdapter()
            msg.arg = db.loginCheck(user)
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.REGISTER or msg.id == MsgContract.ID.IMPORT_ACCOUNT:
            user:Account = msg.value
            db = DbAdapter()
            msg.arg = db.register(user)
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.LOGOUT:
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.UPDATE_PASSWORD:
            user:Account = msg.value
            db = DbAdapter()
            db.updatePassword(user)
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.GET_ACCOUNTS:
            db = DbAdapter()
            msg.value = db.getAccountsData()
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.DELETE_ACCOUNT:
            user:Account = msg.value
            db = DbAdapter()
            db.deleteAccount(user)
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.ADD_STUDENTS:
            students = msg.value
            db = DbAdapter()
            db.addStudents(self.coachname, students)
        elif msg.id == MsgContract.ID.GET_STUDENTS:
            db = DbAdapter()
            students = db.getStudents(self.coachname)
            msg.value = students
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.ADD_HISTORY:
            db = DbAdapter()
            history = msg.value
            db.addHistory(history['datetime'], self.coachname, history['student'], history['balltype'], history['ballnum'])
        elif msg.id == MsgContract.ID.GET_HISTORY:
            db = DbAdapter()
            history = db.getHistory(self.coachname)
            msg.value = history
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.SET_REPLAY_DIR:
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.CAMERA_START:
            for node in self.nodes.values():
                if node.node_type == 'Reader':
                    node.start()
        elif msg.id == MsgContract.ID.CAMERA_STOP:
            for node in self.nodes.values():
                if node.name == 'Reader':
                    node.stop()
                    del self.nodes[node.name]
        elif msg.id == MsgContract.ID.CAMERA_STREAM:
            if msg.value == None:
                topic = "cam_control"
                payload = {"streaming": msg.arg}
                for node in self.nodes.values():
                    if node.name == 'Reader':
                        node.isStreaming = False
                self.mqttService.sendMessage(MqttContract.ID.PUBLISH, topic, payload)
            else:
                camera:CameraReader = msg.value
                if camera.name in self.nodes:
                    topic = camera.name
                    payload = {"streaming": camera.isStreaming}
                    self.mqttService.sendMessage(MqttContract.ID.PUBLISH, topic, payload)
        elif msg.id == MsgContract.ID.CAMERA_GAIN:
            if msg.value != None:
                camera:CameraReader = msg.value
                topic = camera.name
                payload = {"Gain": camera.gain}
                self.mqttService.sendMessage(MqttContract.ID.PUBLISH, topic, payload)
        elif msg.id == MsgContract.ID.CAMERA_RESTART:
            if msg.value == None:
                topic = "cam_control"
            else:
                node = msg.value
                topic = node.name
            payload = {"streaming": False}
            self.mqttService.sendMessage(MqttContract.ID.PUBLISH, topic, payload)
            time.sleep(1)
            payload = {"streaming": True}
            self.mqttService.sendMessage(MqttContract.ID.PUBLISH, topic, payload)
        elif msg.id == MsgContract.ID.CAMERA_READY_NUM:
            num_cam_ready = 0
            for node in self.nodes.values():
                if "CameraReader" in node.name and node.state == Node.State.READY:
                    num_cam_ready +=1
            logging.debug(f"#CameraReader={num_cam_ready} are ready.")
            if msg.reply != None:
                msg.value = num_cam_ready
                msg.reply(msg)
        elif msg.id == MsgContract.ID.CAMERA_SCREENSHOT:
            topic = msg.value["topic"]
            path = msg.value["save_path"]
            payload = {"Screenshot": path}
            self.mqttService.sendMessage(MqttContract.ID.PUBLISH, topic, payload)
        elif msg.id == MsgContract.ID.CAMERA_RECORD:
            record = msg.arg
            if record == True:
                path = msg.value
                payload = {"recording": True, "path": path}
            else:
                payload = {"recording": False}
            self.mqttService.sendMessage(MqttContract.ID.PUBLISH, "cam_control", payload)
        elif msg.id == MsgContract.ID.QUERY_PLAYER_NAME:
            if msg.reply != None:
                msg.value = self.user_name
                msg.reply(msg)
        elif msg.id == MsgContract.ID.TRAINING_START or msg.id == MsgContract.ID.CAMERA_SETTING or \
            msg.id == MsgContract.ID.CAMERA_PREVIEW or msg.id == MsgContract.ID.CAMERA_INTRINSIC:
            self.callback.emit(msg)
        elif msg.id == MsgContract.ID.TRACKING_START:
            payload = {"streaming": True}
            self.mqttService.sendMessage(MqttContract.ID.PUBLISH, "cam_control", payload)
            # create wait Tracking finish Thread
        elif msg.id == MsgContract.ID.TRACKING_STOP:
            for node in self.nodes.values():
                node.stop()
            self.nodes.clear()
            msg.reply()
        elif msg.id == MsgContract.ID.IS_ALL_READY:
            count = 0
            for node in self.nodes.values():
                if node.state == Node.State.READY:
                    count+=1
            if count == len(self.nodes):
                msg.value = True
            else:
                msg.value = False
            msg.reply(msg)
        elif msg.id == MsgContract.ID.SERVE_MACHINE_START:
            self.serveMachineCommuniator.start()
        elif msg.id == MsgContract.ID.SERVE_MACHINE_REQUEST:
            self.serveMachineCommuniator.sendMessage(msg.request)
        elif msg.id == MsgContract.ID.SERVE_MACHINE_STOP:
            self.serveMachineCommuniator.stop()

        elif msg.id == MsgContract.ID.SAVE_CAMERA_CONFIG:
            for node in self.nodes.values():
                if node.name == msg.value["topic"]:
                    config_file = f"{ROOTDIR}/Reader/{node.brand}/location/{node.place}/{node.hw_id}.cfg"
            cameraCfg =  loadConfig(config_file)
            cameraCfg['Camera']['Gain'] = str(msg.value["value"])
            saveConfig(config_file, cameraCfg)
        elif msg.id == MsgContract.ID.CALCULATE_CAMERA_EXTRINSIC:
            for node in self.nodes.values():
                if node.name == msg.value["topic"]:
                    config_file = f"{ROOTDIR}/Reader/{node.brand}/location/{node.place}/{node.hw_id}.cfg"
            cameraCfg =  loadConfig(config_file)
            camera_ks = np.array(json.loads(cameraCfg['Other']['ks']))
            dist = np.array(json.loads(cameraCfg['Other']['dist']))
            nmtx = np.array(json.loads(cameraCfg['Other']['newcameramtx']))
            image = cv2.imread(msg.value["image_path"])
            court3D = msg.value["court3D"]
            court2D = msg.value["court2D"]
            hf = Hfinder2(camera_ks=camera_ks, dist=dist, nmtx=nmtx, court2D=court2D, court3D=court3D)
            Hmtx = hf.getH()
            Kmtx = nmtx
            projection_mat = hf.getProjection_mat()
            extrinsic_mat = hf.getExtrinsic_mat()
            h2p = H2Pose(Kmtx, Hmtx)
            poses = h2p.getC2W()
            eye = h2p.getCamera().T
            eye[0][2] = abs(eye[0][2])

            # [TODO] Check if keys not in [Other] Section

            cameraCfg['Other']['poses'] = str(poses.tolist())
            cameraCfg['Other']['eye'] = str(eye.tolist())
            cameraCfg['Other']['hmtx'] = str(Hmtx.tolist())
            cameraCfg['Other']['projection_mat'] = str(projection_mat.tolist())
            cameraCfg['Other']['extrinsic_mat'] = str(extrinsic_mat.tolist())

            # msg = ID.SAVE_DEVICE_CONFIG
            logging.info('Poses:\n{}\n'.format(poses))
            logging.info('Eye:\n{}\n'.format(eye))
            logging.info('Hmtx:\n{}\n'.format(Hmtx))
            logging.info('projection_mat:\n{}\n'.format(projection_mat))
            logging.info('extrinsic_mat:\n{}\n'.format(extrinsic_mat))
            saveConfig(config_file, cameraCfg)
        else:
            logging.warning(f"{msg.id} is no supported.")

    def addNodes(self, new_nodes):
        if not self.isRunning():
            logging.debug("SystemService is not start up")
            return

        for new_node in new_nodes:
            for node in self.nodes.values():
                if node.name == new_node.name:
                    node.stop()
                    del self.nodes[node.name]
            new_node.start()
            self.nodes[new_node.name] = new_node

    def delNode(self, node_name):
        self.nodes[node_name].stop()
        del self.nodes[node_name]

    def run(self):
        try:
            logging.info(f"{self.__class__.__name__}: start up...")
            self.mqttService.start()
            while True:
                if self.messageQueue.empty():
                    self.waitEvent.wait()
                    self.waitEvent.clear()
                try:
                    msg = self.messageQueue.get_nowait()
                    if msg.id == MsgContract.ID.STOP:
                        self.mqttService.stop()
                        break
                    else:
                        self.handleMessage(msg)
                except queue.Empty:
                    logging.warn(f"{self.__class__.__name__}: the message queue is empty.")
        finally:
            if self.mqttService.isRunning() :
                logging.error("error: MQTTService is running")
            logging.info(f"{self.__class__.__name__}: shutdown...")

# Used for communication between UI Services (App.) and system (coachAI)
class MQTTService(QThread):
    callback = pyqtSignal(bytes)
    def __init__(self, broker_ip = 'localhost'):
        super().__init__()

        # Event Queue
        self.waitEvent = threading.Event()
        self.messageQueue = queue.Queue(maxsize=10)

        # Setup MQTT
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker_ip)

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"MQTTService: Application Connected with result code: {rc}")
        self.client.subscribe('system_status')

    def on_message(self, client, userdata, msg):
        self.sendMessage(id=MqttContract.ID.SUBSCRIBE, topic=msg.topic, payload=msg.payload)

    def stop(self):
        self.sendMessage(id=MqttContract.ID.STOP)

    def sendMessage(self, id, topic=None, payload=None):
        msg = MqttContract(id, topic, payload)
        self.messageQueue.put_nowait(msg)
        self.waitEvent.set()

    def run(self):
        try:
            self.client.loop_start()
            while True:
                if self.messageQueue.empty():
                    self.waitEvent.wait()
                    self.waitEvent.clear()
                else:
                    msg = self.messageQueue.get_nowait()
                    if msg.id == MqttContract.ID.STOP:
                        break
                    elif msg.id == MqttContract.ID.PUBLISH:
                        logging.debug(f"topic: {msg.topic}, payload: {msg.payload}")
                        self.client.publish(msg.topic, json.dumps(msg.payload))
                    elif msg.id == MqttContract.ID.SUBSCRIBE:
                        self.callback.emit(msg.payload)
        finally:
            self.client.loop_stop()
            logging.info("MQTTService: Application disconnected...")

