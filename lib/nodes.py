'''
functions: node related functions
'''
import os
import logging

from PyQt5.QtCore import QProcess

from enum import Enum, auto

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)

# setup Camera
def setupCameras(cfg):
    cameras = []
    for node_name, node_info in cfg.items():
        # Get All Cameras
        if 'node_type' in node_info:
            if node_info['node_type'] == 'Reader':
                camera = CameraReader(node_name, node_info, cfg['Project']['place'])
                cameras.append(camera)
    return cameras

# setup Offline track 3d trajectory
def setupOfflineTrackingNodes(project_name, cfg, replay_path):
    nodes = []
    camera_cfgs = []
    place = cfg['Project']['place']
    for node_name, node_info in cfg.items():
        # Get All Cameras
        if 'node_type' in node_info:
            if node_info['node_type'] == 'Reader':
                camera_cfg = f"{ROOTDIR}/Reader/{node_info['brand']}/location/{place}/{node_info['hw_id']}.cfg"
                camera_cfgs.append(camera_cfg)
    for node_name, node_info in cfg.items():
        if 'node_type' in node_info:
            # Get File Reader L/R
            #if node_info['node_type'] == 'FileReader':
            #    reader = FileReader(project_name, node_name, node_info, replay_path)
            #    nodes.append(reader)
            if node_info['node_type'] == 'TrackNet':
                tracknet = TrackNet(project_name, node_name, node_info, replay_path)
                nodes.append(tracknet)
            #elif node_info['node_type'] == 'Model3D':
            #    model3D = Model3D(project_name, node_name, node_info, replay_path, camera_cfgs)
            #    nodes.append(model3D)
    return nodes

def setupRNNPredictNodes(project_name, cfg):
    nodes = []
    for node_name, node_info in cfg.items():
        # Get All Cameras
        if 'node_type' in node_info:
            if node_info['node_type'] == 'RNN':
                camera = RnnPredictor(project_name, node_name, node_info)
                nodes.append(camera)
    return nodes

def setupAnalyzerNodes(cfg, replay_path):
    nodes = []
    video_path = replay_path
    camera_cfg = None
    place = cfg['Project']['place']
    fps = cfg['Project']['fps']
    for node_name, node_info in cfg.items():
        if node_name in video_path:
            camera_cfg = f"{ROOTDIR}/Reader/{node_info['brand']}/location/{place}/{node_info['hw_id']}.cfg"
            break

    video_path = os.path.abspath(video_path)
    camera_cfg = os.path.abspath(camera_cfg)
    output_folder = os.path.dirname(video_path)

    node = ActionAnalyzer("Analyzer", camera_cfg, video_path, output_folder, fps)
    nodes.append(node)

    return nodes

class Node():
    class State(Enum):
        NO_START = auto()
        READY = auto()
        TERMINATED = auto()

    def __init__(self):
        self.name = "None"
        self.command = ""
        self.process = None
        self.state = Node.State.NO_START

    def stop(self):
        if self.process is not None:
            if self.process.state() == QProcess.Running:
                self.process.kill()
                self.process.terminate()

    def start(self):
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        logging.debug(self.command)
        self.process.start("/bin/bash", ['-c', self.command])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        logging.debug(stdout)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        logging.error(stderr)

class CameraReader(Node):
    def __init__(self, node_name, node_info, place):
        super().__init__()
        self.name = node_name
        self.node_type = node_info['node_type']
        self.brand = node_info['brand']
        self.hw_id = node_info['hw_id']
        self.output_topic = node_info['output_topic']
        self.command = f"python3 {ROOTDIR}/Reader/{node_info['brand']}/main.py " \
                        f"--nodename {node_name}"
        self.place = place
        self.isStreaming = False
        self.gain = 135

class FileReader(Node):
    def __init__(self, project_name, node_name, node_info, load_path):
        super().__init__()
        self.name = node_name
        self.node_type = node_info['node_type']
        video_path = os.path.realpath(os.path.join(load_path, f"{node_info['file_name']}.avi"))
        csv_path = os.path.realpath(os.path.join(load_path, f"{node_info['file_name']}.csv"))
        self.command = f"python3 {ROOTDIR}/Reader/FileReader/FileReader.py " \
             f"--project {project_name} --nodename {node_name} --file {video_path} --csv {csv_path}"

class TrackNet(Node):
    def __init__(self, project_name, node_name, node_info, load_path):
        super().__init__()
        self.name = node_name
        self.node_type = node_info['node_type']
        video_path = os.path.realpath(os.path.join(load_path, f"{node_info['file_name']}.avi"))
        csv_path = os.path.realpath(os.path.join(load_path, f"{node_info['file_name']}.csv"))
        self.command = f"python3 {ROOTDIR}/TrackNet/TrackNet10/TrackNet.py " \
            f"--nodename {node_name} --data {video_path} --input_csv {csv_path} --save_csv {load_path}"

class Model3D(Node):
    def __init__(self, project_name, node_name, node_info, save_path, camera_cfgs):
        super().__init__()
        self.name = node_name
        self.node_type = node_info['node_type']
        self.command = f"python3 {ROOTDIR}/Model3D/Model3D.py " \
            f"--project {project_name} --nodename {node_name} --save_path {save_path} " \
            f"--camera_cfgs"
        for cam_cfg in camera_cfgs:
            self.command += f" {cam_cfg}"

class RnnPredictor(Node):
    def __init__(self, project_name, node_name, node_info):
        super().__init__()
        self.name = node_name
        self.node_type = node_info['node_type']
        self.command = f"python3 {ROOTDIR}/RNN/RnnPredictor.py " \
            f"--project {project_name} --nodename {node_name}"

class ActionAnalyzer(Node):
    def __init__(self, node_name, camera_cfg, video_path, output_folder, fps):
        super().__init__()
        self.name = node_name
        self.command = f"python3 main.py " \
            f"--camera_cfg {camera_cfg} --output_folder {output_folder} \
              --run {video_path} --fps {fps}"
