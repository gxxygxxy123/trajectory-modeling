import threading
import cv2
import logging
import configparser
import copy
import queue
import base64
import paho.mqtt.client as mqtt
import numpy
import json
import pickle
import time
import sys
import os
import shutil
import numpy as np
import argparse

import gi
gi.require_version("Tcam", "0.1")
gi.require_version("Gst", "1.0")

from gi.repository import GLib, GObject, Gst, Tcam
from datetime import datetime
from typing import Optional

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from common import loadNodeConfig, loadConfig
from inspector import sendPerformance, sendNodeStateMsg
from point import Point
from writer import CSVWriter

def try_put_frame(q: queue.Queue, fid, image, timestamp):
    try:
        q.put_nowait((fid, image, timestamp))
    except queue.Full:
        pass

def convert_sample_to_numpy(sample):
    buf = sample.get_buffer()
    # timestamp = datetime.now().timestamp()
    timestamp = buf.pts / 1000000000
    caps = sample.get_caps()
    mem = buf.get_all_memory()
    success, info = mem.map(Gst.MapFlags.READ)
    if success:
        data = info.data
        mem.unmap(info)
        bpp = 4
        dtype = numpy.uint8
        if( caps.get_structure(0).get_value('format') == "BGRx" ):
            bpp = 4

        if(caps.get_structure(0).get_value('format') == "GRAY8" ):
            bpp = 1

        if(caps.get_structure(0).get_value('format') == "GRAY16_LE" ):
            bpp = 1
            dtype = numpy.uint16

        img_mat = numpy.ndarray(
            (caps.get_structure(0).get_value('height'),
            caps.get_structure(0).get_value('width'),
            bpp),
            buffer=data,
            dtype=dtype)
    return timestamp, img_mat

def gst_to_opencv(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()

    arr = numpy.ndarray(
        (caps.get_structure(0).get_value('height'),
         caps.get_structure(0).get_value('width'),
         3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=numpy.uint8)
    return arr

class RawImgPublisher(threading.Thread):
    def __init__(self, broker, output_topic, queue):
        threading.Thread.__init__(self)

        self.frame_queue = queue
        self.last_send_t = 0
        self.interval = 0

        # feature: screenshot
        self.oneShot = False
        self.oneShotPath = f"{ROOTDIR}/replay/screenshot.jpg"

        # setup MQTT client
        client = mqtt.Client()
        # client.on_publish = self.on_publish
        client.connect(broker)

        self.client = client
        self.output_topic = output_topic
        logging.info(f"output topic: {output_topic}")

    def on_publish(self, mosq, userdata, mid):
        logging.debug("send")

    def stop(self):
        logging.info("stop")
        self.alive = False

    def screenshot(self, saved_path):
        self.oneShotPath = saved_path
        self.oneShot = True

    def setFPS(self, fps):
        if self.alive is False:
            if fps <= 0:
                self.interval = 0
            else:
                self.interval = 1/fps
            logging.info(f"Raw Image Publisher FPS: {fps}")

    def setResize(self, flag, width=720, height=540):
        self.isResize = flag
        self.width = width
        self.height = height

    def run(self):
        self.alive = True
        logging.info("Raw Image Publisher is ready.")
        timestamp_base = 0
        while self.alive:
            if self.frame_queue.qsize() > 0:
                fid, image, timestamp = self.frame_queue.get_nowait()
                # screen shot origin picture
                if self.oneShot:
                    cv2.imwrite(self.oneShotPath, image)
                    self.oneShot = False
                if self.isResize:
                    image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                if (timestamp - self.last_send_t) > self.interval:
                    # publish raw image
                    ret, buf = cv2.imencode('.jpg', image)
                    if ret == True:
                        imdata = pickle.dumps(buf)
                        payload = { 'id': fid, 'timestamp': timestamp, 'raw_data': base64.b64encode(imdata).decode('ascii')}
                        self.client.publish(self.output_topic, json.dumps(payload))
                    self.last_send_t = timestamp
            else:
                time.sleep(0.01)
        logging.info("Raw Image Publisher terminated.")

class CameraReader(threading.Thread):
    def __init__(self, info, frame_queue: queue.Queue, nodename, mqtt_client):
        threading.Thread.__init__(self)
        self.killswitch = threading.Event()

        # camera fps, width, height
        self.fps = float(info['fps']) # TODO update fps in UI
        self.width = int(info['width'])
        self.height = int(info['height'])

        # frames
        self.frame_queue = frame_queue
        self.isRecording = False
        self.isStreaming = False
        self.isStop = False
        self.client = mqtt_client
        self.nodename = nodename
        self.timestamp_base = 0
        # initialize
        self.initial(info)
        # open device
        self.isOpen = False
        self.openDevice()
        if self.isOpen:
            logging.info(f"Open device: {self.serialnumber}")
            self.frame_count = 0
            # load configs
            self.loadCameraConfigs()
            # image processing
            # self.kernel = np.ones((5, 5), np.uint8)
        else:
            logging.warning(f"Can't find out the device: {self.serialnumber}")
        # Recorder
        self.recorder = None

    def initial(self, info):
        # setup
        Gst.init([])
        self.camera = None
        # setup camera
        self.serialnumber = info['hw_id'] + "-v4l2"
        # load camera config
        cfg_file = f"{DIRNAME}/location/{info['place']}/{info['hw_id']}.cfg"
        self.cfg = loadConfig(cfg_file)

    def openDevice(self):
        try:
            self.camera = Gst.parse_launch(
            "tcambin name=source"
            " ! capsfilter name=caps"
            #" ! tee name=t"
            " ! queue"
            #" ! videoconvert"
            " ! appsink name=sink")
        except GLib.Error as error:
            logging.error(f"Error creating camera: {error}")
            raise
        # Quere the source module.
        source = Gst.ElementFactory.make("tcambin")
        serials = source.get_device_serials_backend()
        logging.debug(f"serials: {serials}")
        if self.serialnumber in serials:
            self.source = self.camera.get_by_name("source")
            self.source.set_property("serial", self.serialnumber)
            self.isOpen = True
        else:
            return False
        # Query a pointer to the appsink, so we can assign the callback function.
        appsink = self.camera.get_by_name("sink")
        appsink.set_property("max-buffers",10)
        appsink.set_property("drop",0)
        # tell appsink to notify us when it receives an image
        appsink.set_property("emit-signals", True)
        appsink.connect('new-sample', self.onFrameReady)

        self.camera.set_state(Gst.State.READY)
        self.property_names = self.source.get_tcam_property_names()
        return True

    def closeDevice(self):
        if self.isOpen:
            self.camera.set_state(Gst.State.PAUSED)
            self.camera.set_state(Gst.State.NULL)
            self.camera = None
            self.isOpen = False

    def loadCameraConfigs(self): # TODO change to load_camera_config ?
        for name, value in self.cfg.items('Camera'):
            self.setProperties(name, value)

    def onFrameReady(self, appsink):
        try:
            #sample = appsink.get_property('last-sample')
            sample = appsink.emit("pull-sample")
            timestamp_frame, image = convert_sample_to_numpy(sample)
            self.frame_count += 1
            timestamp = self.timestamp_base + timestamp_frame
            if self.isRecording:
                buf = sample.get_buffer()
                self.recorder_src.emit("push-buffer", buf)
                # logging.debug(f"fid: {self.frame_count}, t: {timestamp_frame}, timestamp: {timestamp}")
                self.csvWriter.writePoints(Point(fid=self.frame_count,timestamp=timestamp))
            try_put_frame(self.frame_queue, self.frame_count, image, timestamp)
        except GLib.Error as error:
            logging.error(f"Error onFrameReady : {error}")
            raise
        return Gst.FlowReturn.OK

    def setProperties(self, name, new_value):
        if not self.isOpen:
            return
        # To modify the name to property of camera
        # 'Exposure'
        if name == 'ExposureAuto':
            name = "Exposure Auto"
        elif name == 'ExposureTimeAbs':
            name = "Exposure Time (us)"
            new_value = f"{new_value}000"
        # 'Gain'
        elif name == 'GainAuto':
            name = "Gain Auto"
        # 'White Balance'
        elif name == 'BalanceWhiteAuto':
            name = "Whitebalance Auto"
        elif name == 'BalanceRatioRed':
            name = "Whitebalance Red"
        elif name == 'BalanceRatioBlue':
            name = "Whitebalance Blue"

        if name not in self.property_names:
            logging.warning(f"name: {name}")
            return

        (ret, value,
         min_value, max_value,
         default_value, step_size,
         value_type, flags,
         category, group) = self.source.get_tcam_property(name)

        if not ret:
            logging.warning(f"could not receive value {name}")
            raise

        try:
            if value_type == "enum" :
                if new_value:
                    new_value = "On"
                else:
                    new_value = "Off"
                self.source.set_tcam_property(name, new_value)
            elif value_type == "boolean":
                if new_value == "On":
                    self.source.set_tcam_property(name, True)
                elif new_value == "Off":
                    self.source.set_tcam_property(name, False)
            elif value_type == "integer":
                new_value = int(new_value)
                if new_value > max_value:
                    new_value = max_value
                elif new_value < min_value:
                    new_value = min_value
                elif new_value == value:
                    return
                self.source.set_tcam_property(name, new_value)
        except Exception as e:
            logging.error(e)

    def setFramerate(self, fps): # TODO TODO TODO UI change all cameras to the same
        self.fps = fps
        # reset Camera
        if self.isOpen:
            self.stopStreaming()
            time.sleep(0.5)
            self.startStreaming()

    def setCaps(self):
        caps = Gst.Caps.new_empty()
        format = f"video/x-raw,format=BGRx,width={self.width}," \
                    f"height={self.height}," \
                    f"framerate={int(self.fps)}/1"
        structure = Gst.Structure.new_from_string(format)
        caps.append_structure(structure)
        structure.free()
        self.caps = caps
        capsfilter = self.camera.get_by_name("caps")
        capsfilter.set_property("caps", caps)

    def startStreaming(self):
        if self.isOpen:
            logging.info(f"Camera {self.serialnumber} start to stream")
            try:
                self.setCaps()
                self.camera.set_state(Gst.State.PLAYING)
                self.timestamp_base = datetime.now().timestamp()
                error = self.camera.get_state(5000000000)
                if error[1] != Gst.State.PLAYING:
                    logging.error(f"Error starting camera {self.serialnumber}")
                    return False
                logging.info(f"{self.serialnumber} Start Streaming... t: {self.timestamp_base}")
                self.isStreaming = True
            except: # GError as error:
                logging.error(f"Error starting camera: {self.serialnumber}")
                raise
        else:
            logging.warning(f"Camera device {self.serialnumber} don't open")

    def stopStreaming(self):
        if self.isOpen:
            logging.info(f"Camera {self.serialnumber} stop ...")
            self.camera.set_state(Gst.State.PAUSED)
            self.camera.set_state(Gst.State.READY)
            self.isStreaming = False
        else:
            logging.warning(f"Camera device {self.serialnumber} don't open")

    def _bus_call(self, gst_bus, message):
        t = message.type
        logging.debug(f"Received msg from {message.src.get_name()}")
        if message.src.get_name() == self.recorder_sink:
            self.file_location = ""

        if t == Gst.MessageType.EOS:
            logging.debug(f"Received EOS from {message.src.get_name()}")
            if (message.src.get_name() == self.recorder_sink and
                    self.media_type == MediaType.video):
                logging.debug(f"sink sent EOS {message.get_structure().to_string()}")
                self.file_location = ""
                self.recorder.set_state(Gst.State.PAUSED)
                self.recorder.set_state(Gst.State.READY)
                self.recorder.set_state(Gst.State.NULL)
                self.recorder_src = None
                self.recorder = None

    def publishRecordingStopped(self):
        payload = {self.nodename: "RecordingStopped"}
        self.client.publish('system_status', json.dumps(payload))

    def openDeviceWithRecord(self, file_path, file_name):
        self.recorder_name = "recorder-src"
        self.recorder_sink = "fsink"
        try:
            if self.recorder is not None:
                (ret, state, pending) = self.recorder.get_state(0)
                logging.debug(f"{ret}, {state}, {pending}")
                if state == Gst.State.PLAYING:
                    self.recorder.set_state(Gst.State.PAUSED)
                    self.recorder.set_state(Gst.State.READY)
                    self.recorder.set_state(Gst.State.NULL)
                    self.recorder_src = None
                    self.recorder = None
            self.file_location = f"{file_path}/{file_name}.avi"
            self.csvWriter = CSVWriter(name=self.nodename, filename=f"{file_path}/{file_name}.csv")
            record_str = (
                "appsrc name={} is-live=true format=3 "
                "! {} "
                #"! queue leaky=downstream "
                "! queue "
                "! videoconvert "
                "! x264enc ! mp4mux  "
                #"! avimux "
                "! filesink name=fsink location={}").format(
                                                self.recorder_name,
                                                self.caps.to_string(),
                                                self.file_location)
            self.recorder = Gst.parse_launch(record_str)
            logging.info(f"Recording file to {self.file_location}")
        except GLib.Error as error:
            logging.error(f"Error creating camera: {error}")
            raise

        self.recorder_src = self.recorder.get_by_name(self.recorder_name)
        self.recorder_src.set_property("caps", self.caps)
        self.recorder_src.set_property("do-timestamp", True)

        self.recorder.set_name(self.recorder_name)

    def startRecord(self, file_path, file_name):
        # open device with record
        if self.isRecording or not self.isStreaming:
            return
        logging.info(f"Camera {self.serialnumber} start Recording...")
        self.openDeviceWithRecord(file_path, file_name)
        self.recorder.set_state(Gst.State.PLAYING)
        self.isRecording = True

        bus = self.recorder.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._bus_call)

    def stopRecord(self):
        self.isRecording = False
        logging.info(f"Camera {self.serialnumber} stop Recording...")
        self.killswitch.set()

    def stop(self):
        self.isStop = True
        self.killswitch.set()

    def run(self):
        logging.debug(f"Camera Reader {self.serialnumber} started.")
        try:
            logging.info(f"Camera Reader {self.serialnumber} is ready.")
            while True:
                self.killswitch.wait()
                if self.isStop:
                    break
                elif self.isRecording == False:
                    # wait For write done
                    if self.recorder is not None:
                        self.recorder.send_event(Gst.Event.new_eos())
                        time.sleep(2)
                        self.csvWriter.close()
                    self.publishRecordingStopped()
                    self.killswitch.clear()
        except: # GError as error:
            logging.error(f"Error starting camera {self.serialnumber}.")
            raise
        finally:
            if self.recorder is not None:
                (ret, state, pending) = self.recorder.get_state(0)
                logging.debug(f"{ret}, {state}, {pending}")
                if state == Gst.State.PLAYING:
                    self.recorder.set_state(Gst.State.PAUSED)
                    self.recorder.set_state(Gst.State.READY)
                    self.recorder.set_state(Gst.State.NULL)
                    self.recorder_src = None
                    self.recorder = None
            self.stopStreaming()
            self.closeDevice()
        logging.info(f"Camera Reader {self.serialnumber} terminated.")

class MainThread(threading.Thread):
    def __init__(self, args, info):
        threading.Thread.__init__(self)
        self.killswitch = threading.Event()

        # setup configuration
        self.info = info
        self.project_name = args.project
        self.nodename = args.nodename

        # setup buffer queue
        queue_size = int(self.info['queue_size'])
        broker = self.info['mqtt_broker']
        self.frame_queue = queue.Queue(maxsize=queue_size)

        # setup MQTT client
        broker = self.info['mqtt_broker']
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(broker)
        self.client = client
        self.general_topic = info['general_topic']

        # setup Camera Reader
        self.cameraReader = CameraReader(info, self.frame_queue, self.nodename, self.client)

        # setup Raw Image Publisher
        output_topic = self.info['output_topic']
        monitor_topic = self.info['monitor_topic']
        self.rawImgPublisher = RawImgPublisher(broker, output_topic, self.frame_queue)

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"{self.nodename} Connected with result code: {rc}")
        self.client.subscribe(self.nodename)
        self.client.subscribe(self.general_topic)

    def on_message(self, client, userdata, msg):
        cmds = json.loads(msg.payload)

        if 'stop' in cmds:
            if cmds['stop'] == True:
                self.stop()
        elif 'stopStreaming' in cmds:
            if cmds['stopStreaming'] == True:
                self.stopStreaming()
        elif 'startStreaming' in cmds:
            if cmds['startStreaming'] == True:
                self.startStreaming()

        # backup Video
        if 'startRecording' in cmds:
            if cmds['startRecording'] == True:
                if 'video_path' in cmds:
                    video_path = cmds['video_path']
                    # copy project config
                    cfg_file = f"{ROOTDIR}/projects/{self.project_name}.cfg"
                    destination = f"{video_path}/{self.project_name}.cfg"
                    shutil.copyfile(cfg_file, destination)
                    # copy device config
                    cfg_file = f"{DIRNAME}/location/{self.info['place']}/{self.info['hw_id']}.cfg"
                    destination = f"{video_path}/{self.info['hw_id']}.cfg"
                    shutil.copyfile(cfg_file, destination)
                    # start to record
                    self.cameraReader.startRecord(video_path, self.nodename)
                    self.rawImgPublisher.setResize(True, 512, 288)
                else:
                    logging.warning("please set video path in cmds('video_path')")
        elif 'stopRecording' in cmds:
            if cmds['stopRecording'] == True:
                self.cameraReader.stopRecord()
                self.rawImgPublisher.setResize(True)

        # feature: screenshot
        if 'Screenshot' in cmds:
            self.screenshot(cmds['Screenshot'])

        # change camera setting
        if 'ExposureAuto' in cmds:
            self.cameraReader.setProperties('ExposureAuto', cmds['ExposureAuto'])
        if 'Gain' in cmds:
            self.cameraReader.setProperties('Gain', cmds['Gain'])
        if 'GainAuto' in cmds:
            self.cameraReader.setProperties('GainAuto', cmds['GainAuto'])
        if 'Brightness' in cmds:
            self.cameraReader.setProperties('Brightness', cmds['Brightness'])
        if 'BalanceWhiteAuto' in cmds:
            self.cameraReader.setProperties('BalanceWhiteAuto', cmds['BalanceWhiteAuto'])
        if 'ExposureTimeAbs' in cmds:
            self.cameraReader.setProperties('ExposureTimeAbs', cmds['ExposureTimeAbs'])
        if 'BalanceRatioRed' in cmds:
            self.cameraReader.setProperties('BalanceRatioRed', cmds['BalanceRatioRed'])
        if 'BalanceRatioBlue' in cmds:
            self.cameraReader.setProperties('BalanceRatioBlue', cmds['BalanceRatioBlue'])
        if 'Framerate' in cmds: # TODO change all cameras to the same
            self.cameraReader.setFramerate(cmds['Framerate'])

    def screenshot(self, save_path):
        logging.info(f"To save screen in path {save_path}")
        self.rawImgPublisher.screenshot(save_path)

    def stopStreaming(self):
        self.cameraReader.stopStreaming()

    def startStreaming(self):
        self.cameraReader.startStreaming()
        self.rawImgPublisher.setResize(True)

    def setupCameraReader(self):
        self.cameraReader = CameraReader(self.info)

    def stop(self):
        self.killswitch.set()

    def run(self):
        logging.debug("{} started.".format(self.nodename))

        try:
            self.cameraReader.start()
            self.rawImgPublisher.start()
            self.client.loop_start()
            logging.info("{} is ready.".format(self.nodename))
            sendNodeStateMsg(self.client, self.nodename, "ready")
            self.killswitch.wait()
        finally:
            self.rawImgPublisher.stop()
            self.rawImgPublisher.join()
            if self.cameraReader.is_alive():
                self.cameraReader.stop()
                self.cameraReader.join()
            sendNodeStateMsg(self.client, self.nodename, "terminated")
            self.client.loop_stop()
        logging.info("{} terminated.".format(self.nodename))

def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = 'CameraReader')
    parser.add_argument('--project', type=str, default = 'coachbox', help = 'project name (default: coachbox)')
    parser.add_argument('--nodename', type=str, default = 'CameraReader', help = 'mqtt node name (default: CameraReader)')
    args = parser.parse_args()

    return args

def main():
    # Parse arguments
    args = parse_args()
    # Load configs
    projectCfg = f"{ROOTDIR}/projects/{args.project}.cfg"
    settings = loadNodeConfig(projectCfg, args.nodename)

    # Start MainThread
    mainThread = MainThread(args, settings)
    mainThread.start()
    mainThread.join()

if __name__ == '__main__':
    main()
