import sys
import os
import logging
import configparser
import copy
import cv2
import numpy as np
import threading
import queue
import base64
import paho.mqtt.client as mqtt
import time
import json
import pickle
import argparse
import shutil

from datetime import datetime
from typing import Optional
from vimba import *

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from common import load_config, load_camera_config
from inspector import sendPerformance, sendNodeStateMsg
from point import Point
from writer import CSVWriter

def get_camera(camera_id: Optional[str]) -> Camera:
    with Vimba.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)
            except VimbaCameraError:
                logging.error(f"Failed to access Camera {camera_id}. Abort.")
                sys.exit(1)

class Recorder(threading.Thread):
    def __init__(self, nodename, video_path, fps, width, height):
        threading.Thread.__init__(self)

        self.fileName = f"{video_path}/{nodename}.avi"
        self.csvWriter = CSVWriter(name=nodename, filename=f"{video_path}/{nodename}.csv")
        self.writer = cv2.VideoWriter(self.fileName, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        self.waitEvent = threading.Event()
        self.isRecording = False
        self.queue = []

    def try_put_frame(self, cv_image):
        self.queue.append(cv_image)
        self.waitEvent.set()

    def try_write_timestamp(self, fid, timestamp):
        self.csvWriter.writePoints(Point(fid=fid,timestamp=timestamp))

    def stop(self):
        self.isRecording = False
        self.waitEvent.set()

    def run(self):
        self.isRecording = True
        while self.isRecording or len(self.queue) > 0:
            if len(self.queue) > 0:
                cv_image = self.queue.pop(0)
                self.writer.write(cv_image)
            else:
                self.waitEvent.wait()
                self.waitEvent.clear()
        self.csvWriter.close()
        self.writer.release()


class RawImgPublisher(threading.Thread):
    def __init__(self, broker, nodename, output_topic, monitor_topic, queue_size, fps):
        threading.Thread.__init__(self)

        self.frame_queue = []
        self.queue_size = queue_size
        self.fps = fps
        self.recorder = None
        self.waitEvent = threading.Event()
        if fps <= 0:
            self.interval = 0
        else:
            self.interval = 1/fps

        # feature: screenshot
        self.oneShot = False
        self.oneShotPath = f"{ROOTDIR}/replay/screenshot.jpg"

        # setup MQTT client
        client = mqtt.Client()
        # client.on_publish = self.on_publish
        client.connect(broker)

        self.client = client
        self.nodename = nodename
        self.output_topic = output_topic
        self.monitor_topic = monitor_topic

        logging.info("Raw Image Publisher output topic: {}, monitor topic: {}".format(output_topic, monitor_topic))

        self.timestamp_base = 1
        self.timestamp_tick_frequency = 1


        self.frame_id = -1

    def set_timestamp_base(self, t):
        self.timestamp_base = t

    def set_timestamp_tick_frequency(self, f):
        self.timestamp_tick_frequency = f

    def setRecorder(self, recorder):
        self.recorder = recorder

    def clearRecorder(self):
        self.recorder = None

    def __len__(self):
        return len(self.frame_queue)

    def try_put_frame(self, frame: Optional[Frame]):
        if len(self.frame_queue) < self.queue_size :
            self.frame_queue.append(frame)
            self.waitEvent.set()
            fid = frame.get_id()-1
            if fid != self.frame_id+1:
                for i in range(self.frame_id+1, fid):
                    logging.warning(f"[{self.nodename}] Dropped Frame {i}.")
            self.frame_id = fid
        else:
            # discard 1 second data
            del self.frame_queue[0:self.fps]
            logging.warning(f"[{self.nodename}] Frame Queue is full. Discard.")

    def on_publish(self, mosq, userdata, mid):
        logging.debug("send")

    def stop(self):
        logging.debug(f"Raw Image Publisher {self.output_topic} stop")
        self.alive = False

    def screenshot(self, saved_path):
        self.oneShotPath = saved_path
        self.oneShot = True

    def setResize(self, flag, width=720, height=540):
        self.isResize = flag
        self.width = width
        self.height = height

    def run(self):
        logging.debug(f"Raw Image Publisher {self.output_topic} started.")
        self.alive = True
        logging.info(f"Raw Image Publisher {self.output_topic} is ready.")
        while self.alive:
            if self.frame_queue:
                frame = self.frame_queue.pop(0)
                if self.timestamp_tick_frequency == 1 or self.timestamp_base == 1:
                    logging.warning(f"[{self.nodename}] AV CAMERA TIMESTAMP DIDN'T RESET!")
                timestamp = frame.get_timestamp()/self.timestamp_tick_frequency + self.timestamp_base
                # logging.info(f"Time:{timestamp}, queue:{len(self.frame_queue)}")
                if frame != None:
                    # publish raw image
                    frame.convert_pixel_format(PixelFormat.Bgr8)
                    cv_image_origin = frame.as_opencv_image()
                    # screen shot origin picture
                    if self.oneShot:
                        cv2.imwrite(self.oneShotPath, cv_image_origin)
                        self.oneShot = False
                    if self.isResize:
                        cv_image_resized = cv2.resize(cv_image_origin, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    if self.recorder is not None and self.recorder.isRecording:
                        # Recorded CSV Frame ID start from 0
                        self.recorder.try_put_frame(cv_image_origin)
                        self.recorder.try_write_timestamp(fid=frame.get_id()-1, timestamp=timestamp)

                    ret, buf = cv2.imencode('.jpg', cv_image_resized)
                    if ret == True:
                        # CSV Frame ID start from 0
                        imdata = pickle.dumps(buf)
                        payload = { 'id': frame.get_id()-1, 'timestamp': timestamp, 'raw_data': base64.b64encode(imdata).decode('ascii')}

                        self.client.publish(self.output_topic, json.dumps(payload))

                    # publish timestamp for monitor
                    # payload = { 'id': frame.get_id(), 'timestamp': frame.get_timestamp()}
                    # self.client.publish(self.monitor_topic, json.dumps(payload))

            else:
                self.waitEvent.wait()
                self.waitEvent.clear()

        logging.info(f"Raw Image Publisher {self.output_topic} terminated.")

class FrameHandler:
    def __init__(self, rawImgPublisher):

        self.rawImgPublisher = rawImgPublisher
        self.shutdown_event = threading.Event()

    def stop(self):
        self.shutdown_event.set()

    def __call__(self, cam: Camera, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            frame_copy = copy.deepcopy(frame)
            self.rawImgPublisher.try_put_frame(frame_copy)
        cam.queue_frame(frame)

class CameraReader(threading.Thread):
    def __init__(self, rawImgPublisher, cam_id, camera_cfg, settings):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        
        # camera initial
        self.setupCamera(cam_id, camera_cfg, settings)

        self.rawImgPubliser = rawImgPublisher
        self.handler = FrameHandler(rawImgPublisher)

    def setupCamera(self, cam_id, camera_cfg, settings):
        with Vimba.get_instance():
            with get_camera(cam_id) as cam:
                self.camera = cam
        cfg = load_camera_config(camera_cfg)

        # setup Camera by HW configure file
        for name, value in cfg.items('Camera'):
            self.setFeature(name, value)

        # setup Camera FPS, width, height by project config
        self.setFeature('Framerate', float(settings['fps']))
        self.setFeature('Width', int(settings['width']))
        self.setFeature('Height', int(settings['height']))

    def stop(self):
        self.stopStreaming()

    def stopStreaming(self):
        self.handler.stop()
        logging.info(f"Camera {self.cam_id} stop streaming...")

    def run(self):
        logging.debug(f"Camera Reader {self.cam_id} started.")
        with Vimba.get_instance():
            with self.camera:
                try:
                    # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                    self.reset_camera()
                    self.camera.start_streaming(handler=self.handler,)
                    self.handler.shutdown_event.wait()
                finally:
                    self.camera.stop_streaming()
        logging.info(f"Camera Reader {self.cam_id} terminated.")

    def reset_camera(self):
        # Avoid Dropping Frames
        # feature = self.camera.get_feature_by_name("StreamHoldEnable")
        # feature.set('On')

        # Camera Time Tick Frequency (Default: 10^9)
        tick = self.camera.get_feature_by_name("GevTimestampTickFrequency").get()
        self.rawImgPubliser.set_timestamp_tick_frequency(tick)

        # Reset Camera Interval Time to 0 (Default Unit: nanoseconds, MAX: 2^63)
        cmd = self.camera.get_feature_by_name("GevTimestampControlReset")
        cmd.run()

        # Unix Timestamp when Camera start streaming
        self.rawImgPubliser.set_timestamp_base(datetime.now().timestamp())

    def setBalanceRatio(self, name, value):
        feature = self.camera.get_feature_by_name('BalanceWhiteAuto')
        feature.set('Off')
        selector = self.camera.get_feature_by_name('BalanceRatioSelector')
        ratio = self.camera.get_feature_by_name('BalanceRatioAbs')
        selector.set(name)
        # range 0.80 - 3.00 <= 1 - 255
        val = (float(value) + 45.0) / 100
        min_, max_ = ratio.get_range()
        if val < min_:
            val = min_
        elif val > max_:
            val = max_
        ratio.set(val)

    def setGain(self, value):
        feature = self.camera.get_feature_by_name('GainAuto')
        feature.set('Off')
        gain = self.camera.get_feature_by_name('Gain')
        # range 0 - 40 <= 0 - 480
        val = int(value/480 * 40)
        min_, max_ = gain.get_range()
        if val < min_:
            val = min_
        elif val > max_:
            val = max_
        gain.set(val)

    # change camera settings
    def setFeature(self, name, value):
        with Vimba.get_instance():
            with self.camera:
                try:
                    if name == 'BalanceRatioRed':
                        self.setBalanceRatio('Red', value)
                    elif name == 'BalanceRatioBlue':
                        self.setBalanceRatio('Blue', value)
                    elif name == 'ExposureTimeAbs':
                        val = int(value) * 1000
                        feature = self.camera.get_feature_by_name('ExposureAuto')
                        feature.set('Off')
                        feature = self.camera.get_feature_by_name(name)
                        feature.set(val)
                    elif name == 'Gain':
                        self.setGain(int(value))
                    elif name == 'Brightness':
                        pass
                    elif name == 'Framerate': # TODO Remove outside
                        feature = self.camera.get_feature_by_name('AcquisitionFrameRateAbs')
                        feature.set(value)
                    else:
                        feature = self.camera.get_feature_by_name(name)
                        feature.set(value)
                except (AttributeError, VimbaFeatureError):
                    logging.error(f"Camera {self.cam_id} Feature {name} not found.")
                    pass

class MainThread(threading.Thread):
    def __init__(self, args, settings):
        threading.Thread.__init__(self)

        self.project_name = args.project
        self.nodename = args.nodename
        self.settings = settings

        # camera fps, width, height
        self.fps = float(self.settings['fps'])
        self.width = int(self.settings['width'])
        self.height = int(self.settings['height'])

        # setup buffer queue
        queue_size = int(self.settings['queue_size'])
        broker = self.settings['mqtt_broker']

        # setup Raw Image Publisher
        output_topic = self.settings['output_topic']
        monitor_topic = self.settings['monitor_topic']

        self.rawImgPublisher = RawImgPublisher(broker=broker,
                                               nodename=self.nodename,
                                               output_topic=output_topic,
                                               monitor_topic=monitor_topic,
                                               queue_size=queue_size,
                                               fps=self.fps)

        # setup Camera Reader
        cam_id = self.settings['hw_id']
        camera_cfg = f"{DIRNAME}/location/{self.settings['place']}/{self.settings['hw_id']}.cfg"
        self.cameraReader = CameraReader(self.rawImgPublisher, cam_id, camera_cfg, settings)

        #setup MQTT client
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(broker)
        self.client = client
        self.general_topic = settings['general_topic']
        self.killswitch = threading.Event()

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

        # change camera setting
        if 'ExposureAuto' in cmds:
            self.setFeature('ExposureAuto', cmds['ExposureAuto'])
        if 'Gain' in cmds:
            self.setFeature('Gain', cmds['Gain'])
        if 'GainAuto' in cmds:
            self.setFeature('GainAuto', cmds['GainAuto'])
        if 'Brightness' in cmds:
            self.setFeature('Brightness', cmds['Brightness'])
        if 'BalanceWhiteAuto' in cmds:
            self.setFeature('BalanceWhiteAuto', cmds['BalanceWhiteAuto'])
        if 'ExposureTimeAbs' in cmds:
            self.setFeature('ExposureTimeAbs', cmds['ExposureTimeAbs'])
        if 'BalanceRatioRed' in cmds:
            self.setFeature('BalanceRatioRed', cmds['BalanceRatioRed'])
        if 'BalanceRatioBlue' in cmds:
            self.setFeature('BalanceRatioBlue', cmds['BalanceRatioBlue'])
        if 'Framerate' in cmds: # TODO Remove outside
            self.setFeature('Framerate', cmds['Framerate'])

        # feature: screenshot
        if 'Screenshot' in cmds:
            self.screenshot(cmds['Screenshot'])

        # backup Video
        if 'startRecording' in cmds:
            if cmds['startRecording'] == True:
                if 'video_path' in cmds:
                    video_path = cmds['video_path']
                    # copy project config
                    project_cfg = f"{ROOTDIR}/projects/{self.project_name}.cfg"
                    destination = f"{video_path}/{self.project_name}.cfg"
                    shutil.copyfile(project_cfg, destination)
                    # copy device config
                    camera_cfg = f"{DIRNAME}/location/{self.settings['place']}/{self.settings['hw_id']}.cfg"
                    destination = f"{video_path}/{self.settings['hw_id']}.cfg"
                    shutil.copyfile(camera_cfg, destination)
                    # start Recording
                    self.startRecording(video_path=video_path, camera_cfg=camera_cfg)
                else:
                    logging.warning("please set video path in cmds('video_path')")
        elif 'stopRecording' in cmds:
            if cmds['stopRecording'] == True:
                    # stop Recording
                    self.stopRecording()

    def screenshot(self, save_path):
        logging.info(f"To save screen in path {save_path}")
        self.rawImgPublisher.screenshot(save_path)

    def startRecording(self, video_path, camera_cfg):
        self.recorder = Recorder(nodename=self.nodename,
                                 video_path=video_path,
                                 fps=self.fps,
                                 width=self.width,
                                 height=self.height)
        self.rawImgPublisher.setResize(True, 512, 288)
        self.recorder.start()
        self.rawImgPublisher.setRecorder(self.recorder)
        logging.info(f"{self.settings['hw_id']} Start Recording...")

    def stopRecording(self):
        self.rawImgPublisher.setResize(True)
        if self.recorder.is_alive():
            self.recorder.stop()
            self.recorder.join()
        logging.info(f"{self.settings['hw_id']} Stop Recording...")

    def stopStreaming(self):
        self.cameraReader.stopStreaming()
        logging.info(f"{self.settings['hw_id']} Stop Streaming...")

    def startStreaming(self):
        self.rawImgPublisher.setResize(True)
        self.cameraReader.start()
        logging.info(f"{self.settings['hw_id']} Start Streaming...")

    def stop(self):
        self.killswitch.set()
        logging.info(f"{self.nodename} MainThread Stop...")

    # Camera Settings check
    def setFeature(self, name, value):
        if name == 'ExposureTimeAbs' or name == 'BalanceRatioRed' or name == 'BalanceRatioBlue' or name == 'Gain':
            self.cameraReader.setFeature(name, int(value))
        elif name == 'ExposureAuto' or name == 'BalanceWhiteAuto' or name == 'GainAuto':
            if value == 'On':
                value = 'Continuous'
            else:
                value = 'Off'
            self.cameraReader.setFeature(name, value)

        elif name == 'Framerate':
            self.cameraReader.setFeature(name, float(value))

    def run(self):
        try:
            self.rawImgPublisher.start()
            self.client.loop_start()
            logging.info(f"{self.nodename} is ready.")
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

        logging.info(f"{self.nodename} is terminated.")

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
    settings = load_config(projectCfg, args.nodename)

    # Start MainThread
    mainThread = MainThread(args, settings)
    mainThread.start()
    mainThread.join()

if __name__ == '__main__':
    main()
