"""
FileReader : To read frames from the video file and publish on an MQTT topic.
"""
import argparse
import base64
import csv
import json
import logging
import os
import pickle
import sys
import threading
import time
from typing import Optional

import cv2
import paho.mqtt.client as mqtt

# Our System's library
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from common import loadNodeConfig
from frame import Frame
from inspector import sendNodeStateMsg, sendPerformance


class RawImgPublisher(threading.Thread):
    def __init__(self, name, broker, output_topic, reader):
        threading.Thread.__init__(self)

        self.name = name
        self.reader = reader
        self.isReaderFinish = False
        self.interval = 0

        self.alive = False

        # setup MQTT client
        client = mqtt.Client()
        #client.on_publish = self.on_publish
        client.connect(broker)
        self.client = client
        self.output_topic = output_topic

        logging.info("Raw Image Publisher is ready. output topic: {}".format(output_topic))

    def on_publish(self, mosq, userdata, mid):
        logging.debug("send")

    def stop(self):
        self.alive = False

    def setFPS(self, fps):
        if self.alive is False:
            if fps <= 0:
                self.interval = 0
            else:
                self.interval = 1/fps
            logging.info("Raw Image Publisher FPS: {}".format(fps))

    def run(self):
        try:
            self.alive = True
            while self.alive:
                if len(self.reader.frames) > 0:
                    frame = self.reader.frames.pop(0)

                    # publish raw image
                    payload = { 'id': frame.fid, 'raw_data': frame.raw_data, "timestamp": frame.timestamp}
                    #logging.info("send frame id:{}".format(frame.fid))
                    self.client.publish(self.output_topic, json.dumps(payload))
                    # publish progress
                    progress = int(((frame.fid - self.reader.first_id)/ self.reader.fileLength) * 100)
                    payload = { 'progress': progress}
                    #logging.debug("topic: {} msg: {}".format(self.name, json.dumps(payload)))
                    self.client.publish(self.name, json.dumps(payload))

                    # publish timestamp for monitor
                    sendPerformance(self.client, self.output_topic, 'none', 'send', [frame.fid])

                    # wait for timeout (FPS)
                    if self.interval:
                        time.sleep(self.interval)
                elif not self.reader.is_alive():
                    break
                else:
                     time.sleep(0.1)
        finally:
            payload = { 'progress': 100}
            self.client.publish(self.name, json.dumps(payload))
            logging.info("Raw Image Publisher terminated.")

class FileReader(threading.Thread):
    def __init__(self, queue_size):
        threading.Thread.__init__(self)

        self.frames = []
        self.queue_size = queue_size
        self.fileName = ""

    def setVideoFile(self, fileName):
        self.videoFile = fileName

    def setCSVFile(self, fileName):
        self.csvFile = fileName

    def run(self):
        cap = cv2.VideoCapture(self.videoFile)

        if (cap.isOpened() == False):
            logging.error("Error opening video stream of file '%s'" %(self.videoFile))
            return
        else:
            logging.info("Opening video stream of file '%s'"%(self.videoFile))
            logging.info("Opening csv file '%s'"%(self.csvFile))

        self.fileLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            fids = []
            timestamps = []
            with open(self.csvFile, newline='') as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    fid = int(row["Frame"])
                    timestamp = float(row["Timestamp"])
                    fids.append(fid)
                    timestamps.append(timestamp)
            idx = 0
            self.first_id = fids[idx]
            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    logging.info("the video file is end.")
                    break
                if len(self.frames) < self.queue_size:
                    ret, buf = cv2.imencode('.jpg', img) # encode to jpg
                    if ret == True:
                        imdata = pickle.dumps(buf)
                        raw_data = base64.b64encode(imdata).decode('ascii')
                        fid = fids[idx]
                        timestamp = timestamps[idx]
                        frame = Frame(fid=fid, timestamp=timestamp, raw_data=raw_data)
                        self.frames.append(frame)
                        idx += 1
                else:
                    logging.warning("the frame queue is full.")
                    break
        except Exception as e:
            logging.error(e)

        cap.release()

class MainThread(threading.Thread):
    def __init__(self, args, settings):
        threading.Thread.__init__(self)

        self.nodename = args.nodename
        self.settings = settings
        self.isStart = False

        queue_size = int(self.settings['queue_size'])
        broker = self.settings['mqtt_broker']

        # setup File Reader
        self.fileReader = FileReader(queue_size)
        self.fileReader.setVideoFile(args.file)
        self.fileReader.setCSVFile(args.csv)

        # setup Raw Image Publisher
        output_topic = self.settings['output_topic']
        fps = int(self.settings['publish_fps'])
        self.rawImgPublisher = RawImgPublisher(self.nodename, broker, output_topic, self.fileReader)
        #self.rawImgPublisher.setFPS(fps)

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
        elif 'startStreaming' in cmds:
            if cmds['startStreaming'] == True:
                self.startStreaming()
            else:
                self.stopStreaming()
        elif 'setFPS' in cmds:
            self.rawImgPublisher.setFPS(int(cmds['setFPS']))
        elif 'setVideoPath' in cmds:
            self.fileReader.setVideoFile(cmds['setVideoPath'])

    def stopStreaming(self):
        logging.info("Stop Streaming...")
        self.isStart = False
        if self.rawImgPublisher.is_alive():
            self.rawImgPublisher.stop()
            self.rawImgPublisher.join()
        if self.fileReader.is_alive():
            self.fileReader.join()

    def startStreaming(self):
        logging.info("Start Streaming...")
        if not self.isStart:
            self.isStart = True
            self.fileReader.start()
            self.rawImgPublisher.start()

    def stop(self):
        self.killswitch.set()

    def run(self):
        logging.debug("{} started.".format(self.nodename))

        # start
        try:
            self.client.loop_start()
            logging.info("{} is ready.".format(self.nodename))
            sendNodeStateMsg(self.client, self.nodename, "ready")
            self.killswitch.wait()
        finally:
            self.stopStreaming()
            sendNodeStateMsg(self.client, self.nodename, "terminated")
            self.client.loop_stop()
        logging.info("{} terminated.".format(self.nodename))

def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = 'FileReader')
    parser.add_argument('--project', type=str, default = 'coachbox', help = 'project name (default: coachbox)')
    parser.add_argument('--nodename', type=str, default = 'FileReader', help = 'mqtt node name (default: FileReader)')
    parser.add_argument('--file', type=str, default = '../Reader/FileReader/left.avi', help = 'mqtt node name (default: Monitor)')
    parser.add_argument('--csv', type=str, default = '../Reader/FileReader/left.csv', help = 'mqtt node name (default: Monitor)')
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
