"""
CSVReader : To read ball x,y from the tracknet csv and publish on an MQTT topic.
"""
# Warning NO Timestamp info sended (no timestamp in old csv before 2021.10.20)

import os
import csv
import threading
import sys
import cv2
import logging
import configparser
import queue
import base64
import paho.mqtt.client as mqtt
import numpy as np
import time
import json
import pickle
import argparse

from datetime import datetime
from typing import Optional

# Our System's library
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from common import load_config
from inspector import sendPerformance, sendNodeStateMsg
from point import Point, sendPoints

class CSVPublisher(threading.Thread):
    def __init__(self, broker, output_topic):
        threading.Thread.__init__(self)

        self.isReaderFinish = False
        self.interval = 30 # Default 30 TODO?
        self.alive = False

        # setup MQTT client
        client = mqtt.Client()
        #client.on_publish = self.on_publish
        client.connect(broker)
        self.client = client
        self.output_topic = output_topic

        logging.info("CSV Publisher is ready. output topic: {}".format(output_topic))

        self.csv = None
        self.waitEvent = threading.Event()

    def on_publish(self, mosq, userdata, mid):
        logging.debug("send")

    def stop(self):
        self.alive = False
        self.waitEvent.set()

    def setFPS(self, fps):
        logging.info('fps:{}'.format(fps))
        if fps <= 0:
            self.interval = 0
        else:
            self.interval = 1/fps
        logging.info("CSV Publisher FPS: {}".format(fps))

    def setCSVFile(self, csv):
        if os.path.isfile(csv):
            self.csv = csv
            self.waitEvent.set()
        else:
            logging.error("[CSVReader] CSV FILE NOT EXISTS !")

    def run(self):
        try:
            self.alive = True
            while self.alive:
                if self.csv is not None:
                    with open(self.csv, newline='') as csvfile:
                        rows = csv.DictReader(csvfile)
                        for row in rows:
                            # logging.debug(row)
                            if not self.alive:
                                break
                            else:
                                if int(row['Visibility']) == 1:
                                    point = Point(fid=row['Frame'],
                                                  timestamp=row['Timestamp'],
                                                  visibility=row['Visibility'],
                                                  x=row['X'],
                                                  y=row['Y'],
                                                  z=row['Z'],
                                                  event=row['Event'])
                                                  # speed=row['Speed'])
                                    sendPoints(self.client, self.output_topic, point)
                            if self.interval:
                                time.sleep(self.interval)
                    self.csv = None
                else:
                    self.waitEvent.wait()
                    self.waitEvent.clear()
        except Exception as e:
            logging.error(e)
        finally:
            logging.info("CSV Publisher terminated.")

class MainThread(threading.Thread):
    def __init__(self, args, settings):
        threading.Thread.__init__(self)

        self.nodename = args.nodename
        self.settings = settings
        self.isStart = False

        # setup CSV Publisher
        broker = self.settings['mqtt_broker']
        output_topic = self.settings['output_topic']
        fps = int(self.settings['publish_fps'])
        self.csvPublisher = CSVPublisher(broker, output_topic)
        self.csvPublisher.setCSVFile(args.csv)
        self.csvPublisher.setFPS(fps)

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
        self.client.subscribe('system_control')

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
            self.csvPublisher.setFPS(int(cmds['setFPS']))
        elif 'setCSVFile' in cmds:
            self.csvPublisher.setCSVFile(cmds['setCSVFile'])

    def stopStreaming(self):
        logging.info("Stop Streaming...")
        self.isStart = False
        if self.csvPublisher.is_alive():
            self.csvPublisher.stop()
            self.csvPublisher.join()

    def startStreaming(self):
        logging.info("Start Streaming...")
        if not self.isStart:
            time.sleep(2) # wait for receiver TODO
            self.isStart = True
            self.csvPublisher.start()

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
    parser = argparse.ArgumentParser(description = 'CSVReader')
    parser.add_argument('--project', type=str, default = 'data_collect', help = 'project name (default: data_collect)')
    parser.add_argument('--nodename', type=str, default = 'CSVReader', help = 'mqtt node name (default: CSVReader)')
    parser.add_argument('--csv', type=str, default = None, help = 'csv filename (default: None')
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
