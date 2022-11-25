"""
EventDetector : To recognize the event (Hit, Land) by trajectory of 3D-Model
"""
import sys
import os
import logging
import configparser
import queue
import paho.mqtt.client as mqtt
import numpy as np
import json
import math
import time
import argparse

from numpy.linalg import norm
from datetime import datetime
from typing import Optional

'''
Our common function
'''
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from inspector import sendPerformance, sendNodeStateMsg
from point import Point, sendPoints
from writer import CSVWriter

def isHit(v1, v2, v3, v4, v5, GROUND_HEIGHT, max_time=0.5):
    # Small Mid Large Mid Small or L M S M L
    if ((v3.y - v2.y) * (v4.y - v3.y) < 0 and
       (v3.y - v2.y) * (v2.y - v1.y) > 0 and
       (v4.y - v3.y) * (v5.y - v4.y) > 0 and
       v3.z > GROUND_HEIGHT and
       # If time diff too long means ball out-and-in cameras
       abs(v5.timestamp - v1.timestamp) < max_time):
        logging.info(f"Time : {v3.timestamp:>.3f} --> HIT")
        v3.event = 1
        v3.color = 'blue'
        return True # hit point
    else:
        return False

def isLand(v1, v2, v3, GROUND_HEIGHT):
    if (v3.z <= GROUND_HEIGHT and
        v1.z > GROUND_HEIGHT and
        v2.z > GROUND_HEIGHT):
        logging.info(f"Time : {v3.timestamp:>.3f} --> LAND")
        v3.event = 2
        v3.color = 'red'
        return True
    else:
        return False

def isServe(v1, v2, v3, GROUND_HEIGHT, SERVE_HEIGHT):
    if (v3.z >= SERVE_HEIGHT and
        v1.z <= GROUND_HEIGHT and
        v2.z <= GROUND_HEIGHT):
        # Serve ball TODO First Serve will miss
        logging.info(f"Time : {v3.timestamp:>.3f} --> SERVE")
        v3.event = 3
        v3.color = 'green'
        return True
    else:
        return False

class EventDetector():
    def __init__(self, name, client, topic, writer3D, fps):
        self.name = name
        self.points = [] # 3D point
        self.fps = fps

        # setup MQTT client
        self.client = client
        self.topic = topic

        # 3d csv writer
        self.writer3D = writer3D

        self.GROUND_HEIGHT = 0.1
        self.MAX_DELAY_TIME = 1/self.fps
        self.SERVE_HEIGHT = 0.5 # Serving height should higher than this value
        self.pre_calculate_time = float("-inf")

    def addPoint(self, point):
        self.points.append(point)

    def detect(self):
        # 0: nothing, 1: Shot, 2: Land
        event = 0
        calculate_time = datetime.now().timestamp()
        if calculate_time - self.pre_calculate_time >= self.MAX_DELAY_TIME:
            self.pre_calculate_time = calculate_time
            #self.clear_queue()
        else:
            if len(self.points) >= 3:
                isLand(self.points[0], self.points[1], self.points[2], self.GROUND_HEIGHT)
                isServe(self.points[0], self.points[1], self.points[2], self.GROUND_HEIGHT, self.SERVE_HEIGHT)
            if len(self.points) >= 5:
                isHit(self.points[0], self.points[1], self.points[2], self.points[3], self.points[4], self.GROUND_HEIGHT)
                self.writer3D.writePoints(self.points[0])
                sendPoints(self.client,self.topic, self.points[0])
                del self.points[0]

    def close(self):
        self.clear_queue()

    def clear_queue(self):
        while self.points:
            # Duplicate Code TODO
            if len(self.points) >= 3:
                isLand(self.points[0], self.points[1], self.points[2], self.GROUND_HEIGHT)
                isServe(self.points[0], self.points[1], self.points[2], self.GROUND_HEIGHT, self.SERVE_HEIGHT)
            if len(self.points) >= 5:
                isHit(self.points[0], self.points[1], self.points[2], self.points[3], self.points[4], self.GROUND_HEIGHT)

            self.writer3D.writePoints(self.points[0])
            sendPoints(self.client,self.topic, self.points[0])
            del self.points[0]

    def publishHitEvent(self, points):
        # RNN need trajectory : {ball, "HIT, ball, ball, ball, ball"}
        sendPoints(self.client, self.topic, points)
        #sendPerformance(self.client, self.topic, 'none', 'send', fids)

    def publishLandEvent(self, point):
        sendPoints(self.client, self.topic, point)
        #sendPerformance(self.client, self.topic, 'none', 'send', [point.fid])

