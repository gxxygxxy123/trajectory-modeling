import sys
import os
import logging
import json
import queue
import threading
import time
from datetime import datetime
import paho.mqtt.client as mqtt

from frame import Frame
from point import Point
from common import insertById

class RawImgReceiver(threading.Thread):
    def __init__(self, broker, topic, queue_size, callback):
        super().__init__()
        self.killswitch = threading.Event()
        self.callback = callback

        self.topic = topic
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(broker)
        self.client = client

        self.queue = []
        self.queue_size = queue_size

    def try_put_frame(self, data):
        if len(self.queue) < self.queue_size :
            frame = Frame(data['id'], data['timestamp'], data['raw_data'])
            insertById(self.queue, frame)
            # if not self.callback.is_set():
            self.callback.set()
        else:
            # discard 1 second data
            del self.queue[0:30]
            logging.warning("Receiver [{}] is full.".format(self.topic))

    def run(self):
        logging.debug("Receiver [{}] started.".format(self.topic))
        # start
        try:
            self.client.loop_start()
            logging.info("Receiver [{}] is reading".format(self.topic))
            self.killswitch.wait()
        finally:
            self.client.loop_stop()
        # end
        logging.info("Receiver [{}] terminated".format(self.topic))

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"Receiver {self.topic} Connected with result code: {rc}")
        self.client.subscribe(self.topic)

    def on_message(self, client, userdata, msg):
        data = json.loads(msg.payload)
        self.try_put_frame(data)

    def stop(self):
        self.killswitch.set()

    def set_topic(self, topic, clear=True):
        if self.topic is not None:
            self.client.unsubscribe(self.topic)
        self.client.subscribe(topic)
        self.topic = topic
        if clear:
            self.queue.clear()

class PointReceiver(threading.Thread):
    def __init__(self, name, broker, topic, queue_size):
        super().__init__()
        self.killswitch = threading.Event()

        self.name = name
        self.topic = topic
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(broker)
        self.client = client

        self.queue = []
        self.queue_size = queue_size
        logging.debug(f"topic: {self.topic}, queue size:{queue_size} ")

    def try_put_point(self, data):
        if len(self.queue) < self.queue_size :
            for i in range(len(data['linear'])):
                point = Point(fid=data['linear'][i]['id'],
                            timestamp=data['linear'][i]['timestamp'],
                            visibility=data['linear'][i]['visibility'],
                            x=data['linear'][i]['pos']['x'],
                            y=data['linear'][i]['pos']['y'],
                            z=data['linear'][i]['pos']['z'],
                            event=data['linear'][i]['event'],
                            speed=data['linear'][i]['speed'])
                insertById(self.queue, point)
                # logging.debug("Receiver [{}] id:{}, ({:>2.3f}, {:>2.3f}, {:>2.3f}), timestamp:{}".format(self.topic, point.fid, point.x, point.y, point.z, point.timestamp))
        else:
            logging.warning("Receiver [{}] is full.".format(self.topic))

    def run(self):
        logging.debug("Receiver [{}] started.".format(self.topic))
        # start
        try:
            self.client.loop_start()
            logging.info("Receiver [{}] is reading".format(self.topic))
            self.killswitch.wait()
        finally:
            self.client.loop_stop()

        # end
        logging.info("Receiver [{}] terminated".format(self.topic))

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"Point Receiver {self.topic} Connected with result code: {rc}")
        self.client.subscribe(self.topic)

    def on_message(self, client, userdata, msg):
        data = json.loads(msg.payload)
        self.try_put_point(data)
        #fids = []
        #for i in range(len(data['linear'])):
        #    fids.append(data['linear'][i]['id'])
        #sendPerformance(self.client, self.topic, self.name, 'receive', fids)
        #sendPerformance(self.client, self.name, 'total', 'start', fids)

    def stop(self):
        self.killswitch.set()

    def clear(self):
        self.queue.clear()