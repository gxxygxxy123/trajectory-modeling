import os
import sys
import logging
import json
import threading
import paho.mqtt.client as mqtt
from datetime import datetime

class ServicesStartThread(threading.Thread):
    def __init__(self, mqtt_broker, node_names):
        super().__init__()
        # setup Event
        self.waitEvent = threading.Event()
        # setup MQTT client
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(mqtt_broker)
        self.client = client
        # check list
        self.check_list = {}
        for name in node_names:
            self.check_list[name] = False

    def on_connect(self, client, userdata, flag, rc):
        logging.debug(f"ServicesStartThread Connected with result code: {rc}")
        self.client.subscribe("system_status")

    def on_message(self, client, userdata, msg):
        cmds = json.loads(msg.payload)
        for key, value in self.check_list.items():
            if key in cmds:
                if cmds[key] == "ready":
                    self.check_list[key] = True
                    self.waitEvent.set()
                if cmds[key] == "terminated":
                    self.check_list[key] = False
                    self.waitEvent.set()
    def run(self):
        try:
            self.client.loop_start()
            while True:
                if all(value == True for value in self.check_list.values()):
                    break
                else:
                    self.waitEvent.wait()
                    self.waitEvent.clear()
            payload = {"startStreaming": True}
            self.client.publish('cam_control', json.dumps(payload))
        finally:
            while True:
                if all(value == False for value in self.check_list.values()):
                    break
                else:
                    self.waitEvent.wait()
                    self.waitEvent.clear()
            self.client.loop_stop()



def sendPerformance(mqtt_client, from_topic, name, action, fids):
    return;
    # timestamp
    timestamp = datetime.now().timestamp()
    # setup kind
    # kind: 0: send, 1: received, 2: section_name start, 3: section_name end
    if action == 'send':
        kind = 0
    elif action == 'receive':
        kind = 1
    elif action == 'start':
        kind = 2
    elif action == 'end':
        kind = 3
    # setup MQTT message
    payload = {'name': name, 'from_topic':from_topic, 'kind':kind, 'fids':fids, 'timestamp':timestamp}
    mqtt_client.publish('performance', json.dumps(payload))

# 0: start, 1: ready, 2: terminated
def sendNodeStateMsg(mqtt_client, node_name, state):
    #setup MQTT message
    payload = {node_name: state}
    mqtt_client.publish('system_status', json.dumps(payload))
