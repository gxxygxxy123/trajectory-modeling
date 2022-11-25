"""
Triangulation : To combine two 2D points into 3D point
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import paho.mqtt.client as mqtt

# Our System's library
from EventDetector import CSVWriter, EventDetector
from MultiCamTriang import MultiCamTriang

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from common import  loadConfig, loadNodeConfig
from inspector import sendNodeStateMsg
from point import Point
from receiver import PointReceiver
from writer import CSVWriter


def denoise(l, a):
    if len(l) == 0:
       lastpoint = Point(0,0,0,0)
    else:
       lastpoint = l[len(l)-1]

    for i in range(len(l)):
        p = l[i]
        if p.fid > a.fid:
            if (   ((abs(a.x - lastpoint.x) > 300) or (abs(a.y - lastpoint.y) > 300) or (abs(a.z - lastpoint.z) > 300)) and ((a.fid -  lastpoint.fid) == 1)    ):
                logging.info("not order noise : x = {} y = {}".format(a.x, a.y))
                logging.info("not order lastpoint {} {} {}".format( lastpoint.fid, lastpoint.x, lastpoint.y, lastpoint.z))
                return
            else:
                l.insert(i, a)
                return

    if (   ((abs(a.x - lastpoint.x) > 300) or (abs(a.y - lastpoint.y) > 300) or (abs(a.z - lastpoint.z) > 300)) and ((a.fid -  lastpoint.fid) == 1) and (a.fid != 1)  ):
        logging.info("Noise : id = {} x = {} y = {} ".format(a.fid, a.x, a.y))
        logging.info('lastpoint : {} {} {} {}'.format(lastpoint.fid, lastpoint.x, lastpoint.y, lastpoint.z))
        return
    else:
        l.append(a)

class RawTrack2D():
    def __init__(self, name, receiver):
        self.name = name
        self.points = []
        self.receiver = receiver

    def startReceiver(self):
        self.receiver.start()

    def stopReceiver(self):
        self.receiver.stop()
        self.receiver.join()

    def remove(self, idx):
        del self.points[idx]

    # TODO: timestamp replace fid
    def doInterpolation(self):
        if len(self.receiver.queue) >= 2:
            startPoint = self.receiver.queue.pop(0)
            endPoint = self.receiver.queue[0]
            xp = np.linspace(startPoint.x, endPoint.x, endPoint.fid - startPoint.fid + 1)
            yp = np.linspace(startPoint.y, endPoint.y, endPoint.fid - startPoint.fid + 1)
            idp = np.linspace(startPoint.fid, endPoint.fid, endPoint.fid - startPoint.fid + 1)
            newarr = [0 for i in range(endPoint.fid - startPoint.fid + 1)  ]
            for i in range(len(xp)):
                newarr[i] = Point(idp[i], 1, xp[i], yp[i], 0)
            del newarr[0]
            self.points.extend(newarr)

    def insertPoints(self):
        if len(self.receiver.queue) > 0:
            point = self.receiver.queue.pop(0)
            self.points.append(point)

class TriangulationThread(threading.Thread):
    def __init__(self, args, settings, ks, poses, eye, dist, newcameramtx, projection_mat):
        threading.Thread.__init__(self)

        self.fps = float(settings['fps'])

        self.ks = ks
        self.poses = poses
        self.eye = eye
        self.dist = dist
        self.newcameramtx = newcameramtx
        self.projection_mat = projection_mat
        self.nodename = args.nodename

        self.WAIT_TIME = 1/self.fps # waiting for other camaras point (sec)

        if os.path.isfile(args.save_path):
            save_path = os.path.dirname(os.path.abspath(args.save_path))
        else:
            save_path = os.path.abspath(args.save_path)

        input_topics = settings['input_topic'].split(',')
        queue_size = int(settings['queue_size'])
        broker = settings['mqtt_broker']
        self.topic = settings['output_topic']
        # event_topic = settings['output_event_topic']

        # setup MQTT client
        client = mqtt.Client()
        client.on_connect = self.on_connect
        #client.on_publish = self.on_publish
        client.connect(broker)
        self.client = client

        # 2D CSVWriters
        self.csv2DWriters = []

        self.max_FID = 0

        # Setup Event Detector
        filePath = f"{save_path}/{self.nodename}.csv"
        logging.debug("[{}]: Write 3d csv in {} by topic {}".format(self.nodename, filePath, self.topic))
        self.csv3DWriter = CSVWriter(name=self.topic, filename=filePath)
        self.eventDetector = EventDetector(self.nodename, self.client, self.topic, self.csv3DWriter, self.fps)

        # setup TrackNet Receiver
        self.rawTrack2Ds = []
        for in_topic in input_topics:
            receiver = PointReceiver(self.nodename, broker, in_topic, queue_size)
            rawTrack2D = RawTrack2D(name=in_topic, receiver=receiver)
            self.rawTrack2Ds.append(rawTrack2D)
            # Setup 2D track csv writer
            filePath = f"{save_path}/{in_topic}.csv"
            logging.debug("[{}]: Write 2d csv in {} by topic {}".format(self.nodename, filePath, in_topic))
            self.csv2DWriters.append(CSVWriter(name=in_topic, filename=filePath))

        # Setup MultiCamTriang
        self.multiCamTriang = MultiCamTriang(poses, eye, self.newcameramtx)
        self.alive = False

        self.MAXIMUM_FRAME_TIMESTAMP_DELAY = 1/(self.fps*2) # according to FPS

    def on_connect(self, client, userdata, flag, rc):
        logging.info(f"{self.nodename} Connected with result code: {rc}")

    def on_publish(self, mosq, userdata, mid):
        logging.debug("send")

    def stop(self):
        self.alive = False

    def setfps(self, fps):
        assert fps > 0, "[3DModel] FPS > 0"
        self.fps = fps

    def run(self):
        logging.info("TriangulationThread started.")

        self.alive = True
        # start receivers
        for rawTrack2D in self.rawTrack2Ds:
            rawTrack2D.startReceiver()

        prev_point3d = None
        prev_calculate_time = float("inf")

        sendNodeStateMsg(self.client, self.nodename, "ready")

        while self.alive:
            # get point from receiver and to do interpolation
            for rawTrack2D in self.rawTrack2Ds:
                # rawTrack2D.doInterpolation()
                rawTrack2D.insertPoints()
            # Do Triangulation
            points_2D = []

            # Only Several Cameras detect the ball, [0,1,3] means idx 0,1,3 cams detect, 2 misses
            cam_detected_ball = []

            # search smallest timestamp which have detected the ball
            queueAllNotEmpty = True
            force_3d_flag = False
            min_timestamp = float("inf")

            current_time = datetime.now().timestamp()
            for rawTrack2D in self.rawTrack2Ds:
                if len(rawTrack2D.points) > 0:
                    if min_timestamp > rawTrack2D.points[0].timestamp:
                        min_timestamp = rawTrack2D.points[0].timestamp

            # If a camera always miss, no wait if the time exceed WAIT_TIME
            #if current_time - prev_calculate_time >= self.WAIT_TIME:
            #    force_3d_flag = True

            for rawTrack2D in self.rawTrack2Ds:
                if len(rawTrack2D.points) <= 0:
                    queueAllNotEmpty = False

            point3d_fid = None

            if queueAllNotEmpty or force_3d_flag: # Do 3D Triangulation

                for i in range(len(self.rawTrack2Ds)):
                    if len(self.rawTrack2Ds[i].points) > 0:
                        point = self.rawTrack2Ds[i].points[0]
                        if abs(min_timestamp - point.timestamp) <= self.MAXIMUM_FRAME_TIMESTAMP_DELAY:

                            if point3d_fid is None:
                                point3d_fid = point.fid # The FID in which Camera idx is the smallest (In coachbox, it's always CameraL)

                            points_2D.append([[point.x, point.y]])
                            cam_detected_ball.append(i)

                            writer = next((w for w in self.csv2DWriters if w.name == self.rawTrack2Ds[i].name), None)
                            if writer:
                                writer.writePoints(point)

                            # logging.debug('cam {} detected_ball in frame_id {} :'.format(i, self.rawTrack2Ds[i].points[0].fid))
                            self.rawTrack2Ds[i].points.pop(0)
                            prev_calculate_time = datetime.now().timestamp()
                if cam_detected_ball:
                    cam_detected_ball = np.stack(cam_detected_ball, axis=0)


            # to generate point 3d
                if len(points_2D) >= 2:

                    track_2D = np.array(points_2D, dtype = np.float32) # shape:(num_cam,num_frame,2), num_frame=1
                    undistort_track2D_list = []
                    for i in range(len(points_2D)): # for each camera, do undistort
                        temp = cv2.undistortPoints(np.array(track_2D[i], np.float32),
                                                    np.array(self.ks[cam_detected_ball[i]], np.float32),
                                                    np.array(self.dist[cam_detected_ball[i]], np.float32),
                                                    None,
                                                    np.array(self.newcameramtx[cam_detected_ball[i]], np.float32)) # shape:(1,num_frame,2), num_frame=1
                        temp = temp.reshape(-1,2) # shape:(num_frame,2), num_frame=1
                        undistort_track2D_list.append(temp)
                    undistort_track2D = np.stack(undistort_track2D_list, axis=0) # shape:(num_cam,num_frame,2), num_frame=1

                    ###### Shao-Ping #####################################################################
                    # self.multiCamTriang.setTrack2Ds(track_2D)
                    # self.multiCamTriang.setPoses(self.poses[cam_detected_ball])
                    # self.multiCamTriang.setEye(self.eye[cam_detected_ball])
                    # self.multiCamTriang.setKs(self.ks[cam_detected_ball])
                    # track_3D = self.multiCamTriang.calculate3D()

                    ###### Ours ##########################################################################
                    self.multiCamTriang.setTrack2Ds(undistort_track2D)
                    self.multiCamTriang.setProjectionMats(self.projection_mat[cam_detected_ball])
                    track_3D = self.multiCamTriang.rain_calculate3D() # shape:(num_frame,3), num_frame=1
                    ######################################################################################

                    # Use Timestamp to triangulation, so fid is not correct [*]
                    point3d = Point(fid=point3d_fid,
                                    timestamp=min_timestamp,
                                    visibility=1,
                                    x=track_3D[0][0],
                                    y=track_3D[0][1],
                                    z=track_3D[0][2],
                                    color='white')


                    # Ball position (X,Z) when pass above the net (Y=0)
                    if prev_point3d is not None and (prev_point3d.y*point3d.y) <= 0:
                        # TODO if the ball hit the net may detect serveral times
                        tmpx = (prev_point3d.x - point3d.x)/(prev_point3d.y - point3d.y) * (0 - point3d.y) + point3d.x
                        tmpz = (prev_point3d.z - point3d.z)/(prev_point3d.y - point3d.y) * (0 - point3d.y) + point3d.z
                        logging.info(f"Ball Pass Above the Net between fid({prev_point3d.fid},{point3d.fid}): ({tmpx:.2f},{tmpz:.2f})")
                    prev_point3d = point3d

                    # event detect
                    self.eventDetector.addPoint(point3d)
                    self.eventDetector.detect()

            else:
                time.sleep(0.5/self.fps) # less than 1/fps

        # stop receivers
        for rawTrack2D in self.rawTrack2Ds:
            rawTrack2D.stopReceiver()

        self.eventDetector.close()

        # Output to CSV
        for w in self.csv2DWriters:
            w.close()
        self.csv3DWriter.close()

        logging.info("TriangulationThread terminated.")

class MainThread(threading.Thread):
    def __init__(self, args, settings, ks, poses, eye, dist, newcameramtx, projection_mat):
        threading.Thread.__init__(self)
        self.killswitch = threading.Event()

        self.nodename = args.nodename
        broker = settings['mqtt_broker']

        # ToDo: check num of ks, pose, eye is equal to number of input_topic
        self.triangulation = TriangulationThread(args, settings, ks, poses, eye, dist, newcameramtx, projection_mat)

        # Setup MQTT Client
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        self.client.connect(broker)

    def stop(self):
        finish = False
        while not finish:
            for rawTrack2D in self.triangulation.rawTrack2Ds:
                if len(rawTrack2D.points) <= 0:
                    finish = True
                    break

        self.killswitch.set()

    def on_connect(self, client, userdata, flag, rc):
        self.client.subscribe(self.nodename)
        self.client.subscribe('system_control')

    def on_message(self, client, userdata, msg):
        cmds = json.loads(msg.payload)
        if msg.topic == 'system_control':
            if 'stop' in cmds:
                if cmds['stop'] == True:
                    self.stop()
        else:
            if 'stop' in cmds:
                if cmds['stop'] == True:
                    self.stop()

    def run(self):
        logging.info("{} started.".format(self.nodename))
        # start
        try:
            self.triangulation.start()
            self.client.loop_start()
            logging.info("{} is ready.".format(self.nodename))

            self.killswitch.wait()
        finally:
            self.triangulation.stop()
            self.triangulation.join()
            sendNodeStateMsg(self.client, self.nodename, "terminated")
            self.client.loop_stop()

        # end
        logging.info("{} terminated.".format(self.nodename))

def parse_args() -> Optional[str]:
    # Args
    parser = argparse.ArgumentParser(description = '3DModel')
    parser.add_argument('--project', type=str, default = 'coachbox', help = 'project name (default: coachbox)')
    parser.add_argument('--nodename', type=str, default = '3DModel', help = 'mqtt node name (default: 3DModel)')
    parser.add_argument('--save_path', type=str, default = './', help = 'csv path (default: replay/XX/)')
    parser.add_argument('--camera_cfgs', type=str, nargs='+', required=True, help = 'several cameras configs')
    args = parser.parse_args()

    return args

def main():
    # Parse arguments
    args = parse_args()
    # Load configs+
    projectCfg = f"{ROOTDIR}/config"
    settings = loadNodeConfig(projectCfg, args.nodename)
    ks = []
    poses = []
    eye = []
    dist = []
    newcameramtx = []
    projection_mat = []
    for filename in args.camera_cfgs:
        cfg = loadConfig(filename)
        ks.append(np.array(json.loads(cfg['Other']['ks'])))
        poses.append(np.array(json.loads(cfg['Other']['poses'])))
        eye.append(np.array(json.loads(cfg['Other']['eye'])))
        dist.append(np.array(json.loads(cfg['Other']['dist'])))
        newcameramtx.append(np.array(json.loads(cfg['Other']['newcameramtx'])))
        projection_mat.append(np.array(json.loads(cfg['Other']['projection_mat'])))
    ks = np.array(ks)
    poses = np.array(poses)
    eye = np.array(eye)
    dist = np.array(dist)
    newcameramtx = np.array(newcameramtx)
    projection_mat = np.array(projection_mat)
    # Start MainThread
    mainThread = MainThread(args, settings, ks, poses, eye, dist, newcameramtx, projection_mat)
    mainThread.start()
    mainThread.join()

if __name__ == '__main__':
    main()
