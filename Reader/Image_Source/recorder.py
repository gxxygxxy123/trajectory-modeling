import cv2
import threading
import logging
import sys
import os
import gi
import time
import subprocess

gi.require_version("Tcam", "0.1")
gi.require_version("Gst", "1.0")

from gi.repository import GLib, GObject, Gst, Tcam

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from writer import CSVWriter
from point import Point
from camera import convert_sample_to_numpy

class Recorder(threading.Thread):
    def __init__(self, name, path, config):
        threading.Thread.__init__(self)
        # camera fps, width, height
        fps = int(config['fps'])
        width = int(config['width'])
        height = int(config['height'])

        self.fileName = f"{path}/{name}.avi"
        self.csvWriter = CSVWriter(name=path, filename=f"{path}/{name}.csv")
        self.writer = cv2.VideoWriter(self.fileName, cv2.VideoWriter_fourcc(*'XVID'), 30, (1440, 1080))

        self.waitEvent = threading.Event()
        self.alive = False
        self.frame_queue = []

    def try_put_frame(self, fid, sample, timestamp_base):
        timestamp_frame, image = convert_sample_to_numpy(sample)
        timestamp = timestamp_base + timestamp_frame
        self.frame_queue.append(image)
        self.csvWriter.writePoints(Point(fid=fid,timestamp=timestamp))
        self.waitEvent.set()

    def stop(self):
        self.alive = False
        self.waitEvent.set()

    def run(self):
        self.alive = True
        print("recorder start")
        while self.alive:
            if len(self.frame_queue) > 0:
                cv_image = self.frame_queue.pop(0)
                #self.writer.write(cv_image)
            else:
                self.waitEvent.wait()
                self.waitEvent.clear()
        print("recorder stop")
        self.csvWriter.close()
        self.writer.release()

class CameraRecorder():
    def __init__(self, name, path, config):
        # camera fps, width, height
        fps = int(config['fps'])
        width = int(config['width'])
        height = int(config['height'])

        self.name = name
        self.recorder = None
        # open device
        self.setCaps(width, height, fps)
        self.openDevice(name, path)
        # open csv writer
        self.csvWriter = CSVWriter(name=name, filename=f"{path}/{name}.csv")

    def setCaps(self, width=1440, height=1080, fps=60):
        caps = Gst.Caps.new_empty()
        format = f"video/x-raw, format=BGRx, width={int(width)}," \
                    f"height={int(height)}," \
                    f"framerate={int(fps)}/1"
        structure = Gst.Structure.new_from_string(format)
        caps.append_structure(structure)
        structure.free()
        self.caps = caps

    def openDevice(self, name, path):
        Gst.init(sys.argv)
        recorder_name = "recorder-src"
        try:
            if self.recorder is not None:
                (ret, state, pending) = self.recorder.get_state(0)
                logging.debug(f"{ret}, {state}, {pending}")
                if state == Gst.State.PLAYING:
                    self.stop()
            self.file_location = f"{path}/{name}.avi"
            self.csvWriter = CSVWriter(name=name, filename=f"{path}/{name}.csv")
            record_str = (
                "appsrc name={} is-live=true format=3 "
                "! {} "
                "! queue "
                "! videoconvert "
                #"! x264enc ! mp4mux "
                "! avimux "
                "! filesink name=fsink location={}").format(
                                                recorder_name,
                                                self.caps.to_string(),
                                                self.file_location)
            self.recorder = Gst.parse_launch(record_str)
            logging.info(f"Recording file to {self.file_location}")
        except GLib.Error as error:
            logging.error(f"Error creating camera: {error}")
            raise

        self.src = self.recorder.get_by_name(recorder_name)
        self.src.set_property("caps", self.caps)
        self.src.set_property("do-timestamp", True)

        #self.recorder.set_name(recorder_name)

    def try_put_frame(self, fid, sample, timestamp_base):
        buf = sample.get_buffer()
        timestamp_frame = buf.pts / 1000000000
        timestamp = timestamp_base + timestamp_frame
        self.src.emit("push-buffer", buf)
        self.csvWriter.writePoints(Point(fid=fid,timestamp=timestamp))

    def start(self):
        if self.recorder is not None:
                (ret, state, pending) = self.recorder.get_state(0)
                logging.debug(f"{ret}, {state}, {pending}")
                if state == Gst.State.PLAYING:
                    logging.info(f"Camera {self.name} is Recording...")
                    return
        logging.info(f"Camera {self.name} start Recording...")
        self.recorder.set_state(Gst.State.PLAYING)

        bus = self.recorder.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)

    def stop(self):
        self.recorder.send_event(Gst.Event.new_eos())
        self.csvWriter.close()

        # Video Compression
        tmp_file = os.path.splitext(self.file_location)[0] + '_tmp.avi'
        cmd = (f"ffmpeg -i {self.file_location} -c:v libxvid -q:v 5 -q:a 5 {tmp_file} && "
              f"mv -f {tmp_file} {self.file_location}")
        detail = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, close_fds=True, shell=True)
        logging.info(f"Video Compression cmd: {cmd}")

    def _bus_call(self, gst_bus, message):
        t = message.type
        logging.debug(f"Received msg from {message.src.get_name()}")
        if message.src.get_name() == "fsink":
            self.file_location = ""

        if t == Gst.MessageType.EOS:
            logging.debug(f"Received EOS from {message.src.get_name()}")
            if (message.src.get_name() == "fsink"):
                logging.debug(f"sink sent EOS {message.get_structure().to_string()}")
                self.file_location = ""
                self.recorder.set_state(Gst.State.PAUSED)
                self.recorder.set_state(Gst.State.READY)
                self.recorder.set_state(Gst.State.NULL)
                self.recorder_src = None
                self.recorder = None
