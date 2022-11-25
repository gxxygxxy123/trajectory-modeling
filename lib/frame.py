"""
Object: for each frame of camera or video file
"""
import base64
import pickle
import cv2

# improve encode/decode performance
#jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')

class Frame():
    def __init__(self, fid, timestamp, raw_data):
        self.fid = int(fid)
        self.timestamp = float(timestamp)
        self.raw_data = raw_data

    def coverToCV2(self):
        imdata = base64.b64decode(self.raw_data)
        buf = pickle.loads(imdata)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return image

    def coverToCV2ByTurboJPEG(self, jpeg):
        imdata = base64.b64decode(self.raw_data)
        buf = pickle.loads(imdata)
        image = jpeg.decode(buf)
        return image
