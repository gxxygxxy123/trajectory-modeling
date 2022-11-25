import argparse
import os
import sys
import time
import cv2
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Cut Pre & Post points for csv/video')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    parser.add_argument('--pre', type=int, required=True, help = 'Remove pre-N points')
    parser.add_argument('--post', type=int, required=True, help = 'Remove post-N points')
    args = parser.parse_args()

    files = os.listdir(args.folder)

    for f in sorted(files):
        if os.path.isfile(os.path.join(args.folder,f)) and f.endswith(".avi"):
            video_file = os.path.join(args.folder,f)
            video = cv2.VideoCapture(video_file)
            total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            _width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            _height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _fps = float(video.get(cv2.CAP_PROP_FPS))
            _fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
            output_video = cv2.VideoWriter(os.path.join(args.folder,"cut_"+f), _fourcc, _fps, (_width, _height))

            images = [None] * total_frame

            i = 0
            while(True):
                success, image = video.read()
                if not success:
                    break
                images[i] = image
                i = i+1
            assert i == total_frame, f"{i} not equal {total_frame}"
            for i in range(args.pre,total_frame-args.post):
                output_video.write(images[i])
            output_video.release()
            shutil.move(os.path.join(args.folder,"cut_"+f), video_file)

        elif os.path.isfile(os.path.join(args.folder,f)) and f.endswith(".csv"):
            csv_file = os.path.join(args.folder,f)
            df = pd.read_csv(csv_file)
            df = df.iloc[args.pre:df.shape[0]-args.post,:]

            if f.endswith("_ball.csv"):
                df['Frame'] -= args.pre
            df.to_csv(csv_file, encoding = 'utf-8', index = False)