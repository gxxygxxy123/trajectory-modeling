import argparse

import os
import sys

import time

import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'fake timestamp of dataset')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    parser.add_argument('--fps', type=float, default=None, help = 'FPS')
    args = parser.parse_args()
    csv = ['CameraReaderL_ball.csv','CameraReaderR_ball.csv']
    frame = 0

    for f in sorted(os.listdir(args.folder)):
        if os.path.isdir(os.path.join(args.folder,f)):
            for c in csv:
                if os.path.exists(os.path.join(args.folder,f,c)):
                    csv_file = os.path.join(args.folder,f,c)
                    df = pd.read_csv(csv_file)

                    df.Timestamp = np.arange(0,df.shape[0]) / args.fps

                    df.to_csv(csv_file, encoding = 'utf-8', index = False)
                    print(f"{csv_file} Done")
