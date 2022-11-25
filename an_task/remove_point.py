import argparse
import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from pandas import Timestamp
sys.path.append(f"../lib")
sys.path.append(f"../Model3D")
sys.path.append(f"../RNN")
from point import Point, load_points_from_csv, save_points_to_csv
from error_function import space_err, time_err, space_time_err
from threeDprojectTo2D import FitVerticalPlaneTo2D
from physic_model import physics_predict3d


import argparse
import json
import logging
import os
import sys
import threading
import time
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

from pandas import Timestamp
sys.path.append(f"../lib")
sys.path.append(f"../Model3D")
sys.path.append(f"../RNN")
from point import Point, load_points_from_csv, save_points_to_csv
from error_function import space_err, time_err, space_time_err
from physic_model import physics_predict3d
from dataloader import RNNDataSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Remove point')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    parser.add_argument('--fps', type=float, required=True, help = 'FPS')
    args = parser.parse_args()
    # Dataset
    DATASET = args.folder

    dataset = RNNDataSet(dataset_path=DATASET, fps=args.fps, smooth_2d=False)

    M = 30

    # boxplot
    box = {}
    for i in range(1,M+1):
        box[i] = []

    for idx, data in dataset.whole_3d().items():
        for i in range(M):
            if i >= data.shape[0]:
                break
            p1 = data[i,:-1]
            p2 = dataset.whole_3d2d()[idx][i,:-1]
            dist = np.linalg.norm(p1-p2)
            box[i+1].append(dist)

    fig, ax = plt.subplots()
    ax.boxplot(box.values())
    ax.set_xticklabels(box.keys())
    ax.set_xlabel("N-th point")
    ax.set_ylabel("Error(m)")
    ax.set_title(f"Distance from N-th point to plane")

    plt.show()

    """
    # boxplot
    box = {}
    for i in range(0,M+1):
        box[i] = []

    for idx, data in dataset.whole_3d().items():
        _, _, slope, intercept = FitVerticalPlaneTo2D(data, smooth_2d=False)
        for i in range(0,M+1):
            if i >= data.shape[0]-1: # Remove i points must leave at least two points
                break
            _, _, a, b = FitVerticalPlaneTo2D(data[i:], smooth_2d=False)
            v1 = np.array([1,slope])
            v2 = np.array([1,a])
            u1 = v1 / np.linalg.norm(v1)
            u2 = v2 / np.linalg.norm(v2)
            if np.array_equal(u1, u2):
                angle = 0.0
            else:
                angle = np.arccos(np.dot(u1, u2)) * 180 / 3.1415926
            assert angle >= 0 and angle <= 180, f"arccos range: [0,pi]), angle {angle}"
            box[i].append(angle)

    fig, ax = plt.subplots()
    ax.boxplot(box.values(), sym='')
    ax.set_xticklabels(box.keys())
    ax.set_xlabel("Remove pre-N point")
    ax.set_ylabel("Plane Angle Diff(degree)")
    ax.set_title(f"When remove pre-N points, the diff angle between two planes")

    plt.show()
    """

    ### Foreach N points, draw it's plane angle
    N = 10
    ITER = 6
    # boxplot
    box = {}

    for idx, data in dataset.whole_3d().items():
        if data.shape[0] < N*ITER:
            continue
        _, _, slope, intercept = FitVerticalPlaneTo2D(data, smooth_2d=False)
        for i in range(ITER):
            _, _, a, b = FitVerticalPlaneTo2D(data[i*N:(i+1)*N], smooth_2d=False)
            v1 = np.array([1,slope])
            v2 = np.array([1,a])
            u1 = v1 / np.linalg.norm(v1)
            u2 = v2 / np.linalg.norm(v2)
            if np.array_equal(u1, u2):
                angle = 0.0
            else:
                angle = np.arccos(np.dot(u1, u2)) * 180 / 3.1415926
            assert angle >= 0 and angle <= 180, f"arccos range: [0,pi]), angle {angle}"
            if i not in box.keys():
                box[i] = []
            box[i].append(angle)

    fig, ax = plt.subplots()
    ax.boxplot(box.values(), sym='')
    ax.set_xticklabels([f"{s*N+1}~{(s+1)*N}" for s in box.keys()])
    ax.set_xlabel("N-th point")
    ax.set_ylabel("Plane Angle Diff(degree)")
    ax.set_title(f"The diff angle between two planes")

    plt.show()
