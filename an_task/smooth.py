import argparse
import os
import sys
import time
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import math
sns.set_style("white")

sys.path.append(f"../lib")
sys.path.append(f"../Model3D")
sys.path.append(f"../RNN")
from point import Point, load_points_from_csv, save_points_to_csv
from utils.error_function import space_err, time_err, space_time_err
from physic_model import physics_predict3d
from threeDprojectTo2D import FitVerticalPlaneTo2D
from dataloader import RNNDataSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'smooth way comparison')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    parser.add_argument('--fps', type=float, default=None, help = 'Dataset FPS')
    parser.add_argument('--by', type=str, required=True, help = 'point/trajectory')
    parser.add_argument('--each', action="store_true", help = 'Show each trajectory')
    args = parser.parse_args()
    assert args.by == 'point' or args.by == 'trajectory', "args.by wrong"

    # Dataset
    DATASET = args.folder

    # boxplot
    box = {}

    non_smooth_dataset = RNNDataSet(dataset_path=DATASET, fps=args.fps, smooth_2d=False, poly=-1)

    for poly in range(3,10):

        dis_error = []

        only_poly_dataset = RNNDataSet(dataset_path=DATASET, fps=args.fps, smooth_2d=True, poly=poly, smooth_2d_x_accel=False)
        smooth_dataset = RNNDataSet(dataset_path=DATASET, fps=args.fps, smooth_2d=True, poly=poly, smooth_2d_x_accel=True)
        
        for idx, data in only_poly_dataset.whole_2d().items():
            data2 = non_smooth_dataset.whole_2d()[idx].copy()
            assert data2.shape == data.shape, "Error"
            data -= data[0] # move to origin, reset time to zero
            data2 -= data2[0] # move to origin, reset time to zero
            trash0 = smooth_dataset.whole_2d()[idx].copy()
            trash0 -= trash0[0]
            if args.each:
                #plt.plot(data[::10,0],data[::10,1],marker='o',markersize=2, color='red', label ='Smooth (no x smooth)')
                plt.plot(data2[::10,0],data2[::10,1],marker='o',markersize=2, color='blue', label='Origin')
                plt.plot(trash0[::10,0],trash0[::10,1],marker='o',markersize=2, color='green', label='Smooth')

            tmp = np.linalg.norm(data[:,:-1]-data2[:,:-1], axis=1).tolist()

            if args.by == 'point':
                dis_error = dis_error + tmp
            elif args.by == 'trajectory':
                dis_error.append(sum(tmp)/len(tmp))

            if args.each:
                plt.title(f"{idx} Poly {poly}")
                plt.legend()
                plt.xlabel("Distance(m)")
                plt.ylabel("Height(m)")
                plt.show()

        box[f"Poly {poly}"] = np.array(dis_error) * 100 # unit: cm

        print(f"Poly {poly} mean: {np.mean(dis_error)*100:.1f} cm, std: {np.std(dis_error)*100:.1f} cm")

        # plt.title(f"Poly fit {poly}")
        # plt.show()

    fig, ax = plt.subplots()
    ax.boxplot(box.values(),sym='')
    ax.set_xticklabels(box.keys())
    ax.set_ylabel("L2 Error(cm)")
    ax.set_title(f"L2 error before and after polynomial fitting. unit: {args.by}")

    plt.show()

    ### Foreach N points, draw it's smooth influence
    max_len = non_smooth_dataset.max_len()
    point_2derror = [[] for i in range(max_len)]

    poly = 7

    only_poly_dataset = RNNDataSet(dataset_path=DATASET, fps=args.fps, smooth_2d=True, poly=poly, smooth_2d_x_accel=False)
    smooth_dataset = RNNDataSet(dataset_path=DATASET, fps=args.fps, smooth_2d=True, poly=poly, smooth_2d_x_accel=True)
    
    for idx, data in smooth_dataset.whole_2d().items():
        data2 = only_poly_dataset.whole_2d()[idx].copy()
        assert data2.shape == data.shape, f"{data2.shape} {data.shape} {idx}"
        data -= data[0] # move to origin, reset time to zero
        data2 -= data2[0] # move to origin, reset time to zero

        err = np.linalg.norm(data[:,[0,1]]-data2[:,[0,1]], axis=1)

        for i in range(err.shape[0]):
            point_2derror[i].append(err[i])


    mean_error = np.zeros(len(point_2derror))
    std_error = np.zeros(len(point_2derror))
    for i in range(len(point_2derror)):
        mean_error[i] = np.mean(point_2derror[i]) * 100
        std_error[i] = np.std(point_2derror[i]) * 100
    fig, ax = plt.subplots()

    ax.plot(np.arange(mean_error.shape[0])/args.fps, mean_error, label='L2 Error')
    ax.plot(np.arange(std_error.shape[0])/args.fps, std_error, label='Std Dev')

    ax.set_xlabel("time(s)")
    ax.set_ylabel("Error(cm)")
    ax.set_title(f"L2 error of 2D points before and after smooth")
    plt.legend()
    plt.show()

