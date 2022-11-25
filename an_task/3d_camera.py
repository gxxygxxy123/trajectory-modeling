import argparse
import os
import sys
import time
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
from physic_model import physics_predict3d_v2
from threeDprojectTo2D import FitVerticalPlaneTo2D
from dataloader import RNNDataSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '3d camera')
    #parser.add_argument('--by', type=str, required=True, help = 'point/trajectory')
    args = parser.parse_args()
    #assert args.by == 'point' or args.by == 'trajectory', "args.by wrong"

    smooth_dataset = RNNDataSet(dataset_path='../trajectories_dataset/vicon/', fps=120, smooth_2d=True, csvfile='Model3D.csv')
    vicon_dataset = RNNDataSet(dataset_path='../trajectories_dataset/vicon/', fps=300, smooth_2d=False, poly=-1, csvfile='vicon.csv')

    nsm_sterror = []
    sm_sterror = []

    fig3d = plt.figure()
    ax3d = fig3d.gca(projection='3d')
    p1 = (-0.691,1.432)
    p2 = (0.564,1.434)
    p3 = (0.642,-1.476)
    p4 = (-0.66,-1.478)
    ax3d.plot([p1[0],p1[0]+0.15], [p1[1],p1[1]], 0,  color='white')
    ax3d.plot([p2[0],p2[0]-0.15], [p2[1],p2[1]], 0,  color='white')
    ax3d.plot([p3[0],p3[0]-0.15], [p3[1],p3[1]], 0,  color='white')
    ax3d.plot([p4[0],p4[0]+0.15], [p4[1],p4[1]], 0,  color='white')
    ax3d.plot([p1[0],p1[0]], [p1[1],p1[1]-0.15], 0,  color='white')
    ax3d.plot([p2[0],p2[0]], [p2[1],p2[1]-0.15], 0,  color='white')
    ax3d.plot([p3[0],p3[0]], [p3[1],p3[1]+0.15], 0,  color='white')
    ax3d.plot([p4[0],p4[0]], [p4[1],p4[1]+0.15], 0,  color='white')
    ax3d.plot([p4[0],p4[0]], [p4[1],p4[1]+0.15], 0,  color='white')
    ax3d.plot([-0.16,0.16], [0,0], 0.02, linewidth=3.0, color='black')
    ax3d.plot([0,0], [0,0.5], 0.02, linewidth=3.0, color='black')

    for idx, tra_v in vicon_dataset.whole_3d().items():

        # Our Trajectories
        tra_nsm3d = smooth_dataset.whole_3d()[idx].copy()

        # Remove extra Vicon trajectory & Reset ts
        start = np.argmin(np.linalg.norm(tra_v[:,[0,1,2]]-tra_nsm3d[0,[0,1,2]],axis=1))
        end = np.argmin(np.linalg.norm(tra_v[:,[0,1,2]]-tra_nsm3d[-1,[0,1,2]],axis=1))
        tra_v = tra_v[start:end+1]
        tra_v[:,3] -= tra_v[0,3]

        # Restart Our Trajectory
        start = np.argmin(np.linalg.norm(tra_nsm3d[:,[0,1,2]]-tra_v[0,[0,1,2]],axis=1))
        tra_nsm3d = tra_nsm3d[start:]
        tra_nsm3d[:,3] -= tra_nsm3d[0,3]

        _,tra_sm3d,_,_ = FitVerticalPlaneTo2D(tra_nsm3d)

        nsm_sterror.append(space_time_err(tra_nsm3d,tra_v))
        sm_sterror.append(space_time_err(tra_sm3d,tra_v))

        if idx == 41:
            ax3d.plot(tra_nsm3d[:,0], tra_nsm3d[:,1], tra_nsm3d[:,2], color='blue', markersize=1, label='Our Camera System (120 fps)')
            ax3d.plot(tra_v[:,0], tra_v[:,1], tra_v[:,2], color='red', markersize=1, label='VICON System (300 fps)')

    print(f"Non-smooth mean:{np.mean(nsm_sterror)}, std:{np.std(nsm_sterror)}")
    print(f"Smooth mean:{np.mean(sm_sterror)}, std:{np.std(sm_sterror)}")

    fig, ax = plt.subplots()
    ax.boxplot([nsm_sterror, sm_sterror])
    ax.set_xticklabels(['Non-smooth', 'Smooth'])
    ax.set_ylabel("Error(m)")
    ax.set_title(f"Spatial-temporal Error between VICON (300 fps) and Our System (120 fps)")

    fig3d.set_facecolor('gray')
    ax3d.set_facecolor('gray')
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_xticks(np.arange(-2, 2.01, 0.5))
    ax3d.set_yticks(np.arange(-2, 2.01, 0.5))
    ax3d.set_zticks(np.arange(0, 4.01, 0.5))
    ax3d.grid(False)
    ax3d.w_xaxis.pane.fill = False
    ax3d.w_yaxis.pane.fill = False
    ax3d.w_zaxis.pane.fill = False
    ax3d.legend()

    plt.show()
