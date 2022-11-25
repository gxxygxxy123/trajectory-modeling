from datetime import datetime
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import griddata
import math
import pickle
import argparse
import warnings
import seaborn as sns
from utils.error_function import space_err, time_err, space_time_err, time_after_err
from dataloader import RNNDataSet, PhysicsDataSet
from utils.velocity import v_of_2dtraj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find the relationship between alpha and velocity and pitch angle")
    parser.add_argument("--folder", type=str, help="Test Folder", required=True)
    parser.add_argument("--fps", type=float, required=True, help="Trajectories FPS")
    parser.add_argument("-t","--time", type=float, required=True, help="Input Time")
    args = parser.parse_args()

    N = math.floor(args.fps*args.time)+1
    print(f"N: {N}")
    d = RNNDataSet(dataset_path=args.folder, fps=args.fps)

    alpha = []
    key = []

    for a in np.arange(0.05, 0.50, 0.01): # Find the relationship between alpha and velocity
        physics_d = PhysicsDataSet(in_max_time=args.time, out_max_time=0, drop_mode=0, elevation_range = (-90.0,70.0),
                                speed_range = (1.0,250.0), alpha=a, random='arange',e_step=1, s_step=2)
        for idx, traj_2d in physics_d.whole_2d().items():
            traj_2dxy = np.delete(traj_2d, 2, axis=1) # remove time
            assert traj_2dxy.shape[0] == N
            alpha.append(a)
            key.append((traj_2dxy[:N]-traj_2dxy[0]).flatten())
    print("Database built done.")

    key = np.stack(key,axis=0)


    # Find the relationship between alpha and velocity
    x_step = 4
    y_step = 5
    x = np.arange(-90,91, x_step)
    y = np.arange(0, 251, y_step)
    z = np.zeros((x.shape[0],y.shape[0]))

    V_E_alpha = [[[] for _ in range(y.shape[0])] for _ in range(x.shape[0])]

    for idx, traj_2d in d.whole_2d().items():
        for i in range(traj_2d.shape[0]-N):
            features = (traj_2d[i:i+N,[0,1]] - traj_2d[i,[0,1]]).flatten()
            if np.linalg.norm(features-key, axis=1).min() > 0.05*(N**0.5): # less than 0.05m
                continue
            _,v,e = v_of_2dtraj(traj_2d[i:i+N]-traj_2d[i],speed_t=0.0, V_2point=True, ang_t=0.02)

            v_tmp = round(v*3600/1000)
            e_tmp = round(e)
            if v_tmp >= 0 and v_tmp < 250 and e_tmp >= -90 and e_tmp < 90:
                V_E_alpha[round((e_tmp+90)/x_step)][round(v_tmp/y_step)].append(alpha[np.linalg.norm(features-key, axis=1).argmin()])

    print(f"Start Ploting...")
    plt.clf()

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if V_E_alpha[i][j]:
                z[i,j] = sum(V_E_alpha[i][j])/len(V_E_alpha[i][j])

    X,Y = np.meshgrid(x, y)

    plt.pcolormesh(X, Y, z.T)

    plt.xlabel("Angle(degree)")
    plt.ylabel("Velocity(km/hr)")

    plt.colorbar()
    plt.show()