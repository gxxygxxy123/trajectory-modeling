import torch
import pandas as pd
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
from datetime import datetime
import pickle

from blstm import Blstm

from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from dataloader import RNNDataSet, PhysicsDataSet
from utils import predict
from utils.error_function import space_err, time_err, space_time_err, time_after_err
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from point import Point, load_points_from_csv, save_points_to_csv

parser = argparse.ArgumentParser(description="BLSTM Testing Program")
parser.add_argument("-t","--time", type=float, help="Input Sequence Time (default 0.2)", default=0.2)
parser.add_argument("-w","--weight", type=str, help="BLSTM Weight", required=True)
parser.add_argument("--folder", type=str, help="Test Folder", required=True)
parser.add_argument("--fps", type=float, required=True, help="Trajectories FPS")
parser.add_argument('--draw_predict', action="store_true", help = 'Draw predict dataset')
parser.add_argument('--no_show', action="store_true", help = 'No plt.show()')
args = parser.parse_args()

# Argument
N = math.floor(args.time*args.fps)+1
TEST_DATASET = args.folder

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset
test_dataset = RNNDataSet(dataset_path=TEST_DATASET, fps=args.fps, smooth_2d=True, smooth_3d=False)
#test_dataset = PhysicsDataSet(out_max_time=10.0, cut_under_ground=True, alpha=0.242, random='experiment', model='BLSTM')

trajectories_2d = test_dataset.whole_2d()
trajectories_3d = test_dataset.whole_3d()

# Evaluation
blstm_space_2d = []
blstm_space_time_2d = []
blstm_space_3d = []
blstm_space_time_3d = []
blstm_time_2d = []
blstm_time_after_2d = []
cost_time = []

weight_dict = torch.load(args.weight)

HIDDEN_SIZE = weight_dict['hidden_size']
N_LAYERS = weight_dict['hidden_layer']
mean = weight_dict['mean']
std = weight_dict['std']

model = Blstm(hidden_size=HIDDEN_SIZE, hidden_layer=N_LAYERS,device=device).to(device)
model.load_state_dict(torch.load(args.weight)['state_dict'])
model.eval()

# Figure
fig, ax = plt.subplots()
ax.set_title("BLSTM")

fig2, ax2 = plt.subplots()
ax2.set_title(f"[BLSTM] Time After Error(2D) FPS:{args.fps}, t: {args.time}")
ax2.set_xlabel("Time(s)")
ax2.set_ylabel("Error(m)")

with torch.no_grad():
    for idx, traj_2d in trajectories_2d.items():
        if traj_2d.shape[0] <= N:
            continue

        inp = traj_2d[:N].copy()
        gt = traj_2d[N:].copy()

        t1 = datetime.now().timestamp()
        out = predict.predict2d_BLSTM(inp, model, mean=mean, std=std, out_time=2.0, fps=args.fps, touch_ground_stop=True, device=device)
        cost_time.append(datetime.now().timestamp()-t1)

        if args.draw_predict:
            p = ax.plot(inp[:,0], inp[:,1], marker='o', markersize=1)
            ax.plot(gt[:,0], gt[:,1], color=p[0].get_color(), linestyle='--')
            ax.plot(out[inp.shape[0]:,0], out[inp.shape[0]:,1], marker='o', markersize=1, alpha=0.3, color=p[0].get_color())

        blstm_space_2d.append(space_err(gt, out[inp.shape[0]:]))
        blstm_space_time_2d.append(space_time_err(gt, out[inp.shape[0]:]))
        blstm_time_2d.append(time_err(gt, out[inp.shape[0]:]))

        tmp = []
        for t in np.arange(0, 2, 0.02):
            tmp.append(time_after_err(traj_2d, out, t))
        blstm_time_after_2d.append(tmp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        blstm_time_after_2d = np.nanmean(np.array(blstm_time_after_2d), axis=0)

    # for idx, traj_3d in trajectories_3d.items():
    #     if traj_3d.shape[0] <= N:
    #         continue

    #     inp = traj_3d[:N].copy()
    #     gt = traj_3d[N:].copy()

    #     out = predict.predict3d(inp, model, 'BLSTM', mean, std, out_time=3.0, fps=args.fps, touch_ground_stop=True, device=device)

    #     blstm_space_3d.append(space_err(gt, out[inp.shape[0]:]))
    #     blstm_space_time_3d.append(space_time_err(gt, out[inp.shape[0]:]))


print(f"===BLSTM===")
print(f"Weight: {args.weight}")
print(f"Space Error 2D (Average): {np.nanmean(blstm_space_2d):.3f}m, std:{np.std(blstm_space_2d):.3f}")
print(f"Space Time Error 2D (Average): {np.nanmean(blstm_space_time_2d):.3f}m, std:{np.std(blstm_space_time_2d):.3f}")
# print(f"Space Error 3D (Average): {np.nanmean(blstm_space_3d):.3f}m")
# print(f"Space Time Error 3D (Average): {np.nanmean(blstm_space_time_3d):.3f}m")
#print(f"Time Error (Average): {np.nanmean(blstm_time_2d):.4f}s")
print(f"Cost Time (Average): {sum(cost_time)/len(cost_time):.4f}s")
print(f"Whole Data: {len(test_dataset.whole_2d())}, Used Data: {len(blstm_space_time_2d)}")
ax2.plot(np.arange(0, 2, 0.02), blstm_time_after_2d)

if not args.no_show:
    plt.show()

with open(f"{os.path.basename(args.weight)}.pkl", "wb") as fp:
    pickle.dump(blstm_time_after_2d, fp)

