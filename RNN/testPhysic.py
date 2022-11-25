import torch
import pandas as pd
import os
import sys
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import argparse
import warnings
from datetime import datetime
import pickle

from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from dataloader import RNNDataSet, PhysicsDataSet
from physic_model import physics_predict3d, physics_predict3d_v2
from utils.velocity import v_of_2dtraj
from utils.error_function import space_err, time_err, space_time_err, time_after_err
from sklearn import linear_model

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from point import Point, load_points_from_csv, save_points_to_csv


np.seterr(all="ignore")

parser = argparse.ArgumentParser(description="Physical Model Testing Program")
parser.add_argument("--folder", type=str, help="Test Folder", required=True)
parser.add_argument("--fps", type=float, required=True, help="Trajectories FPS")
parser.add_argument("-t","--time", type=float, required=True, help="Input Time")
parser.add_argument("--ang_time", type=float, default=0.05, help="Input Time to calculate pitch angle (default 0.05)")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--V_2point', action="store_true", help = 'V_2point')
group.add_argument('--tangent_xt', action="store_true", help = 'V by tangent in x-t figure')
parser.add_argument('--draw_predict', action="store_true", help = 'Draw predict dataset')
parser.add_argument('--no_show', action="store_true", help = 'No plt.show()')
parser.add_argument('--alpha', type=float, default=0.2151959552)
args = parser.parse_args()


assert args.time >= args.ang_time

# Argument
TEST_DATASET = args.folder

# N-M:N
N = math.floor(args.fps*args.time)+1
M = N-math.ceil(args.fps*args.time-args.fps*args.ang_time) # avoid round-off error

print(f"M: {M}, N: {N}")

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset
test_dataset = RNNDataSet(dataset_path=TEST_DATASET, fps=args.fps, smooth_2d=True, smooth_3d=False)
# Overfitting experiment
# test_dataset = PhysicsDataSet(out_max_time=10.0, cut_under_ground=True, alpha=args.alpha, random='experiment', model='BLSTM')


trajectories_2d = test_dataset.whole_2d()
trajectories_3d = test_dataset.whole_3d()


# Evaluation
physics_space_2d = []
physics_space_time_2d = []
physics_space_3d = []
physics_space_time_3d = []
physics_time_2d = []
physics_time_after_2d = []
cost_time = []

# Figure
fig, ax = plt.subplots()
ax.set_title(f"Physic(Test)")

fig2, ax2 = plt.subplots()
ax2.set_title(f"[Physics] Time After Error(2D) FPS:{args.fps}, t: {args.time}")
ax2.set_xlabel("Time(s)")
ax2.set_ylabel("Error(m)")

for idx, traj_2d in trajectories_2d.items():
    if traj_2d.shape[0] <= N:
        continue

    traj_3d2d = np.insert(traj_2d,1,0,axis=1)

    # Old
    # a = linear_model.LinearRegression().fit(traj_3d2d[M:N,0].reshape(-1,1),traj_3d2d[M:N,2]).coef_[0]
    # vxyz = np.array([1,0,a]) / np.linalg.norm(np.array([1,0,a])) * np.linalg.norm(traj_3d2d[M+1,:3]-traj_3d2d[M,:3])/(traj_3d2d[M+1,3]-traj_3d2d[M,3])
    t1 = datetime.now().timestamp()

    vxy, _, _ = v_of_2dtraj(traj_2d[:], speed_t = args.time-args.ang_time, V_2point=args.V_2point, tangent_xt=args.tangent_xt)
    vxyz = np.insert(vxy, 1, 0, axis=0)
    output_3d = physics_predict3d_v2(traj_3d2d[N-M], v=vxyz, fps=args.fps, alpha=args.alpha)

    cost_time.append(datetime.now().timestamp()-t1)

    if args.draw_predict:
        p = ax.plot(traj_2d[:N,0], traj_2d[:N,1], marker='o', markersize=1)
        ax.plot(traj_2d[N:,0], traj_2d[N:,1], color=p[0].get_color(), linestyle='--')
        ax.plot(output_3d[M:,0], output_3d[M:,2], marker='o', markersize=1, alpha=0.3, color=p[0].get_color())

    physics_space_2d.append(space_err(traj_3d2d[N:], output_3d[M:]))
    physics_space_time_2d.append(space_time_err(traj_3d2d[N:], output_3d[M:]))

    physics_time_2d.append(time_err(traj_3d2d[N:], output_3d[M:]))

    tmp = []
    #print(f"{traj_3d2d[N:].shape}, { output_3d[N-M:].shape}")
    for t in np.arange(N/args.fps, 2, 0.02):
        tmp.append(time_after_err(traj_3d2d[N:], output_3d[M:], t))
    physics_time_after_2d.append(tmp)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    physics_time_after_2d = np.nanmean(np.array(physics_time_after_2d), axis=0)

# for idx, trajectory in trajectories_3d.items():
#     output_3d = physics_predict3d(trajectory[0,:], trajectory[1,:])

#     # p = ax3.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2],marker='o',markersize=2)
#     # ax3.plot(output_3d[:,0],output_3d[:,1],output_3d[:,2],marker='o',markersize=2,alpha=0.3,color=p[0].get_color(), linestyle='--')


#     physics_space_3d.append(space_err(trajectory[2:], output_3d[2:]))
#     physics_space_time_3d.append(space_time_err(trajectory[2:], output_3d[2:]))



print(f"===Physics===")
print(f"Space Error 2D (Average): {np.mean(physics_space_2d):.3f}m, std:{np.std(physics_space_2d):.3f}")
print(f"Space Time Error 2D (Average): {np.mean(physics_space_time_2d):.3f}m, std:{np.std(physics_space_time_2d):.3f}")
#print(f"Space Error 3D (Average): {np.mean(physics_space_3d):.3f}m")
#print(f"Space Time Error 3D (Average): {np.mean(physics_space_time_3d):.3f}m")
#print(f"Time Error 2D (Average): {np.mean(physics_time_2d):.4f}s")
print(f"Cost Time (Average): {sum(cost_time)/len(cost_time):.4f}s")
print(f"Whole Data: {len(test_dataset.whole_2d())}, Used Data: {len(physics_space_time_2d)}")
ax2.plot(np.arange(N/args.fps, 2, 0.02), physics_time_after_2d)
if not args.no_show:
    plt.show()
if args.V_2point:
    with open(f"Physics_{args.time}_V_2point_{args.alpha:.3f}.pkl", "wb") as fp:
        pickle.dump(physics_time_after_2d, fp)
# else:
#     with open(f"Physics_{args.time}.pkl", "wb") as fp:
#         pickle.dump(physics_time_after_2d, fp)