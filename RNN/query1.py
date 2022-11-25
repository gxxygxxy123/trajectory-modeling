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
    parser = argparse.ArgumentParser(description="Query Model 1 Testing Program")
    parser.add_argument("--folder", type=str, help="Test Folder", required=True)
    parser.add_argument("--fps", type=float, required=True, help="Trajectories FPS")
    parser.add_argument("-t","--time", type=float, required=True, help="Input Time")
    parser.add_argument('--draw_predict', action="store_true", help = 'Draw predict dataset')
    parser.add_argument('--no_show', action="store_true", help = 'No plt.show()')
    parser.add_argument('--physics', action="store_true", help = 'Use physics data as db')
    parser.add_argument('--alpha', type=float, default=0.2151959552)
    args = parser.parse_args()
    # Evaluation
    query_space_2d = []
    query_space_time_2d = []
    query_space_3d = []
    query_space_time_3d = []
    query_time_2d = []
    query_time_after_2d = []
    cost_time = []
    # Figure
    fig, ax = plt.subplots()
    ax.set_title(f"Query(Test)")
    fig2, ax2 = plt.subplots()
    ax2.set_title(f"[Query1] Time After Error(2D) FPS:{args.fps}, t: {args.time}")
    ax2.set_xlabel("Time(s)")
    ax2.set_ylabel("Error(m)")

    N = math.floor(args.fps*args.time)+1
    print(f"N: {N}")
    d = RNNDataSet(dataset_path=args.folder, fps=args.fps)
    # Overfitting experiment
    # d = PhysicsDataSet(out_max_time=10.0, cut_under_ground=True, alpha=args.alpha, random='experiment', model='BLSTM')

    date_idx = []
    key = []
    value = []

    if args.physics:
        physics_d = PhysicsDataSet(in_max_time=args.time, out_max_time=1/args.fps, drop_mode=0, elevation_range = (-89.0,89.0),
                                speed_range = (1.0,400.0), alpha=args.alpha, random='arange')
        for idx, traj_2d in physics_d.whole_2d().items():
            traj_2dxy = np.delete(traj_2d, 2, axis=1) # remove time
            assert traj_2dxy.shape[0] == N+1
            date_idx.append(f"p{idx}")
            key.append((traj_2dxy[:N]-traj_2dxy[0]).flatten())
            value.append((traj_2dxy[N]-traj_2dxy[0]))
        print("Database built done.")
    else:
        for idx, traj_2d in d.whole_2d().items():
            traj_2dxy = np.delete(traj_2d, 2, axis=1) # remove time
            for i in range(traj_2dxy.shape[0]-N):
                date_idx.append(idx)
                key.append((traj_2dxy[i:i+N]-traj_2dxy[i]).flatten())
                value.append((traj_2dxy[i+N]-traj_2dxy[i]))

    key = np.stack(key,axis=0)
    value = np.stack(value,axis=0)

    for idx,traj_2d in d.whole_2d().items():
        if traj_2d.shape[0] <= N:
            continue
        mask = [True if i != idx else False for i in date_idx]
        tmp_k = key[mask]
        tmp_v = value[mask]
        tmp_d = [i for i in date_idx if i != idx]
        assert len(tmp_d) == len(tmp_k) and len(tmp_d) == len(tmp_v)
        result = traj_2d[0:N,[0,1]].copy()
        iter = 0 #debug
        t1 = datetime.now().timestamp()
        query = False

        while True:
            features = (result[-N:] - result[-N]).flatten()
            if np.linalg.norm(features-tmp_k, axis=1).min() > 0.05*(N**0.5): # less than 0.05m
                break
            pred = tmp_v[np.linalg.norm(features-tmp_k, axis=1).argmin()] + result[-N]
            # Touch ground
            if pred[1] < 0:
                query = True
                break
            else:
                result = np.vstack((result,pred))

            iter+=1
            if iter>1000:
                print(f"{idx} ... iter over {iter}")
                sys.exit(0) 
        if not query:
            continue
        pad_t = np.expand_dims(np.arange(0,result.shape[0]) * (1/args.fps) + traj_2d[0,2], axis=1)
        result = np.concatenate((result, pad_t), axis=1) # Add time, same as data
        cost_time.append(datetime.now().timestamp()-t1)
        if args.draw_predict: # or idx == 202205091158005
            p = ax.plot(traj_2d[:N,0], traj_2d[:N,1], marker='o', markersize=1)
            ax.plot(traj_2d[N:,0], traj_2d[N:,1], color=p[0].get_color(), linestyle='--')
            ax.plot(result[N:,0], result[N:,1], marker='o', markersize=1, alpha=0.3, color=p[0].get_color())
        # if space_time_err(traj_2d[N:], result[N:]) > 0.85:
        #     print(f"{idx}, error {space_time_err(traj_2d[N:], result[N:])}")
        query_space_2d.append(space_err(traj_2d[N:], result[N:]))
        query_space_time_2d.append(space_time_err(traj_2d[N:], result[N:]))
        query_time_2d.append(time_err(traj_2d[N:], result[N:]))
        tmp = []
        for t in np.arange(0, 2, 0.02):
            tmp.append(time_after_err(traj_2d, result, t))
        query_time_after_2d.append(tmp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        query_time_after_2d = np.nanmean(np.array(query_time_after_2d), axis=0)
    ax2.plot(np.arange(0, 2, 0.02), query_time_after_2d)


    print(f"===Query===")
    print(f"Space Error 2D (Average): {np.mean(query_space_2d):.3f}m, max:{np.max(query_space_2d):.3f}, std:{np.std(query_space_2d):.3f}")
    print(f"Space Time Error 2D (Average): {np.mean(query_space_time_2d):.3f}m, , max:{np.max(query_space_time_2d):.3f}, std:{np.std(query_space_time_2d):.3f}")
    #print(f"Space Error 3D (Average): {np.mean(query_space_3d):.3f}m")
    #print(f"Space Time Error 3D (Average): {np.mean(query_space_time_3d):.3f}m")
    #print(f"Time Error 2D (Average): {np.mean(query_time_2d):.4f}s")
    print(f"Cost Time (Average): {sum(cost_time)/len(cost_time):.4f}s")
    print(f"Whole Data: {len(d.whole_2d())}, Used Data: {len(query_space_time_2d)}")


    if args.physics:
        with open(f"query1_{args.time}_{args.alpha:.3f}.pkl", "wb") as fp:
            pickle.dump(query_time_after_2d, fp)
    else:
        with open(f"query1_{args.time}.pkl", "wb") as fp:
            pickle.dump(query_time_after_2d, fp)

    if not args.no_show:
        plt.show()