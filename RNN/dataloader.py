import torch.nn as nn
import torch
import pandas as pd
import os
import random
import numpy as np
import seaborn as sns
import math
import time
import random
import sys
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from scipy.integrate import solve_ivp

from threeDprojectTo2D import fit_3d, fit_2d, FitVerticalPlaneTo2D
from physic_model import bm_ball

sys.path.append(f"../lib")
from point import Point, load_points_from_csv, save_points_to_csv, np2Point



def points_change_fps(points_list: list, fps):
    points = points_list.copy()
    new_points = []
    for i in range(len(points)-1):
        assert points[i].timestamp < points[i+1].timestamp, "Points ts isn't sorted"
    init_ts = points[0].timestamp
    for i in range(len(points)):
        assert points[i].visibility == 1, "Points Vis != 1."
        points[i].timestamp -= init_ts

    ts = 0.0
    fid = 0

    for i in range(len(points)-1):
        while points[i].timestamp <= ts and points[i+1].timestamp >= ts:
            x = (points[i].x * (points[i+1].timestamp - ts) + points[i+1].x * (ts-points[i].timestamp)) / (points[i+1].timestamp - points[i].timestamp)
            y = (points[i].y * (points[i+1].timestamp - ts) + points[i+1].y * (ts-points[i].timestamp)) / (points[i+1].timestamp - points[i].timestamp)
            z = (points[i].z * (points[i+1].timestamp - ts) + points[i+1].z * (ts-points[i].timestamp)) / (points[i+1].timestamp - points[i].timestamp)
            v = 1
            new_points.append(Point(fid=fid, timestamp=ts, visibility=v, x=x, y=y, z=z))
            fid += 1
            ts += 1/fps

    return new_points


class RNNDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, fps=None, smooth_2d=True, smooth_3d=False, poly=7, smooth_2d_x_accel=True, csvfile='Model3D.csv'):

        super(RNNDataSet).__init__()
        self.trajectories_2d = {}
        self.trajectories_3d = {}
        self.trajectories_3d2d = {}

        self.dataset_fps = fps
        self.dt = []

        self.max_traj_len = 0

        print(f"RNNDataset poly: {poly}")

        for idx in sorted(os.listdir(dataset_path)):
            if not os.path.isdir(os.path.join(dataset_path,idx)):
                continue
            csv_file = os.path.join(dataset_path,idx, csvfile)
            if not os.path.exists(csv_file):
                continue

            # Load Data From csv
            points = load_points_from_csv(csv_file)

            # Calculate the dataset FPS
            if self.dataset_fps == None:
                for i in range(len(points)-1):
                    self.dt.append(points[i+1].timestamp-points[i].timestamp)

            # Remove unvisible points
            points = [p for p in points if p.visibility == 1]

            # 3D Trajectory Timestamp reset to zero
            t0 = points[0].timestamp
            for p in points:
                p.timestamp -= t0

            # Change ts to fixed fps
            if self.dataset_fps != None:
                points = points_change_fps(points, self.dataset_fps)

            # Convert Data to numpy array

            one_trajectory = np.stack([p.toXYZT() for p in points if p.visibility == 1], axis=0)

            if smooth_3d:
                one_trajectory[:,0], one_trajectory[:,1], one_trajectory[:,2],_,_,_ = fit_3d(one_trajectory[:,0], one_trajectory[:,1], one_trajectory[:,2], N=one_trajectory.shape[0], deg=4)
                save_points_to_csv([np2Point(p,fid=fid) for fid, p in enumerate(one_trajectory)], csv_file=os.path.join(dataset_path,idx,'smooth_3d.csv'))

            self.trajectories_3d[int(idx)] = one_trajectory.copy()

            curve_2d, curve_3d2d, slope, intercept = FitVerticalPlaneTo2D(one_trajectory, smooth_2d=smooth_2d, poly=poly, smooth_2d_x_accel=smooth_2d_x_accel)
            #curve_2d = [M,3]
            #curve_3d2d = [M,4]

            if smooth_2d:
                save_points_to_csv([np2Point(p,fid=fid) for fid, p in enumerate(curve_3d2d)], csv_file=os.path.join(dataset_path,idx,'smooth_3d2d.csv'))
            else:
                save_points_to_csv([np2Point(p,fid=fid) for fid, p in enumerate(curve_3d2d)], csv_file=os.path.join(dataset_path,idx,'nosmooth_3d2d.csv'))

            assert curve_2d[0,2] == 0.0 and curve_3d2d[0,3] == 0.0

            self.trajectories_2d[int(idx)] = curve_2d.copy()
            self.trajectories_3d2d[int(idx)] = curve_3d2d.copy()

            self.max_traj_len = max(self.max_traj_len, one_trajectory.shape[0])

        if self.dataset_fps == None:
            self.dataset_fps = 1/(sum(self.dt)/len(self.dt))

    def whole_2d(self):
        # {1:t1, 2:t2, 3:t3, 4:t4} ...
        # t1,t2,t3,t4 : unequal length
        # t1 : np array, shape: [?, 3]  3 represents (XY, Z, t)
        return self.trajectories_2d

    def whole_3d(self):
        # {1:t1, 2:t2, 3:t3, 4:t4} ...
        # t1,t2,t3,t4 : unequal length
        # t1 : np array, shape: [?, 4]  4 represents (X, Y, Z, t)
        return self.trajectories_3d

    def whole_3d2d(self):
        # {1:t1, 2:t2, 3:t3, 4:t4} ...
        # t1,t2,t3,t4 : unequal length
        # t1 : np array, shape: [?, 4]  4 represents (X, Y, Z, t) on the projected plane
        return self.trajectories_3d2d

    def max_len(self):
        return self.max_traj_len

    def fps(self):
        return self.dataset_fps

    def show_each(self):
        for idx, data in self.whole_2d().items():
            # if idx != 202205241256004: for making ppt
            #     continue
            plt.plot(data[:,0],data[:,1],marker='o',markersize=2)
            plt.xlabel("Distance(m)")
            plt.ylabel("Height(m)")
            plt.title(f"idx: {idx}")
            plt.show()


class PhysicsDataSet(torch.utils.data.Dataset):
    def __init__(self, datas=0,
                       in_max_time=0.1,
                       out_max_time=2.0,
                       cut_under_ground=False,
                       noise_t=False,
                       noise_xy=False,
                       dxyt=True,
                       network_in_dim=2, # 2: x,y 3: x,y,t
                       drop_mode=0, # 0: fixed, 1: unequal length but continue, 2: random drop
                       fps_range = (120.0,120.0),
                       elevation_range = (-80.0,80.0), 
                       speed_range = (10.0,240.0), # km/hr
                       output_fps_range = (120.0,120.0),
                       starting_point=[0, 0, 2],
                       model='TF',
                       alpha=0.2151959552,
                       g=9.81,
                       random='uniform',
                       fps_step = 1.0,
                       e_step = 0.5,
                       s_step = 0.5
                       ):

        self.data = {}

        data_src = []
        data_trg = []

        self.model = model

        if random == 'uniform':
            random_datas = np.random.uniform(low =[fps_range[0], elevation_range[0], speed_range[0], output_fps_range[0]],
                                                    high=[fps_range[-1],elevation_range[-1],speed_range[-1],output_fps_range[-1]],
                                                    size=(datas,4))
            # random_datas = np.concatenate((random_datas, np.array([[120.0,-15.5,200.5,120.0],[120.0,0.5,150.5,120.0],[120.0,45.5,100.5,120.0],[120.0,10.5,40.5,120.0]])))

        elif random == 'arange':
            assert fps_range[0] == 120.0 and fps_range[1] == 120.0 and output_fps_range[0] == 120.0 and output_fps_range[1] == 120.0
            # fps step: 1
            # e,s step: 0.5
            fps_rand = np.arange(fps_range[0], fps_range[1]+fps_step, fps_step)
            e_rand = np.arange(elevation_range[0], elevation_range[1]+e_step, e_step)
            s_rand = np.arange(speed_range[0], speed_range[1]+s_step, s_step)
            output_fps_rand = np.arange(output_fps_range[0], output_fps_range[1]+fps_step, fps_step)
            random_datas = np.array(np.meshgrid(fps_rand, e_rand, s_rand, output_fps_rand)).T.reshape(-1,4)
            print(f"e step: {e_step}, s step: {s_step}")
        elif random == 'experiment':
            random_datas = [[120.0,-15.5,200.5,120.0],[120.0,0.5,150.5,120.0],[120.0,45.5,100.5,120.0],[120.0,10.5,40.5,120.0]]
            random_datas = np.stack(random_datas)

        else:
            sys.exit(0)

        self.trajectories_2d = {}
        self.trajectories_3d = {}

        print(f"===Physics Dataset ({model})===\n"
              f"FPS: {fps_range[0]:.1f} ~ {fps_range[-1]:.1f}\n"
              f"Elevation: {elevation_range[0]:.2f} ~ {elevation_range[-1]:.2f} degree\n"
              f"Speed: {speed_range[0]:.1f} ~ {speed_range[-1]:.1f} km/hr\n"
              f"Output Fps: {output_fps_range[0]:.1f} ~ {output_fps_range[-1]:.1f}\n"
              f"In Max Time: {in_max_time:.4f}s\n"
              f"Out Max Time: {out_max_time:.4f}s\n"
              f"Datas: {random_datas.shape[0]}\n"
              f"========Physics Dataset Option========\n"
              f"Cut Under ground: {cut_under_ground}\n"
              f"Noise t: {noise_t}\n"
              f"Noise xy: {noise_xy}\n"
              f"dxyt: {dxyt}\n"
              f"network input dim: {network_in_dim}\n"
              f"drop point mode: {drop_mode}\n"
              f"starting point: {starting_point}\n"
              f"alpha: {alpha}\n"
              f"g: {g}\n"
              f"random : {random}\n")

        idx = 1
        for fps,e,s,output_fps in random_datas:
            if drop_mode == 2:
                in_t = np.sort(np.random.choice(np.arange(0,in_max_time*fps)*(1/fps),
                        size=round(random.uniform(2,in_max_time*fps)), # at least 1 vector
                        replace=False))
            elif drop_mode == 1:
                in_t = np.arange(0,round(random.uniform(2,in_max_time*fps)))*(1/fps)
            elif drop_mode == 0:
                in_t = np.arange(0, math.floor(in_max_time*fps)+1)*(1/fps)

            assert drop_mode == 0, "[TODO], loss point pad 0. [2022.07.19]"
            # assert dxyt == True

            out_t = np.arange(0,math.floor(out_max_time*output_fps))*(1/output_fps) + in_t[-1] + (1/output_fps)

            teval = np.concatenate((in_t, out_t))

            # Add noise to dt
            if noise_t:
                teval += np.random.normal(0,0.02/fps, teval.shape) # 1/fps / 50

            # reset time to zero
            teval = teval - teval[0]

            s = s * 1000/3600 # km/hr -> m/s
            initial_velocity = [s * math.cos(e/180*math.pi), 0, s * math.sin(e/180*math.pi)]
            traj = solve_ivp(lambda t, y: bm_ball(t, y, alpha=alpha, g=g), [0, teval[-1]], starting_point + initial_velocity, t_eval = teval) # traj.t traj.y
            xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
            t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)

            trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

            assert len(in_t)+len(out_t) == trajectories.shape[0]

            # # Cut under ground part
            if cut_under_ground:
                while(trajectories[-1][2] <= 0):
                    trajectories = trajectories[:-1] # pop last point
                if trajectories.shape[0] <= len(in_t):
                    continue

            # Add noise to dx,dy (except starting point)
            if noise_xy:
                trajectories[1:,[0,2]] += np.random.normal(0,0.01, trajectories[1:,[0,2]].shape) # 1cm

            self.trajectories_2d[int(idx)] = trajectories[:,[0,2,3]].copy()
            self.trajectories_3d[int(idx)] = trajectories[:,[0,1,2,3]].copy()

            if network_in_dim == 2:
                in_dim = np.array([0,2]) # x,y (in 2d)
            elif network_in_dim == 3:
                in_dim = np.array([0,2,3]) # x,y,t (in 2d)

            # Input (dx,dy)/(dx,dy,dt)
            if dxyt:
                tmp = np.diff(trajectories[:,in_dim],axis=0)
                data_src.append(tmp[:len(in_t)-1]) # assert drop_mode == 0, "[TODO], loss point pad 0. [2022.07.19]"
                if model == 'TF' or model == 'Seq2Seq':
                    data_trg.append(tmp[len(in_t)-1:])
                elif model == 'BLSTM':
                    data_trg.append(tmp[len(in_t)-1:(len(in_t)-1)*2])

            # Input (x,y)/(x,y,t)
            else:
                tmp = trajectories[:,in_dim] - trajectories[0,in_dim] # move x,y,t to origin
                assert not np.any(tmp[0]), "x,y,t is not 0"
                data_src.append(tmp[:len(in_t)])
                if model == 'TF' or model == 'Seq2Seq':
                    data_trg.append(tmp[len(in_t):])
                elif model == 'BLSTM':
                    data_trg.append(tmp[len(in_t):len(in_t)*2])

            idx += 1
        
        # TODO TF train with different seq length?
        if self.model == 'TF':
            self.data['src'] = np.stack(data_src,0)
            self.data['trg'] = np.stack(data_trg,0)

        # TODO src length shape equal
        elif self.model == 'BLSTM' or self.model == 'Seq2Seq':
            data_src = sorted(data_src, key=lambda x: len(x), reverse=True)
            src_lens = [len(x) for x in data_src]
            for i in range(len(data_src)):
                data_src[i] = np.pad(data_src[i],((0,max(src_lens)-len(data_src[i])),(0,0)), 'constant', constant_values=0)

            data_trg = sorted(data_trg, key=lambda x: len(x), reverse=True)
            trg_lens = [len(x) for x in data_trg]
            for i in range(len(data_trg)):
                data_trg[i] = np.pad(data_trg[i],((0,max(trg_lens)-len(data_trg[i])),(0,0)), 'constant', constant_values=0)

            self.data['src'] = np.stack(data_src,0)
            self.data['trg'] = np.stack(data_trg,0)
            # (datas, max src/trg len, features)

            self.data['src_lens'] = src_lens
            self.data['trg_lens'] = trg_lens

        all_datas = np.concatenate((self.data['src'],self.data['trg']),axis=1)

        self._mean = np.nanmean(np.where(all_datas!=0,all_datas,np.nan),(0,1))
        self._std = np.nanstd(np.where(all_datas!=0,all_datas,np.nan),(0,1))

        self.fps_range = fps_range
        self.output_fps_range = output_fps_range

    def __len__(self):
        if self.model == 'TF':
            return self.data['src'].shape[0]
        elif self.model == 'BLSTM' or self.model == 'Seq2Seq':
            return len(self.data['src'])

    def __getitem__(self, index):
        if self.model == 'TF':
            return {'src':torch.Tensor(self.data['src'][index]), 'trg':torch.Tensor(self.data['trg'][index])}
        elif self.model == 'BLSTM' or self.model == 'Seq2Seq':
            return {'src':torch.Tensor(self.data['src'][index]), 'trg':torch.Tensor(self.data['trg'][index]),
                    'src_lens':self.data['src_lens'][index], 'trg_lens':self.data['trg_lens'][index]}

    def mean(self):
        return torch.Tensor(self._mean)

    def std(self):
        return torch.Tensor(self._std)

    def whole_2d(self):
        # {1:t1, 2:t2, 3:t3, 4:t4} ...
        # t1,t2,t3,t4 : unequal length
        # t1 : np array, shape: [?, 3]  3 represents (XY, Z, t)
        return self.trajectories_2d

    def whole_3d(self):
        # {1:t1, 2:t2, 3:t3, 4:t4} ...
        # t1,t2,t3,t4 : unequal length
        # t1 : np array, shape: [?, 4]  4 represents (X, Y, Z, t)
        return self.trajectories_3d

    def fps(self):
        assert self.fps_range[0] == self.fps_range[1] and self.fps_range[0] == self.output_fps_range[0] and self.fps_range[0] == self.output_fps_range[1]
        return self.fps_range[0]