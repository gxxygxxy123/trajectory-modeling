from datetime import datetime
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import argparse
import warnings
import seaborn as sns
from sklearn import linear_model
from utils.velocity import v_of_2dtraj

sns.set()
from utils.error_function import space_err, time_err, space_time_err, time_after_err
from dataloader import RNNDataSet, PhysicsDataSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query Model 2 Testing Program")
    parser.add_argument("--folder", type=str, help="Test Folder", required=True)
    parser.add_argument("--fps", type=float, required=True, help="Trajectories FPS")
    parser.add_argument("-t","--time", type=float, required=True, help="Input Time")
    parser.add_argument("--ang_time", type=float, default=0.05, help="Input Time to calculate pitch angle (default 0.05)")
    parser.add_argument('--draw_predict', action="store_true", help = 'Draw predict dataset')
    parser.add_argument('--no_show', action="store_true", help = 'No plt.show()')
    parser.add_argument('--interp', action="store_true", help = 'Interpolation') # Default: Nearest Neighbor. This option is interpolation under --physics (experiment, DONT USE!)
    parser.add_argument('--V_2point', action="store_true", help = 'V_2point')
    parser.add_argument('--physics', action="store_true", help = 'Use physics data as db')
    parser.add_argument('--alpha', type=float, default=0.2151959552)
    args = parser.parse_args()

    # Finally, I decide that query2 always use V_2point to calcuate velocity ~~
    args.V_2point = True
    assert args.V_2point == True 

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
    ax2.set_title(f"[Query2] Time After Error(2D) FPS:{args.fps}, t: {args.time}")
    ax2.set_xlabel("Time(s)")
    ax2.set_ylabel("Error(m)")

    fig3, ax3 = plt.subplots()
    fig3_v_ang = [(60,0,'black'),(63,0,'red'),(60,3,'blue'),(63,3,'green')]
    group = [None] * len(fig3_v_ang)
    # ax3.set_xticks(np.arange(-1,1,0.1))
    # ax3.set_yticks(np.arange(0,1,0.1))
    ax3.set_title(f"Datas of query model")
    ax3.set_xlabel("Distance(cm)")
    ax3.set_ylabel("Height(cm)")

    # N-M:N
    N = math.floor(args.fps*args.time)+1
    M = N-math.ceil(args.fps*args.time-args.fps*args.ang_time)  # avoid round-off error

    print(f"M: {M}, N: {N}")

    ### Database (Table)
    db = {}
    if args.physics:
        v_width = 1
        ang_width = 1
    else:
        v_width = 3
        ang_width = 3
    for v in np.arange(0,400,v_width): # velocity, foreach 1 km/hr
        for ang in np.arange(-90,91,ang_width): # pitch angle, foreach 1 degree
            db[(v,ang)] = {}

    d = RNNDataSet(dataset_path=args.folder, fps=args.fps)
    # Overfitting experiment
    # d = PhysicsDataSet(out_max_time=10.0, cut_under_ground=True, alpha=args.alpha, random='experiment', model='BLSTM')

    if args.physics:
        physics_d = PhysicsDataSet(in_max_time=args.ang_time, out_max_time=1/args.fps, drop_mode=0, elevation_range = (-89.0,89.0),
                                   speed_range = (1.0,400.0), alpha=args.alpha, random='arange', e_step=0.5, s_step=0.5)
        for idx, traj_2d in physics_d.whole_2d().items():
            assert traj_2d.shape[0] == M+1, f"{traj_2d.shape[0]}=={M+1}"
            _, instantaneous_velocity, ang = v_of_2dtraj(traj_2d[:M], speed_t=traj_2d[0,2], ang_t=args.ang_time, V_2point=args.V_2point)
            instantaneous_velocity *= 3600/1000

            db[(round(instantaneous_velocity/v_width)*v_width,round(ang/ang_width)*ang_width)][f"p{idx}_{0}"] = traj_2d[M,[0,1]]-traj_2d[M-1,[0,1]]

        print("Database built done.")
    else:
        for idx, traj_2d in d.whole_2d().items():

            for i in range(traj_2d.shape[0]-M):

                # Old
                # slope = linear_model.LinearRegression().fit(traj_2d[i:i+M,0].reshape(-1,1),traj_2d[i:i+M,1]).coef_[0] # y = slope*x+b
                # instantaneous_velocity = np.linalg.norm(traj_2d[i+1,[0,1]]-traj_2d[i,[0,1]])/(traj_2d[i+1,2]-traj_2d[i,2]) * 3600/1000
                # ang = math.degrees(math.atan2(slope,1))

                _, instantaneous_velocity, ang = v_of_2dtraj(traj_2d[i:i+M], speed_t=traj_2d[i,2], ang_t=args.ang_time, V_2point=args.V_2point)
                instantaneous_velocity *= 3600/1000

                db[(round(instantaneous_velocity/v_width)*v_width,round(ang/ang_width)*ang_width)][f"{idx}_{i}"] = traj_2d[i+M,[0,1]]-traj_2d[i+M-1,[0,1]] # Using M points (i~i+M-1) to predict (i+M), move to origin

    too_long = 0

    for idx,traj_2d in d.whole_2d().items():
        if traj_2d.shape[0] <= N:
            continue
        result = traj_2d[:N,[0,1]].copy()
        # print(f"{idx} ...")
        query = False
        t1 = datetime.now().timestamp()
        idx = 0
        while True:
            # Old
            # slope = linear_model.LinearRegression().fit(result[-M:,0].reshape(-1,1),result[-M:,1]).coef_[0] # y = slope*x+b
            # instantaneous_velocity = np.linalg.norm(result[-M+1,[0,1]]-result[-M,[0,1]]) * args.fps * 3600/1000
            # ang = math.degrees(math.atan2(slope,1))

            _, instantaneous_velocity, ang = v_of_2dtraj(np.concatenate((result[-M:], np.expand_dims(np.arange(M)*(1/args.fps), axis=1)), axis=1), speed_t=0.0 , V_2point=args.V_2point)
            instantaneous_velocity *= 3600/1000

            if (round(instantaneous_velocity/v_width)*v_width,round(ang/ang_width)*ang_width) not in db.keys():
                print(f"({round(instantaneous_velocity/v_width)*v_width},{round(ang/ang_width)*ang_width}) not in db.keys(), {result.shape}")
                break

            pred = []
            # Nearest velocity & angle
            if not args.interp:
                for data_idx, data in db[(round(instantaneous_velocity/v_width)*v_width,round(ang/ang_width)*ang_width)].items():
                    if data_idx.startswith(f"{idx}_"):
                        continue
                    pred.append(data)
            else:
                q0 = (instantaneous_velocity, ang)

                Q = [(math.floor(instantaneous_velocity/v_width)*v_width, math.floor(ang/ang_width)*ang_width),
                     (math.ceil(instantaneous_velocity/v_width)*v_width, math.floor(ang/ang_width)*ang_width),
                     (math.floor(instantaneous_velocity/v_width)*v_width, math.ceil(ang/ang_width)*ang_width),
                     (math.ceil(instantaneous_velocity/v_width)*v_width, math.ceil(ang/ang_width)*ang_width)]

                for i in range(4):
                    tmp = []
                    A = (v_width-abs(q0[0]-Q[i][0]))*(ang_width-abs(q0[1]-Q[i][1]))/(v_width*ang_width)
                    assert A >= 0
                    for data_idx, data in db[Q[i]].items():
                        if data_idx.startswith(f"{idx}_"):
                            continue
                        tmp.append(data)
                    if len(tmp) == 0:
                        print(f"Empty: {q0}")
                        continue
                    tmp = np.mean(np.stack(tmp, axis=0), axis=0)

                    pred.append(tmp*A*4) # 4 corner

            if not pred:
                # print(f"db({round(instantaneous_velocity/v_width)*v_width},{round(ang/ang_width)*ang_width}) no data")
                break
            else:
                pred = np.stack(pred,0)
                pred = np.mean(pred, axis=0) + result[-1]

                # Touch ground
                if pred[1] < 0:
                    query = True
                    break
                else:
                    result = np.vstack((result,pred))
            if idx >= args.fps*10: # more than 10 sec
                too_long += 1
                break
            idx += 1
        if not query: # Query Failed
            continue
        else: # Query Success

            pad_t = np.expand_dims(np.arange(0,result.shape[0]) * (1/args.fps) + traj_2d[0,2], axis=1)
            result = np.concatenate((result, pad_t), axis=1) # Add time, same as traj_2d

            cost_time.append(datetime.now().timestamp()-t1)

            if args.draw_predict: # or idx == 202205091158005
                p = ax.plot(traj_2d[:N,0], traj_2d[:N,1], marker='o', markersize=1)
                ax.plot(traj_2d[N:,0], traj_2d[N:,1], color=p[0].get_color(), linestyle='--')
                ax.plot(result[N:,0], result[N:,1], marker='o', markersize=1, alpha=0.3, color=p[0].get_color())

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

    # for idx,(v,ang,color) in enumerate(fig3_v_ang):
    #     points = np.stack(list(db[(v,ang)].values()))

    #     ax3.scatter(points[:,0]*100,points[:,1]*100,c=color, label=f"θ:{ang}, v:{v}")
    # ax3.set_xticks([7,9,11,13])
    # ax3.set_yticks([-3,-1,1,3])
    # plt.legend()

    # plt.show()

    print(f"===Query===")
    print(f"Space Error 2D (Average): {np.mean(query_space_2d):.3f}m, max:{np.max(query_space_2d):.3f}, std:{np.std(query_space_2d):.3f}")
    print(f"Space Time Error 2D (Average): {np.mean(query_space_time_2d):.3f}m, , max:{np.max(query_space_time_2d):.3f}, std:{np.std(query_space_time_2d):.3f}")
    #print(f"Space Error 3D (Average): {np.mean(query_space_3d):.3f}m")
    #print(f"Space Time Error 3D (Average): {np.mean(query_space_time_3d):.3f}m")
    #print(f"Time Error 2D (Average): {np.mean(query_time_2d):.4f}s")
    print(f"Cost Time (Average): {sum(cost_time)/len(cost_time):.4f}s")
    print(f"Whole Data: {len(d.whole_2d())}, Used Data: {len(query_space_time_2d)}, Too long (More than 10 sec): {too_long}")

    if args.V_2point:
        if args.physics:
            with open(f"query2_{args.time}_V_2point_{args.alpha:.3f}.pkl", "wb") as fp:
                pickle.dump(query_time_after_2d, fp)
        else:
            with open(f"query2_{args.time}_V_2point.pkl", "wb") as fp:
                pickle.dump(query_time_after_2d, fp)
    else:
        if args.physics:
            with open(f"query2_{args.time}_{args.alpha:.3f}.pkl", "wb") as fp:
                pickle.dump(query_time_after_2d, fp)
        else:
            with open(f"query2_{args.time}.pkl", "wb") as fp:
                pickle.dump(query_time_after_2d, fp)


    if not args.no_show:
        plt.show()