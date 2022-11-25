import argparse
import os
import sys
import math
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model

sys.path.append(f"../lib")
sys.path.append(f"../Model3D")
sys.path.append(f"../RNN")
from point import Point, load_points_from_csv, save_points_to_csv
from utils.error_function import space_err, time_err, space_time_err
from utils.velocity import v_of_2dtraj
from threeDprojectTo2D import FitVerticalPlaneTo2D
from physic_model import physics_predict3d, physics_predict3d_v2
from dataloader import RNNDataSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Validate Physics Model and Find Good coeffs')
    parser.add_argument('--folder', type=str, required=True, help = 'Root Folder Path')
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument("--ang_time", type=float, default=0.05, help="Input Time to calculate pitch angle (default: 0.05)")
    parser.add_argument("--offset_t", type=float, default=0.0, help="Ignore offset time")
    parser.add_argument('--smooth', action="store_true", help = 'Smooth when projecting to 2d')
    parser.add_argument('--poly', type=int, default=7)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--V_2point', action="store_true", help = 'V_2point')
    group.add_argument('--tangent_xt', action="store_true", help = 'V by tangent in x-t figure')

    args = parser.parse_args()
    # Dataset
    DATASET = args.folder

    N = math.floor(args.fps*(args.offset_t+args.ang_time))+1
    M = N-math.ceil(args.fps*args.offset_t)

    # Use which points to physically predict
    if "vicon" in args.folder:
        csvfile = 'vicon.csv'
    else:
        csvfile = 'Model3D.csv'

    print(f"Input time to calculate pitch angle: {args.ang_time}(s), M: {M}, N:{N}")
    # Load Dataset
    trajectories_3d2d = []
    trajectories_3d = []
    trajectories_2d = []

    dataset = RNNDataSet(dataset_path=DATASET, fps=args.fps, poly=args.poly, smooth_2d=args.smooth, csvfile=csvfile)

    for idx, data in dataset.whole_3d().items():
        trajectories_3d.append(data.copy())
    for idx, data in dataset.whole_2d().items():
        trajectories_2d.append(data.copy())

    # points_3d2d = [Point(visibility=1,x=i[0],y=i[1],z=i[2],timestamp=i[3]) for i in tra_3d2d]
    # save_points_to_csv(points_3d2d, os.path.join(DATASET, folder, '3d2d.csv'))

    # points_physics_3d2d = [Point(visibility=1,x=i[0],y=i[1],z=i[2],timestamp=i[3]) for i in tra_physics_2d]
    # save_points_to_csv(points_physics_3d2d, os.path.join(DATASET, folder, 'physic_3d2d.csv'))


    alpha_default = 0.2152
    g_default = 9.81

    ##### Find the best coeff for the dataset #####
    # dx, dy = 0.002,0.04
    # y, x = np.mgrid[slice(g_default-0.01-0.8, g_default-0.01+0.8 + dy, dy),
    #                 slice(alpha_default-0.0002-0.015, alpha_default-0.0002+0.015 + dx, dx)]
    dx, dy = 0.001,0.02 # 0.001, 0.02
    y, x = np.mgrid[slice(g_default-0.01-2, g_default-0.01+2 + dy, dy),
                    slice(alpha_default-0.0002-0.010, alpha_default-0.0002+0.065 + dx, dx)]
    z_space_2d = np.zeros(shape=x.shape,dtype=float)
    z_space_time_2d = np.zeros(shape=x.shape,dtype=float)
    z_space_time_2d_each = np.zeros(shape=(x.shape[0],x.shape[1],len(trajectories_2d)),dtype=float)
    z_space_3d = np.zeros(shape=x.shape,dtype=float)
    z_space_time_3d = np.zeros(shape=x.shape,dtype=float)

    traj_v = [None] * len(trajectories_2d)

    print(f"({x.shape[0],x.shape[1]})")

    for i in range(x.shape[0]):
        print(i)
        for j in range(x.shape[1]):
            alpha = x[i][j]
            g = y[i][j]
            space_2d = []
            space_time_2d = []
            space_3d = []
            space_time_3d = []
            for k, traj_2d in enumerate(trajectories_2d):

                traj_3d = np.insert(traj_2d,1,0,axis=1)

                # Old
                # slope = linear_model.LinearRegression().fit(traj_3d[:N,0].reshape(-1,1),traj_3d[:N,2]).coef_[0] # y = slope*x+b
                # vxyz = np.array([1,0,slope]) / np.linalg.norm(np.array([1,0,slope])) * np.linalg.norm(traj_3d[1,:3]-traj_3d[0,:3])/(traj_3d[1,3]-traj_3d[0,3])
                vxy, speed, _, = v_of_2dtraj(traj_2d, speed_t=traj_2d[M,2], ang_t=args.ang_time, V_2point=args.V_2point, tangent_xt=args.tangent_xt)
                vxyz = np.insert(vxy, 1, 0, axis=0)

                tra_physics_2d = physics_predict3d_v2(traj_3d[M], v=vxyz, fps=args.fps, alpha=alpha,g=g)

                space_time_2d.append(space_time_err(traj_3d[N:], tra_physics_2d[N-M:]))

                # space_3d.append(space_err(trajectories_3d[k][SECOND_POINT_IDX+1:], tra_physics_2d[2:]))
                # space_time_3d.append(space_time_err(trajectories_3d[k][SECOND_POINT_IDX+1:], tra_physics_2d[2:]))

                z_space_time_2d_each[i][j][k] = space_time_err(traj_3d[N:], tra_physics_2d[N-M:])

                traj_v[k] = speed*3600/1000

            space_time_2d = np.stack(space_time_2d)
            #space_time_3d = np.stack(space_time_3d)

            # z_space_2d[i][j] = np.nanmean(space_2d)

            
            z_space_time_2d[i][j] = np.nanmean(space_time_2d)

            # z_space_3d[i][j] = np.nanmean(space_3d)
            # z_space_time_3d[i][j] = np.sqrt(np.nanmean((space_time_3d)**2))



    """
    ##### Space 2D
    plt.axvline(x=alpha_default, color='k', linestyle='--')
    plt.axhline(y=g_default, color='k', linestyle='--')
    plt.xlabel("Air drag acceleration")
    plt.ylabel("Gravity")
    z_space_2d = z_space_2d[:-1,:-1] 
    z_min, z_max = z_space_2d.min(), z_space_2d.max() 
    c = plt.pcolormesh(x, y, z_space_2d, cmap ='hot', vmin = z_min, vmax = z_max)
    plt.title(f"Space 2D Error(m). Best(a,g) at ({x[np.unravel_index(z_space_2d.argmin(), z_space_2d.shape)]:.3f},{y[np.unravel_index(z_space_2d.argmin(), z_space_2d.shape)]:.3f})")
    plt.colorbar(c)
    plt.scatter(x[np.unravel_index(z_space_2d.argmin(), z_space_2d.shape)], y[np.unravel_index(z_space_2d.argmin(), z_space_2d.shape)],c='red')
    plt.show()
    """
    ##### Space Time 2D
    fig = plt.figure(1)
    plt.axvline(x=alpha_default, color='k', linestyle='--')
    plt.axhline(y=g_default, color='k', linestyle='--')
    plt.xlabel("Air drag acceleration")
    plt.ylabel("Gravity")
    z_space_time_2d = z_space_time_2d[:-1,:-1]
    z_space_time_2d_each = z_space_time_2d_each[:-1,:-1,:]
    z_min, z_max = z_space_time_2d.min(), z_space_time_2d.max()

    c = plt.pcolormesh(x, y, z_space_time_2d, cmap ='hot', vmin = z_min, vmax = z_max)
    plt.title(f"Space Time 2D Error(m). Best(a,g) at ({x[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)]:.3f},{y[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)]:.2f})")
    print(f"Best(a,g) at ({x[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)]:.5f},{y[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)]:.5f})")
    plt.colorbar(c)
    for k in range(z_space_time_2d_each.shape[2]):
        plt.scatter(x[np.unravel_index(z_space_time_2d_each[:,:,k].argmin(), z_space_time_2d.shape)], y[np.unravel_index(z_space_time_2d_each[:,:,k].argmin(), z_space_time_2d.shape)],
                    color=(1-min(1,traj_v[k]/300),1,1-min(1,traj_v[k]/300)),s=10)
    plt.scatter(x[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)], y[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)],c='green',marker='*',s=120)


    if args.V_2point:
        filename = f"poly{args.poly}_t{args.ang_time}_V_2point"
    elif args.tangent_xt:
        filename = f"poly{args.poly}_t{args.ang_time}_tangent_xt"

    plt.savefig(f"./Figures/{filename}.png")

    with open(f"./validate_figure/{filename}.pkl", "wb") as fp:
        pickle.dump(fig, fp)

    plt.show()
    print(f"min space time 2d: {z_min} (m)")
    """
    ##### Space 3D
    plt.axvline(x=alpha_default, color='k', linestyle='--')
    plt.axhline(y=g_default, color='k', linestyle='--')
    plt.xlabel("Air drag acceleration")
    plt.ylabel("Gravity")
    z_space_3d = z_space_3d[:-1,:-1] 
    z_min, z_max = z_space_3d.min(), z_space_3d.max()
    c = plt.pcolormesh(x, y, z_space_3d, cmap ='hot', vmin = z_min, vmax = z_max)
    plt.title(f"Space 3D Error(m). Best(a,g) at ({x[np.unravel_index(z_space_3d.argmin(), z_space_3d.shape)]:.3f},{y[np.unravel_index(z_space_3d.argmin(), z_space_3d.shape)]:.3f})")
    plt.colorbar(c)
    plt.scatter(x[np.unravel_index(z_space_3d.argmin(), z_space_3d.shape)], y[np.unravel_index(z_space_3d.argmin(), z_space_3d.shape)],c='red')
    plt.show()
    
    ##### Space Time 3D
    plt.axvline(x=alpha_default, color='k', linestyle='--')
    plt.axhline(y=g_default, color='k', linestyle='--')
    plt.xlabel("Air drag acceleration")
    plt.ylabel("Gravity")
    z_space_time_3d = z_space_time_3d[:-1,:-1] 
    z_min, z_max = z_space_time_3d.min(), z_space_time_3d.max()
    c = plt.pcolormesh(x, y, z_space_time_3d, cmap ='hot', vmin = z_min, vmax = z_max)
    plt.title(f"Space Time 3D Error(m). Best(a,g) at ({x[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)]:.3f},{y[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)]:.3f})")
    print(f"Best(a,g) at ({x[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)]:.3f},{y[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)]:.3f})")
    plt.colorbar(c)
    plt.scatter(x[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)], y[np.unravel_index(z_space_time_3d.argmin(), z_space_time_3d.shape)],c='red')
    plt.show()
    """

    # # DEBUG
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # t = trajectories_2d[trash].copy()
    # t = np.insert(t,1,0,axis=1)
    # ax.plot(t[:,0],t[:,1],t[:,2],c='red')

    # t2 = physics_predict3d(t[FIRST_POINT_IDX],t[SECOND_POINT_IDX],alpha=alpha_default,g=g_default)
    # t3 = physics_predict3d(t[FIRST_POINT_IDX],t[SECOND_POINT_IDX],
    #      alpha=x[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)],
    #      g=y[np.unravel_index(z_space_time_2d.argmin(), z_space_time_2d.shape)])
    # ax.plot(t2[:,0],t2[:,1],t2[:,2],c='blue')
    # ax.plot(t3[:,0],t3[:,1],t3[:,2],c='green')
    # plt.show()



    # width = 0.15

    # x = np.arange(len(folder_name))

    # plt.bar(x, z_space_2d, width, color='orange', label='Space 2D')
    # plt.bar(x + width, z_space_time_2d, width, color='red', label='Space & Time 2D')
    # plt.bar(x + width*2, z_space_3d, width, color='green', label='Space 3D')
    # plt.bar(x + width*3, z_time_3d, width, color='blue', label='Time 3D')
    # plt.bar(x + width*4, z_space_time_3d, width, color='pink', label='Space & Time 3D')
    # plt.xticks(x + width*2, folder_name)
    # plt.ylabel('error(meter)')
    # plt.title('Vicon & Physic Model')
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')

    # plt.show()