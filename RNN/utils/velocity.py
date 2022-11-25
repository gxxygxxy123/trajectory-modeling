import math
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sys

def v_of_2dtraj(traj, speed_t, ang_t=0.05, V_2point=False, vx_t_poly=3, tangent_xt=False):
    # traj: a complete 2d trajectory (?,3), contains X,Y,t
    # speed_t: calculate velocity at speed_t
    # ang_t: calculate pitch angle using linear regression in ang_t(s)
    # V_2point: use two point to calculate vecocity
    # vx_t_poly: polynomial regression of vx-t figure to calculate instantaneous velocity
    # *return velocity vector Vxy(m/s), instantaneous velocity, pitch angle, at traj[M]

    assert type(traj) is np.ndarray and traj.ndim == 2 and traj.shape[1] == 3 and speed_t >= 0
    assert np.all(traj[:-1,2] <= traj[1:,2]), "timestamp of gt_curve should be sorted"

    assert np.allclose(np.diff(traj[:,2]),traj[1,2]-traj[0,2]) # the timestamp diff is all same, i.e. a fix sampling rate timestamp 
    fps = 1/(traj[1,2]-traj[0,2])

    # M ~ N
    M = math.ceil(fps*(speed_t-traj[0,2]))
    N = math.floor(fps*(speed_t-traj[0,2]+ang_t))

    slope = linear_model.LinearRegression().fit(traj[M:N,0].reshape(-1,1),traj[M:N,1]).coef_[0] # y = slope*x+b

    if V_2point:
        instantaneous_velocity = np.linalg.norm(traj[M+1,[0,1]]-traj[M,[0,1]])*fps
    elif tangent_xt:
        # tangent of x-t figure
        x_t = np.polyfit(traj[:,2], traj[:,0], 7)
        vx_t = np.polyder(x_t)
        instantaneous_vx = np.polyval(vx_t, traj[M,2])
        instantaneous_velocity = instantaneous_vx * ((slope**2+1)**0.5)

        # plt.plot(traj[:,2], traj[:,0])
        # plt.xlabel(f"time(s)")
        # plt.ylabel(f"X")
        # plt.show()

    else: # NOT GOOD, DONT USE, TODO REMOVE
        # vx-t at traj[M,2]
        vx_t = np.polyfit(traj[:-1,2]+0.5/fps, np.diff(traj[:,0])*fps, vx_t_poly) # vx-t
        instantaneous_vx = np.polyval(vx_t, traj[M,2])
        instantaneous_velocity = instantaneous_vx * ((slope**2+1)**0.5)

        plt.plot(traj[:-1,2]+0.5/fps, np.diff(traj[:,0])*fps)
        plt.plot(traj[:-1,2]+0.5/fps, np.polyval(vx_t, traj[:-1,2]+0.5/fps))
        plt.scatter(traj[M,2],np.polyval(vx_t, traj[M,2]), color='red')
        plt.xlabel(f"time(s)")
        plt.ylabel(f"Velocity(m/s)")
        plt.show()


    vxy = np.array([1,slope]) / np.linalg.norm(np.array([1,slope])) * instantaneous_velocity

    return vxy, instantaneous_velocity, math.degrees(math.atan2(slope,1))