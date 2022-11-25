# Lejun Shen, IACSS, 2017
# Measurement and Performance Evaluation of Lob Technique using Aerodynamic Model In Badminton Matches
import matplotlib.pyplot as plt
import math
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
import seaborn as sns
sns.set()

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")
from point import Point

def physics_predict3d(starting_point, second_point, flight_time=10, touch_ground_cut=True, alpha=0.2151959552, g=9.81):
    # starting_point, second_point, shape: (4,) 4: XYZt
    fps = 1/(second_point[3] - starting_point[3])

    initial_velocity = (second_point[:3]-starting_point[:3]) * fps # shape: (3,) unit: m/s

    traj = solve_ivp(lambda t, y: bm_ball(t, y, alpha=alpha, g=g), [0, flight_time], np.concatenate((starting_point[:3], initial_velocity)), t_eval = np.arange(0, flight_time, 1/fps)) # traj.t traj.y

    xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
    t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)
    trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

    # Cut the part under the ground
    if touch_ground_cut:
        for i in range(trajectories.shape[0]-1):
            if trajectories[i,2] >= 0 and trajectories[i+1,2] <= 0:
                trajectories = trajectories[:i+1,:]
                break
    # Add timestamp correctly
    trajectories[:,3] += (starting_point[3]) # shape: (N points, 4)

    return trajectories # shape: (N points, 4) , include input two Points

def physics_predict3d_v2(starting_point, v, fps, flight_time=10, touch_ground_cut=True, alpha=0.2151959552, g=9.81):

    initial_velocity = v

    traj = solve_ivp(lambda t, y: bm_ball(t, y, alpha=alpha, g=g), [0, flight_time], np.concatenate((starting_point[:3], initial_velocity)), t_eval = np.arange(0, flight_time, 1/fps)) # traj.t traj.y

    xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
    t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)
    trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)

    # Cut the part under the ground
    if touch_ground_cut:
        for i in range(trajectories.shape[0]-1):
            if trajectories[i,2] >= 0 and trajectories[i+1,2] <= 0:
                trajectories = trajectories[:i+1,:]
                break
    # Add timestamp correctly
    trajectories[:,3] += (starting_point[3]) # shape: (N points, 4)

    return trajectories # shape: (N points, 4) , include starting_point


def bm_ball(t,x,alpha=0.2151959552, g=9.81):
    # velocity
    v = math.sqrt(x[3]**2+x[4]**2+x[5]**2)
    # ordinary differential equations (3)
    xdot = [ x[3], x[4], x[5], -alpha*x[3]*v, -alpha*x[4]*v, -g-alpha*x[5]*v]
    return xdot

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

if __name__ == '__main__':
    # test physics_predict3d function
    # p1 = np.array([5.6, 6.3, 2.9, 123.45])
    # p2 = np.array([5.4, 6.0, 3.1, 123.616])
    # a = physics_predict3d(p1, p2)
    # print(a)
    # sys.exit(1)

    ###########################
    flight_time = 5
    fps = 120
    starting_point = [0, 0, 2.5]
    datas = 100
    ###########################

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_xlabel("Elevation(degree)")
    ax1.set_ylabel("Initial Velocity(km/hr)")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_title(f"Physics Model, Datas: {datas}")

    # Elevation & Speed Distribution
    corr = [[1,0],[0,1]]
    mu = 0, 75
    scale = 35, 20

    elevation, speed = get_correlated_dataset(datas, corr, mu, scale)

    elevation[elevation>85] = 80.0
    elevation[elevation<-80] = -80.0
    speed[speed<10] = 10.0
    # elevation = np.arange(-60,60,1) # -60 ~ 60
    # speed = np.arange(30, 150, 1) # 30 ~ 150


    speed = speed * 1000/3600 # km/hr -> m/s

    cnt = 0
    for e,s in zip(elevation,speed):
        initial_velocity = [s * math.cos(e/180*math.pi), 0, s * math.sin(e/180*math.pi)]

        traj = solve_ivp(bm_ball, [0, flight_time], starting_point + initial_velocity, t_eval = np.arange(0, flight_time, 1/fps)) # traj.t traj.y
        cnt += 1

        xyz = np.swapaxes(traj.y[:3,:], 0, 1) # shape: (N points, 3)
        t = np.expand_dims(traj.t,axis=1) # shape: (N points, 1)
        trajectories = np.concatenate((xyz, t),axis=1) # shape: (N points, 4)
        # Cut the part under the ground
        for i in range(trajectories.shape[0]-1):
            if trajectories[i,2] >= 0 and trajectories[i+1,2] <= 0:
                trajectories = trajectories[:i+1,:]
                break

        ax1.scatter(e, s*3600/1000, color='red')
        ax2.plot(trajectories[:,0], trajectories[:,2], label=f"2D Physic-based trajectories FPS:{fps}, total: {cnt}",  marker='o', markersize=1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()