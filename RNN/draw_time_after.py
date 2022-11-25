import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns
sns.set()
sns.color_palette("Paired")

parser = argparse.ArgumentParser(description="Draw time t")

parser.add_argument("-t","--time", type=float, default=0.1, help="Input Time (default 0.1)")

args = parser.parse_args()

t = args.time
fps = 120

plt.title(f"Time After Error(2D) FPS:{fps}. Input time: {t}")
plt.xlabel("Time(s)")
plt.ylabel("Error(m)")

if os.path.exists(f"./pkl/query1_{t}.pkl"):
    with open(f"./pkl/query1_{t}.pkl", "rb") as fp:   # Unpickling
        query1_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), query1_time_after_2d, label='Query Model 1(Our Data)', linestyle='-.')

if os.path.exists(f"./pkl/query1_{t}_0.215.pkl"):
    with open(f"./pkl/query1_{t}_0.215.pkl", "rb") as fp:   # Unpickling
        query1_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), query1_time_after_2d, label='Query Model 1(0.215)', linestyle='--')

if os.path.exists(f"./pkl/query1_{t}_0.242.pkl"):
    with open(f"./pkl/query1_{t}_0.242.pkl", "rb") as fp:   # Unpickling
        query1_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), query1_time_after_2d, label='Query Model 1(0.242)')

if os.path.exists(f"./pkl/query2_{t}_V_2point.pkl"):
    with open(f"./pkl/query2_{t}_V_2point.pkl", "rb") as fp:   # Unpickling
        query2_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), query2_time_after_2d, label='Query Model 2(Our Data)', linestyle='-.')


if os.path.exists(f"./pkl/query2_{t}_V_2point_0.215.pkl"):
    with open(f"./pkl/query2_{t}_V_2point_0.215.pkl", "rb") as fp:   # Unpickling
        query2_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), query2_time_after_2d, label='Query Model 2(0.215)', linestyle='--')

if os.path.exists(f"./pkl/query2_{t}_V_2point_0.242.pkl"):
    with open(f"./pkl/query2_{t}_V_2point_0.242.pkl", "rb") as fp:   # Unpickling
        query2_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), query2_time_after_2d, label='Query Model 2(0.242)')

if os.path.exists(f"./pkl/Physics_{t}_V_2point_0.215.pkl"):
    with open(f"./pkl/Physics_{t}_V_2point_0.215.pkl", "rb") as fp:   # Unpickling
        physics_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(int(fps*t)/fps, 2, 0.02), physics_time_after_2d, label='Physical Model(0.215)', linestyle='--')

if os.path.exists(f"./pkl/Physics_{t}_V_2point_0.242.pkl"):
    with open(f"./pkl/Physics_{t}_V_2point_0.242.pkl", "rb") as fp:   # Unpickling
        physics_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(int(fps*t)/fps, 2, 0.02), physics_time_after_2d, label='Physical Model(0.242)')

if os.path.exists(f"./pkl/BLSTM_{t}_0.215.pkl"):
    with open(f"./pkl/BLSTM_{t}_0.215.pkl", "rb") as fp:   # Unpickling
        blstm_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), blstm_time_after_2d, label='BLSTM(0.215)', linestyle='--')

if os.path.exists(f"./pkl/BLSTM_{t}_0.242.pkl"):
    with open(f"./pkl/BLSTM_{t}_0.242.pkl", "rb") as fp:   # Unpickling
        blstm_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), blstm_time_after_2d, label='BLSTM(0.242)')


if os.path.exists(f"./pkl/TF_{t}_0.215.pkl"):
    with open(f"./pkl/TF_{t}_0.215.pkl", "rb") as fp:   # Unpickling
        tf_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), tf_time_after_2d, label='Transformer(0.215)', linestyle='--')

if os.path.exists(f"./pkl/TF_{t}_0.242.pkl"):
    with open(f"./pkl/TF_{t}_0.242.pkl", "rb") as fp:   # Unpickling
        tf_time_after_2d = pickle.load(fp)
        plt.plot(np.arange(0, 2, 0.02), tf_time_after_2d, label='Transformer(0.242)')

plt.legend()
plt.show()