import math
import numpy as np
import os
import sys
import bisect



DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(os.path.dirname(DIRNAME))
sys.path.append(f"{ROOTDIR}/lib")
from point import load_points_from_csv

def space_err(gt_curve, pd_curve, bidirectional=False):
    # ground truth & predict curve shape: (?, 3) or (?, 4), must include timestamp
    # assert (gt_curve.ndim == 2 and pd_curve.ndim == 2 and
    #         gt_curve.shape[1] == pd_curve.shape[1] and
    #         (gt_curve.shape[1] == 3 or gt_curve.shape[1] == 4)), "Wrong"
    if gt_curve.shape[0] == 0 or pd_curve.shape[0] == 0:
        return np.nan

    # timestamp of trajectory should be sorted
    assert np.all(gt_curve[:-1,-1] <= gt_curve[1:,-1]), "timestamp of gt_curve should be sorted"
    assert np.all(pd_curve[:-1,-1] <= pd_curve[1:,-1]), "timestamp of pd_curve should be sorted"

    # If input is 2D (X,Z,t), make it to 3D (X,0,Z,t)
    if gt_curve.shape[1] == 3:
        new_gt = np.zeros((gt_curve.shape[0],4))
        new_gt[:,[0,2,3]] = gt_curve
        gt_curve = new_gt
        new_pd = np.zeros((pd_curve.shape[0],4))
        new_pd[:,[0,2,3]] = pd_curve
        pd_curve = new_pd

    # Discard Timestamp, (X,Y,Z,t) -> (X,Y,Z)
    gt_curve = gt_curve[:,:-1]
    pd_curve = pd_curve[:,:-1]

    # For each point on pd, find the minimal distance with gt
    ans = 0.0
    for P in pd_curve:
        min_distance = float("inf")
        if gt_curve.shape[0] == 1:
            min_distance = np.linalg.norm(P-gt_curve[0])
        else:
            for i in range(gt_curve.shape[0]-1):
                A = gt_curve[i]
                B = gt_curve[i+1]
                ab = B-A
                ap = P-A
                bp = P-B
                if np.dot(ap,ab) <= 0.0:
                    distance = np.linalg.norm(ap)
                elif np.dot(bp,ab) >= 0.0:
                    distance = np.linalg.norm(bp)
                else:
                    # Perpendicular distance of point to segment
                    distance = np.linalg.norm(np.cross(ab,ap))/np.linalg.norm(ab)
                min_distance = min(min_distance, distance)
        ans += min_distance
    ans /= pd_curve.shape[0]

    if not bidirectional:
        return ans

    # For each point on gt, find the minimal distance with pd
    ans2 = 0.0
    for P in gt_curve:
        min_distance = float("inf")
        if pd_curve.shape[0] == 1:
            min_distance = np.linalg.norm(P-pd_curve[0])
        else:
            for i in range(pd_curve.shape[0]-1):
                A = pd_curve[i]
                B = pd_curve[i+1]
                ab = B-A
                ap = P-A
                bp = P-B
                if np.dot(ap,ab) <= 0.0:
                    distance = np.linalg.norm(ap)
                elif np.dot(bp,ab) >= 0.0:
                    distance = np.linalg.norm(bp)
                else:
                    # Perpendicular distance of point to segment
                    distance = np.linalg.norm(np.cross(ab,ap))/np.linalg.norm(ab)
                min_distance = min(min_distance, distance)
        ans2 += min_distance
    ans2 /= gt_curve.shape[0]

    return min(ans,ans2)


def time_err(gt_curve, pd_curve, bidirectional=False):
    # ground truth & predict curve shape: (?, 3) or (?, 4), must include timestamp
    # assert (gt_curve.ndim == 2 and pd_curve.ndim == 2 and
    #         gt_curve.shape[1] == pd_curve.shape[1] and
    #         (gt_curve.shape[1] == 3 or gt_curve.shape[1] == 4)), "Wrong"
    if gt_curve.shape[0] == 0 or pd_curve.shape[0] == 0:
        return np.nan

    # timestamp of trajectory should be sorted
    assert np.all(gt_curve[:-1,-1] <= gt_curve[1:,-1]), "timestamp of gt_curve should be sorted"
    assert np.all(pd_curve[:-1,-1] <= pd_curve[1:,-1]), "timestamp of pd_curve should be sorted"

    # positive means predict too quick, negative means predict LAG

    # If input is 2D (X,Z,t), make it to 3D (X,0,Z,t)
    if gt_curve.shape[1] == 3:
        new_gt = np.zeros((gt_curve.shape[0],4))
        new_gt[:,[0,2,3]] = gt_curve
        gt_curve = new_gt
        new_pd = np.zeros((pd_curve.shape[0],4))
        new_pd[:,[0,2,3]] = pd_curve
        pd_curve = new_pd

    # For each point on pd, find the time diff with the close point C on gt
    ans = 0.0
    for P in pd_curve:
        min_distance = float("inf")
        time_diff = float("inf")
        if gt_curve.shape[0] == 1:
            min_distance = np.linalg.norm(P[:-1]-gt_curve[0,:-1])
            time_diff = gt_curve[0,-1]-P[-1]
        else:
            for i in range(gt_curve.shape[0]-1):
                A = gt_curve[i]
                B = gt_curve[i+1]
                ab = B[:-1]-A[:-1]
                ap = P[:-1]-A[:-1]
                bp = P[:-1]-B[:-1]
                t = float("inf")
                if np.dot(ap,ab) <= 0.0:
                    distance = np.linalg.norm(ap)
                    t = A[-1]-P[-1]
                elif np.dot(bp,ab) >= 0.0:
                    distance = np.linalg.norm(bp)
                    t = B[-1]-P[-1]
                else:
                    # Perpendicular distance of point to segment
                    distance = np.linalg.norm(np.cross(ab,ap))/np.linalg.norm(ab)
                    Ct = (A[-1] * np.linalg.norm(bp) + B[-1] * np.linalg.norm(ap)) / (np.linalg.norm(ap)+np.linalg.norm(bp))
                    t = Ct - P[-1]
                if distance <= min_distance:
                    min_distance = distance
                    time_diff = t
        ans += time_diff
    ans /= pd_curve.shape[0]

    if not bidirectional:
        return ans

    # For each point on gt, find the time diff with the close point C on pd
    ans2 = 0.0
    for P in gt_curve:
        min_distance = float("inf")
        time_diff = float("inf")
        if pd_curve.shape[0] == 1:
            min_distance = np.linalg.norm(P[:-1]-pd_curve[0,:-1])
            time_diff = pd_curve[0,-1]-P[-1]
        else:
            for i in range(pd_curve.shape[0]-1):
                A = pd_curve[i]
                B = pd_curve[i+1]
                ab = B[:-1]-A[:-1]
                ap = P[:-1]-A[:-1]
                bp = P[:-1]-B[:-1]
                t = float("inf")
                if np.dot(ap,ab) <= 0.0:
                    distance = np.linalg.norm(ap)
                    t = A[-1]-P[-1]
                elif np.dot(bp,ab) >= 0.0:
                    distance = np.linalg.norm(bp)
                    t = B[-1]-P[-1]
                else:
                    # Perpendicular distance of point to segment
                    distance = np.linalg.norm(np.cross(ab,ap))/np.linalg.norm(ab)
                    Ct = (A[-1] * np.linalg.norm(bp) + B[-1] * np.linalg.norm(ap)) / (np.linalg.norm(ap)+np.linalg.norm(bp))
                    t = Ct - P[-1]
                if distance <= min_distance:
                    min_distance = distance
                    time_diff = t
        ans2 += time_diff
    ans2 /= gt_curve.shape[0]


    return min(ans,ans2)

def space_time_err(gt_curve, pd_curve, bidirectional=False):

    # ground truth & predict curve shape: (?, 3) or (?, 4), must include timestamp
    # assert (gt_curve.ndim == 2 and pd_curve.ndim == 2 and
    #         gt_curve.shape[1] == pd_curve.shape[1] and
    #         (gt_curve.shape[1] == 3 or gt_curve.shape[1] == 4)), "Wrong"
    if gt_curve.shape[0] == 0 or pd_curve.shape[0] == 0:
        return np.nan

    # timestamp of trajectory should be sorted
    assert np.all(gt_curve[:-1,-1] <= gt_curve[1:,-1]), "timestamp of gt_curve should be sorted"
    assert np.all(pd_curve[:-1,-1] <= pd_curve[1:,-1]), "timestamp of pd_curve should be sorted"

    # If input is 2D (X,Z,t), make it to 3D (X,0,Z,t)
    if gt_curve.shape[1] == 3:
        new_gt = np.zeros((gt_curve.shape[0],4))
        new_gt[:,[0,2,3]] = gt_curve
        gt_curve = new_gt
        new_pd = np.zeros((pd_curve.shape[0],4))
        new_pd[:,[0,2,3]] = pd_curve
        pd_curve = new_pd

    # For each point on pd, find the distance with the point C at same timestamp on gt
    ans = 0.0
    for P in pd_curve:
        distance = float("inf")
        if P[-1] >= gt_curve[-1,-1]:
            distance = np.linalg.norm(P[:-1] - gt_curve[-1,:-1])
        elif P[-1] <= gt_curve[0,-1]:
            distance = np.linalg.norm(P[:-1] - gt_curve[0,:-1])
        else:
            low = bisect.bisect_left(gt_curve[:,-1],P[-1])
            # C is between A & B
            A = gt_curve[low-1]
            B = gt_curve[low]
            assert P[-1] >= A[-1] and P[-1] <= B[-1] and low != 0, f" {B[-1]} >= {P[-1]} >= {A[-1]}"

            # using interpoliation by timestamp to calculate X,Y,Z of C
            C = (A[:-1] * (B[-1]-P[-1]) + B[:-1] * (P[-1]-A[-1])) / (B[-1] - A[-1])
            distance =  np.linalg.norm(P[:-1] - C)

        ans += distance
    ans /= pd_curve.shape[0]

    if not bidirectional:
        return ans

    # For each point on gt, find the distance with the point C at same timestamp on pd
    ans2 = 0.0
    for P in gt_curve:
        distance = float("inf")
        if P[-1] >= pd_curve[-1,-1]:
            distance = np.linalg.norm(P[:-1] - pd_curve[-1,:-1])
        elif P[-1] <= pd_curve[0,-1]:
            distance = np.linalg.norm(P[:-1] - pd_curve[0,:-1])
        else:
            low = bisect.bisect_left(pd_curve[:,-1],P[-1])
            # C is between A & B
            A = pd_curve[low-1]
            B = pd_curve[low]
            assert P[-1] >= A[-1] and P[-1] <= B[-1] and low != 0, f" {B[-1]} >= {P[-1]} >= {A[-1]}"

            # using interpoliation by timestamp to calculate X,Y,Z of C
            C = (A[:-1] * (B[-1]-P[-1]) + B[:-1] * (P[-1]-A[-1])) / (B[-1] - A[-1])
            distance =  np.linalg.norm(P[:-1] - C)

        ans2 += distance
    ans2 /= gt_curve.shape[0]


    return min(ans,ans2)


def time_after_err(gt_curve, pd_curve, time):
    # ground truth & predict curve shape: (?, 3) or (?, 4), must include timestamp
    assert (gt_curve.ndim == 2 and pd_curve.ndim == 2 and
            gt_curve.shape[1] == pd_curve.shape[1] and
            (gt_curve.shape[1] == 3 or gt_curve.shape[1] == 4) and
            #gt_curve[0,-1] == 0.0 and pd_curve[0,-1] == 0.0 and
            time >= 0.0
            ), f"{gt_curve.shape} {pd_curve.shape}"

    if gt_curve.shape[0] == 0 or pd_curve.shape[0] == 0:
        return np.nan

    # timestamp of trajectory should be sorted
    assert np.all(gt_curve[:-1,-1] <= gt_curve[1:,-1]), "timestamp of gt_curve should be sorted"
    assert np.all(pd_curve[:-1,-1] <= pd_curve[1:,-1]), "timestamp of pd_curve should be sorted"

    # output: after TIME, the point error

    if time <= gt_curve[-1,-1] and time <= pd_curve[-1,-1] and time >= gt_curve[0,-1] and time >= pd_curve[0,-1]:

        if gt_curve.shape[0] > 1:
            low = bisect.bisect_left(gt_curve[:,-1],time)
            if low == 0:
                low = 1
            # C is between A & B
            A = gt_curve[low-1]
            B = gt_curve[low]
            assert time >= A[-1] and time <= B[-1], f" {B[-1]} >= {time} >= {A[-1]}"
            gt_position = (A[:-1] * (B[-1]-time) + B[:-1] * (time-A[-1])) / (B[-1] - A[-1])
        else:
            gt_position = gt_curve[0,:-1]
        if pd_curve.shape[0] > 1:
            low = bisect.bisect_left(pd_curve[:,-1],time)
            if low == 0:
                low = 1
            # C is between A & B
            A = pd_curve[low-1]
            B = pd_curve[low]
            assert time >= A[-1] and time <= B[-1], f" {B[-1]} >= {time} >= {A[-1]}"
            pd_position = (A[:-1] * (B[-1]-time) + B[:-1] * (time-A[-1])) / (B[-1] - A[-1])
        else:
            pd_position = pd_curve[0,:-1]
        return np.linalg.norm(gt_position-pd_position)
    else:
        return np.nan
"""
def land_error_2curve(curve1, curve2):
    # curve shape: (?, 2) or (?, 3)
    assert curve1.shape[1] == curve2.shape[1], "Two Curve must be both 2D or 3D!"
    return np.linalg.norm(land_point(curve1)-land_point(curve2))

def time_error_2curve(curve1, curve2, fps):
    # curve shape: (?, 2) or (?, 3)
    assert curve1.shape[1] == curve2.shape[1], "Two Curve must be both 2D or 3D!"
    return abs(curve1.shape[0] - curve2.shape[0]) / fps

def land_point(curve, ground_height=0.1):
    # curve shape: (?, 2) 2: xy,z OR (?, 3): x,y,z
    # return [xy] (2D) or [x,y] (3D)
    ans = np.nan
    assert curve.shape[0] >= 2, "Curve is only a point or empty!"

    for i in range(curve.shape[0]):
        ans = curve[i][:-1]
        if curve[i][-1] <= ground_height:
            break
    return ans
"""
"""
def land_point_2d_old(curve):
    # curve shape: (?, 2) 2: xy,z

    ans = np.nan
    assert curve.shape[0] >= 2, "Curve is only a point or empty!"

    for i in range(1, curve.shape[0]):
        if curve[i][1] * curve[i-1][1] <= 0:
            ans = ((curve[i][0]*abs(curve[i-1][1]) + curve[i-1][0]*abs(curve[i][1])) 
                / abs(curve[i][1]-curve[i-1][1]))
    if np.isnan(ans):
        ans = curve[-1][0] + (curve[-1][0] - curve[-3][0]) * (0 - curve[-1][1])/(0 - curve[-3][1])

    return ans
"""

if __name__ == '__main__':
    l1 = load_points_from_csv('../trajectories_dataset/vicon/20220322_124858/Model3D.csv')
    l1 = [x.toXYZT() for x in l1]
    l1 = np.stack(l1)
    l2 = load_points_from_csv('../trajectories_dataset/vicon/20220322_124858/vicon.csv')
    l2 = [x.toXYZT() for x in l2]
    l2 = np.stack(l2)

    print(f"Space : {space_err(l1,l2):.3f}")
    print(f"Time : {time_err(l1,l2,120,300):.3f}")
    print(f"Space & Time : {space_time_err(l1,l2):.3f}")