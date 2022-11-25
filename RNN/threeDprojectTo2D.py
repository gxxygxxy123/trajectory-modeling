import scipy
import sys
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from skspatial.objects import Plane, Line, Point, Vector
import numpy as np
import shapely.geometry as geom
from scipy import odr
from scipy.signal import savgol_filter

from sklearn import linear_model

def fitPlaneLTSQ(x,y,z): # USELESS, YOU CAN DELETE
	# Input:
	# [[x1 y1 z1]
	#  [x2 y2 z2]
	#  [x3 y3 z3]
	#  [x4 y4 z4]]
    if x.shape != y.shape or x.shape != z.shape or len(x.shape) != 1:
        sys.exit("Shape Not Equal!")
    num = x.shape[0]
    G = np.ones((num, 3))
    G[:, 0] = x  # X
    G[:, 1] = y  # Y
    Z = z # Z
    (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)

def linear_func(p, x):
    a, b = p
    return a*x + b

def FitVerticalPlaneTo2D(points,smooth_2d=True, poly=7, smooth_2d_x_accel=True):
    # points: np array, shape: (M,4) 4: XYZt
    # smooth_2d: Smooth 2D output after projection
    assert (type(points) is np.ndarray) and points.ndim == 2 and points.shape[1] == 4, f"[FitVerticalPlaneTo2D] Shape Not Equal, {points.shape}"

    M = points.shape[0]

    ret_2d = np.zeros((M,3)) # shape: (M,3) 3: XY,Z,t
    ret_3d = np.zeros((M,4)) # shape: (M,4) 4: X,Y,Z,t

    # Weighted Linear regression
    # model = linear_model.LinearRegression()
    # sample_weight = np.zeros(M)
    # sample_weight[:M-1] = np.linalg.norm(np.diff(points[:,[0,1]],axis=0),axis=1)
    # model.fit(points[:,0].reshape(-1,1),points[:,1],sample_weight=sample_weight)
    # a = model.coef_[0]
    # b = model.intercept_

    # ODR (Orthogonal distance regression)
    odr_model = odr.Model(linear_func)
    odr_data = odr.Data(points[:,0], points[:,1])
    ordinal_distance_reg = odr.ODR(odr_data, odr_model,
                                beta0=[0.2, 1.])
    odr_out = ordinal_distance_reg.run()
    a, b = odr_out.beta
    flying_direction = Vector([1,a]).unit()
    line = Line(point=[0, b], direction=flying_direction)
    first_point_projected = line.project_point(Point(points[0,[0,1]]))
    last_point_projected = line.project_point(Point(points[-1,[0,1]]))
    # flying direction is oppsite to the trajectory, change correctly
    if flying_direction.dot(Vector.from_points(first_point_projected, last_point_projected)) < 0:
        flying_direction *= -1

    # y = slope * x + intercept
    # y = a * x + b
    # a = slope
    # b = intercept

    for i in range(M):
        point = Point(points[i,[0,1]])
        point_projected = line.project_point(point)
        ret_2d[i,0] = first_point_projected.distance_point(point_projected)
        ret_2d[i,1] = points[i,2]
        ret_2d[i,2] = points[i,3]
    assert ret_2d[0,0] == 0

    if smooth_2d:
        # Way1: Polynomial Fit (1st good)
        ret_2d[:,[0,1]] = fit_2d(deg=poly,X=ret_2d[:,0],Y=ret_2d[:,1],t=ret_2d[:,2],sample_num=ret_2d.shape[0], smooth_2d_x_accel=smooth_2d_x_accel)
        ret_2d[:,0] -= ret_2d[0,0]

        # Way2: Savgol Filter (2nd good)
        # window_x = int(M/5) if int(M/5) % 2 == 1 else int(M/5)-1
        # if poly < window_x:
        #     ret_2d[:,0] = savgol_filter(ret_2d[:,0], window_x , poly)
        #     ret_2d[:,0] -= ret_2d[0,0]

    for i in range(M):
        ret_3d[i,[0,1]] = first_point_projected + flying_direction * ret_2d[i,0]
        ret_3d[i,2] = ret_2d[i,1]
        ret_3d[i,3] = ret_2d[i,2]

    # ret_2d: + smooth_2d
    # ret_3d: projected on plane at 3d coordinate + smooth_2d

    return ret_2d, ret_3d, a, b

def FitPlane(x,y,z): # USELESS, YOU CAN DELETE
    if x.shape != y.shape or x.shape != z.shape or len(x.shape) != 1:
        sys.exit("Shape Not Equal!")
    points = np.array([[x[i],y[i],z[i]] for i in range(x.shape[0])])
    idx = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) # an unnecessary but keep it not bad
    points = points[idx] # an unnecessary but keep it not bad
    plane = Plane.best_fit(points)
    #print(plane.point)
    #print(plane.normal)
    return plane.point, plane.normal

def ProjectToPlane(x,y,z,p,n): # USELESS, YOU CAN DELETE
    """
    x,y,z : a set of points
    p : a point on plane
    n : a unit normal vector of plane

    a point q = (x,y,z) onto a plane given by a point p and a normal n
    q_proj = q - dot(q-p,n) * n
    """
    if x.shape != y.shape or x.shape != z.shape or len(x.shape) != 1:
        sys.exit("Shape Not Equal!")
    x_new = np.zeros(x.shape[0])
    y_new = np.zeros(x.shape[0])
    z_new = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        q = np.array([x[i],y[i],z[i]])
        q_proj = q - np.dot(q-p,n) * n
        x_new[i] = q_proj[0]
        y_new[i] = q_proj[1]
        z_new[i] = q_proj[2]
    return x_new, y_new, z_new

def ThreeDPlaneTo2DCoor(x,y,z,p,n): # USELESS, YOU CAN DELETE
    """
    x,y,z : a set of points on plane
    p : a point on plane
    n : a unit normal vector of plane
    """

    if x.shape != y.shape or x.shape != z.shape or len(x.shape) != 1:
        sys.exit("Shape Not Equal!")
    P = np.ones((x.shape[0], 3))
    P[:,0] = x
    P[:,1] = y
    P[:,2] = z
    r = p # start of the trajectory

    ex = np.cross(r,n)/np.linalg.norm(np.cross(r,n))
    ey = np.cross(n,ex)/np.linalg.norm(np.cross(n,ex))

    x_new = np.zeros(x.shape[0])
    y_new = np.zeros(y.shape[0])

    for i in range(P.shape[0]):
        x_new[i], y_new[i] = np.dot(np.array([ex,ey]), r - P[i,:])

    x_new -= x_new[0]
    y_new -= y_new[0]

    # Rotate 2d trajectory
    u1 = np.array([x_new[1],y_new[1]]) / np.linalg.norm(np.array([x_new[1],y_new[1]]))
    u2 = np.array([1,0])
    theta = np.arccos(np.dot(u1,u2))
    for i in range(x_new.shape[0]):
        px = x_new[i] * math.cos(theta) - y_new[i] * math.sin(theta)
        py = x_new[i] * math.sin(theta) + y_new[i] * math.cos(theta)
        x_new[i], y_new[i] = px, py

    # Mirror the trajectory by x-axis
    if y_new[y_new.shape[0]//2] > 0:
        y_new = -y_new

    return x_new, y_new

def poly7_origin(x, a, b, c, d, e, f, g):
    # 7-th poly go through origin
    return a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x

def fit_2d(deg, X, Y, t, sample_num, smooth_2d_x_accel, poly=7):
    X = np.copy(X)
    Y = np.copy(Y)
    curve_data=np.zeros((sample_num, 2))
    points=np.zeros((X.shape[0],2))
    points_new=np.zeros((X.shape[0],2))

    fitx = X
    fity = Y
    idx = np.isfinite(fitx) & np.isfinite(fity)
    f1, err, _, _, _ = np.polyfit(fitx[idx], fity[idx], deg, full=True)
    p1 = np.poly1d(f1)

    x_sorted = np.sort(X)
    yvals = p1(x_sorted)

    points[:,0]=fitx
    points[:,1]=fity
    dataX = np.linspace(np.nanmin(fitx[idx]), np.nanmax(fitx[idx]),X.shape[0])
    dataY = p1(dataX)
    curve_data[:,0]=dataX
    curve_data[:,1]=dataY

    coords = curve_data[:,:]
    line = geom.LineString(coords)
    for j in range(0,X.shape[0],1):
        point = geom.Point(points[j,0], points[j,1])
        #print(point.x)
        #if point.x==0 and point.y==0:
        #    break
        if np.isnan(points[j,0]):
            points_new[j,0]=np.nan
            points_new[j,1]=np.nan
        else:
            point_on_line = line.interpolate(line.project(point))

            points_new[j,0]=(point_on_line.x)
            points_new[j,1]=(point_on_line.y)
    # 位移量平滑化
    if smooth_2d_x_accel:
        # Method 1 Ohmybug (vx-t figure)

        # plot_x = np.arange(1,X.shape[0])

        # tmp_x = [None] * X.shape[0]
        # for j in range(X.shape[0]):
        #     point = geom.Point(points[j,[0,1]])
        #     point_on_line = line.interpolate(line.project(point))
        #     tmp_x[j] = point_on_line.x

        # plot_y = np.diff(tmp_x)

        # # 多項式 fit
        # param = np.polyfit(plot_x, plot_y, poly)
        # z = np.polyval(param, plot_x)

        # newX = np.cumsum(np.insert(z,0,0))
        # newY = p1(newX)

        # points_new[:,0] = newX
        # points_new[:,1] = newY

        # Method 2 (x-t figure)
        assert poly == 7, "poly default 7 and use the 'poly7_origin' passing origin. TODO"
        plot_x = t

        tmp_x = [None] * (X.shape[0])
        for j in range(X.shape[0]):
            point = geom.Point(points[j,[0,1]])
            point_on_line = line.interpolate(line.project(point))
            tmp_x[j] = point_on_line.x

        plot_y = np.array(tmp_x)

        # 多項式 fit
        popt, pcov = curve_fit(poly7_origin, plot_x, plot_y)
        newX = poly7_origin(plot_x, *popt)
        newY = p1(newX)
        points_new[:,0] = newX
        points_new[:,1] = newY

        assert newX[0] == 0.0
        # param = np.polyfit(plot_x, plot_y, poly)
        # z = np.polyval(param, plot_x)

        # newX = np.insert(z,0,0)
        # newY = p1(newX)

        # points_new[:,0] = newX
        # points_new[:,1] = newY

        # plt.plot(plot_x, plot_y, label='Unsmooth')
        # plt.plot(plot_x, newX, label='Smooth')
        # plt.xlabel(f"time(s)")
        # plt.ylabel(f"X value of points")
        # plt.legend()
        # plt.show()

    return points_new

def fit_3d(x, y, z, N, deg): # MAY HAVE BUGS!
    x = np.copy(x)
    y = np.copy(y)
    z = np.copy(z)

    x_new=np.zeros(N)
    y_new=np.zeros(N)
    z_new=np.zeros(N)
    curve_x=np.zeros(N)
    curve_y=np.zeros(N)
    curve_z=np.zeros(N)

    f_start=0
    f_end = x.shape[0]
    fitx1 = x[f_start:f_end]
    fity1 = y[f_start:f_end]
    fitz1 = z[f_start:f_end]
    idx = np.isfinite(fitx1) & np.isfinite(fity1) & np.isfinite(fitz1)

    yx= np.polyfit(fity1[idx], fitx1[idx], deg)
    pYX = np.poly1d(yx)
    yz= np.polyfit(fity1[idx], fitz1[idx], deg)
    pYZ = np.poly1d(yz)

    #y_sorted = np.sort(fity1)
    y_sorted = fity1 # an
    x_pYX = pYX(y_sorted)
    z_pYZ = pYZ(y_sorted)

    fitx1_new, fity1_new, fitz1_new, pYX_new, pYZ_new = get_cleaned_fit(fitx1, fity1, fitz1, pYX, pYZ, deg)

    #y_sorted_new = np.sort(fity1_new)
    y_sorted_new = fity1_new   # an
    x_pYX_new = pYX_new(y_sorted_new)
    z_pYZ_new = pYZ_new(y_sorted_new)

    x_new[f_start:f_end]=fitx1_new
    y_new[f_start:f_end]=fity1_new
    z_new[f_start:f_end]=fitz1_new
    curve_x[f_start:f_end]=x_pYX_new
    curve_y[f_start:f_end]=y_sorted_new
    curve_z[f_start:f_end]=z_pYZ_new
    return x_new, y_new, z_new, curve_x, curve_y, curve_z

def get_cleaned_fit(x, y, z, polyYX, polyYZ, deg):
    array_yz = abs(z - polyYZ(y))
    max_accept_deviation = 0.8
    mask_yz = array_yz >= max_accept_deviation
    rows_to_del = np.asarray(tuple(te for te in np.where(mask_yz)[0]))

    for i in rows_to_del:
        x[i]=np.nan
        y[i]=np.nan
        z[i]=np.nan

    array_yx = abs(x - polyYX(y))
    mask_yx = array_yx >= max_accept_deviation
    rows_to_del = np.asarray(tuple(te for te in np.where(mask_yx)[0]))

    for i in rows_to_del:
        x[i]=np.nan
        y[i]=np.nan
        z[i]=np.nan


    idx = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    #####################
    #print(x.shape)
    #x = x[idx]
    #y = y[idx]
    #z = z[idx]
    #print(x.shape)
    ####################
    cYX = np.polyfit(y[idx], x[idx], deg)
    pYX = np.poly1d(cYX)
    cYZ= np.polyfit(y[idx], z[idx], deg)
    pYZ = np.poly1d(cYZ)
    return x, y, z, pYX, pYZ
"""
def plot_result(x_new, y_new, z_new, curve_x, curve_y, curve_z, N,ax):

    camPoints = np.vstack((x_new, y_new, z_new))
    curves = np.vstack((curve_x, curve_y, curve_z))

    data = [camPoints]
    data2 = [curves]


    #ax.scatter(x_new,y_new,z_new)
    print("len(x_new):",len(x_new))
    f_start=0
    f_end = curve_x.shape[0]
    print("len(curve_x):",len(curve_x))
    print("------------------------")
    #for i in range(f_start,f_end):
    #    ax.scatter(curve_x[i], curve_y[i], curve_z[i], color = np.random.rand(3,))
    ax.plot(curve_x[f_start:f_end], curve_y[f_start:f_end], curve_z[f_start:f_end],color = np.random.rand(3,))

    #plt.show()


def plot_result(x_new, y_new, z_new, curve_x, curve_y, curve_z, N):

    camPoints = np.vstack((x_new, y_new, z_new))
    curves = np.vstack((curve_x, curve_y, curve_z))

    data = [camPoints]
    data2 = [curves]
    #fig = plt.figure()
    #ax = p3.Axes3D(fig)

    lines = [ax.plot(data[0][0,0:1], data[0][1,0:1], data[0][2,0:1], 'o')[0]]

    # set the axes properties
    ax.set_xlim3d([-4, 4])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 15])
    ax.set_ylabel('Depth')

    ax.set_zlim3d([np.nanmin(z_new), np.nanmax(z_new)])
    ax.set_zlabel('Y')
    plt.gca().invert_zaxis()
    ax.set_title('')

    ax.scatter3D(x_new,y_new,z_new)


    f_start=0
    f_end = x_new.shape[0]
    ax.plot(curve_x[f_start:f_end], curve_y[f_start:f_end], curve_z[f_start:f_end])

    #plt.show()
"""
