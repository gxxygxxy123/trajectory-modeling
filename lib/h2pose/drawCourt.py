from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sqrt, pi, sin, cos, tan
from PIL import Image
from PIL import ImageOps

import os
import numpy as np
import cv2
import pprint

pp = pprint.PrettyPrinter(indent=4)

quadric = gluNewQuadric()

cam = np.array([0, -20, 5])
obj = np.array([0,   0, 0])
up  = np.array([0,   0, 1])

fovy = 40
width = 1280
height = 720

def startGL():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE)
    glutInitWindowSize(width, height)
    glutCreateWindow("Hello")
    init()

def init():
    glEnable(GL_DEPTH_TEST)
    glClearColor(1., 1., 1., 1.)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # Tell the Opengl, we are going to set PROJECTION function
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()  # always call this before gluPerspective
    # set intrinsic parameters (fovy, aspect, zNear, zFar)
    gluPerspective(fovy, width/height, 0.1, 100000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def drawCourt():
    glColor3f(0, 0, 0)
    glBegin(GL_LINES)
    glVertex3f(-3, 6.7, 0)
    glVertex3f(3, 6.7, 0)
    glEnd()
    glBegin(GL_LINES)
    glVertex3f(-3, -6.7, 0)
    glVertex3f(3, -6.7, 0)
    glEnd()
    glBegin(GL_LINES)
    glVertex3f(3, -6.7, 0)
    glVertex3f(3, 6.7, 0)
    glEnd()
    glBegin(GL_LINES)
    glVertex3f(-3, -6.7, 0)
    glVertex3f(-3, 6.7, 0)
    glEnd()

def drawCircle(center, radius, color):
    sides = 120  # how fineness the circle is
    glColor3f(color[0]/255, color[1]/255, color[2]/255)
    glBegin(GL_POLYGON)
    for i in range(sides):
        cosine = radius * cos(i*2*pi/sides) + center[0]
        sine = radius * sin(i*2*pi/sides) + center[1]
        glVertex3f(cosine, sine, 0)
    glEnd()

def sphere(x,y,z):
    global quadric
    glColor3f(1, 0, 1)
    glTranslatef(x,y,z)
    gluSphere(quadric,0.02,32,32)
    glTranslatef(-x,-y,-z)

def drawTrack(start, end):
    points = 21
    unit = (end - start) / (points-1)
    x = np.linspace(-1.5, 1.5, points)
    y = -x**2 + 4.25

    for p in range(points):
        x_ = start[0] + p*unit[0]
        y_ = start[1] + p*unit[1]
        z_ = y[p]
        sphere(x_,y_,z_)
        # print(x_,y_,z_)

def toRad(deg):
    return deg*pi/180

def toDeg(rad):
    return rad*180/pi

def rotateRYP(points, roll, yaw, pitch):
    R_pitch = np.array([[1,          0,           0],
                        [0, cos(pitch), -sin(pitch)],
                        [0, sin(pitch),  cos(pitch)]])
    R_yaw = np.array([[cos(yaw), -sin(yaw), 0],
                      [sin(yaw),  cos(yaw), 0],
                      [0,         0, 1]])
    R_roll = np.array([[cos(roll), 0, sin(roll)],
                       [0, 1,         0],
                       [-sin(roll), 0, cos(roll)]])

    R = R_roll @ R_yaw @ R_pitch
    if len(points.shape) == 1:  # only one point
        new_points = R @ points.reshape(3, 1)
    else:
        new_points = R @ points.T
    new_points = new_points.T
    return R, new_points

def getLabel(cam_gl, obj_gl, up_gl, R):
    pose_label = {}
    
    cam_right_spcs = (R @ np.array([1,0,0]).reshape(3,1)).reshape(3)
    cam_down_spcs = (R @ np.array([0,-1,0]).reshape(3,1)).reshape(3)
    cam_in_spcs = (R @ np.array([0,0,-1]).reshape(3,1)).reshape(3)
    pose_label['spcs'] = {
        'cam_center':cam_gl.tolist(),
        'axis':[cam_right_spcs.tolist(), 
               cam_down_spcs.tolist(), 
               cam_in_spcs.tolist()]
    }

    T = np.array([cam_right_spcs, cam_down_spcs, cam_in_spcs])
    world_position_ccs = T @ (np.array([0,0,0])-cam_gl)
    world_RG_ccs = T.T[0]
    world_N_ccs = T.T[1]
    world_NxRG_ccs = T.T[2]
    pose_label['ccs'] = {
        'wor_center':world_position_ccs.tolist(),
        'axis':[world_RG_ccs.tolist(), 
                world_N_ccs.tolist(), 
                world_NxRG_ccs.tolist()]
    }
    return pose_label


def writeLabel(label, path):
    with open(path, 'w') as f:
        yaml.dump(label, f, default_flow_style=False, sort_keys=False)

def getImg():
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (width, height), data)
    # in my case image is flipped top-bottom for some reason
    image = ImageOps.flip(image)
    # image.save(outfolder+'/glutout_r{}_{}.png'.format(radius, 0), 'PNG')
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image

def draw():
    global cam, obj, up
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(cam[0], cam[1], cam[2],
              obj[0], obj[1], obj[2],
              up[0], up[1], up[2])

    # draw badminton court
    drawCourt()
    image = getImg()

    # track
    start = np.array([-1.5, 2])
    end   = np.array([2, -5.5])
    drawCircle(start, 0.05, [0, 255, 255])
    drawCircle(end, 0.05, [255, 255, 0])
    drawTrack(start, end)
    track = getImg()

    glutSwapBuffers()
    return image, track

if __name__ == '__main__':
    focal = height / (2*tan(pi*fovy/360))
    print("focal:", focal)

    ccsZ = obj - cam
    ccsZ = ccsZ / np.linalg.norm(ccsZ)
    ccsX = np.cross(ccsZ, up)
    ccsX = ccsX / np.linalg.norm(ccsX)
    ccsY = np.cross(ccsZ, ccsX)
    ccsY = ccsY / np.linalg.norm(ccsY)
    ccsO = cam
    print('ccsO in wcs:',ccsO)
    print('ccsX in wcs:',ccsX)
    print('ccsY in wcs:',ccsY)
    print('ccsZ in wcs:',ccsZ)

    w2c = np.array([ccsX,ccsY,ccsZ])
    print('w2c:')
    pp.pprint(w2c)

    wcsO = w2c @ (np.array([0,0,0]) - cam).reshape(3,1)
    wcsO = wcsO.reshape(3)
    wcsX = w2c @ np.array([1,0,0]).reshape(3,1)
    wcsX = wcsX.reshape(3)
    wcsY = w2c @ np.array([0,1,0]).reshape(3,1)
    wcsY = wcsY.reshape(3)
    wcsZ = w2c @ np.array([0,0,1]).reshape(3,1)
    wcsZ = wcsZ.reshape(3)
    print('wcsO in ccs:',wcsO)
    print('wcsX in ccs:',wcsX)
    print('wcsY in ccs:',wcsY)
    print('wcsZ in ccs:',wcsZ)

    startGL()
    img, track = draw()
    cv2.imwrite('calib_court.png', img)
    cv2.imwrite('track_court.png', track)