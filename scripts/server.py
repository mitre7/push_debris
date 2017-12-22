#!/usr/bin/env python

from push_debris.srv import *
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import math
import collections
import re


def averageGradient(img):
    dx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
    dy = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)

    average_dx = dx.mean()
    average_dy = dy.mean()

    average_gradient = math.atan2(-average_dy, average_dx)
    return average_gradient


def extractState(img, x, y, orientation):
    offset_x = 90
    offset_y = 90
    rows, cols, ch = img.shape
    p = (x, y)
    print(img.shape)
    df = orientation
    print(df)

    fx = 1920/ float(cols)
    fy = 1080/ float(rows)
    small = cv2.resize(img, (1920, 1080))

    #print(int(p[0]*fx), int(p[1]*fy))
    #small = cv2.circle(small, (int(p[0]*fx), int(p[1]*fy)), 5, (0,0,0), 3)
    #cv2.imshow("small", small)
    #cv2.waitKey()

    p = (int(p[0]*fx), int(p[1]*fy))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    M = cv2.getRotationMatrix2D((p[0],p[1]), 90 - df, 1)
    dst = cv2.warpAffine(gray, M, (cols,rows))
    cropped = dst[p[1] - offset_y:p[1] + offset_y, p[0] - offset_x:p[0] + offset_x]

    #cv2.imshow("cropped", cropped)
    #cv2.waitKey()

    grid_width = 3
    grid_height = 3
    grid_step = 60
    state = [0] * 4

    j = 0

    for i in range(grid_width * grid_height):
        if i%2 == 0:
            continue


        row = int(i / grid_width)
        col = i % grid_width

        x1 = row * grid_step
        x2 = (row + 1) * grid_step
        y1 = col * grid_step
        y2 = (col + 1) * grid_step
        crop = cropped[y1:y2, x1:x2]

	#cv2.imshow("cropped", crop)
        #cv2.waitKey()

        if i == 5:
            crop = cropped[y1+5:y2, x1:x2]


        gradient_strength = averageStrength(crop)
        if gradient_strength > 0.99:
            state[j] = 1
        else:
            state[j] = 0

        j += 1

    print(state)

    return tuple(state)


def averageStrength(img):

    dx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
    dy = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)

    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)

    rows, cols = mag.shape

    #cv2.imshow("mag", mag)
    #cv2.waitKey()

    sum = 0
    for x in range(0, cols):
        for y in range(0, rows):
            if mag[y][x] < 50:
                sum += 1

    strength = float (sum) / ( float(rows*cols))

    return strength


def loadQ():
    action_space = 4
    Q = collections.defaultdict(lambda: np.zeros(action_space))
    print("nai")
    key_file = "/home/mitre/Mitre/clopema_packages/src/radioroso_certh/push_debris/txt/key.txt"
    value_file = "/home/mitre/Mitre/clopema_packages/src/radioroso_certh/push_debris/txt/value.txt"
    with open(key_file, 'r') as keyfile, open(value_file, 'r') as valuefile:
            keys = keyfile.readlines()
            values = valuefile.readlines()

    for i in range(len(keys)):
        tmp_key = re.findall('\d+', keys[i])
        key_id = list(map(int, tmp_key))

        values[i] = values[i].replace("[", "")
        values[i] = values[i].replace("]", "")
        tmp_v = values[i].split()
        q_values = list(map(float, tmp_v))

        Q[tuple(key_id)] = q_values

    return Q



def predict_push(req):
    print('inside server!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(req.x, req.y, req.orientation)

    VALID_ACTIONS = [0, 90, 180, 270]
    Q = loadQ()

    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(req.img, "bgr8")

    req.orientation = 180 - req.orientation * 180 / np.pi


    state = extractState(cv_image, req.x, req.y, req.orientation)

    action = np.argmax(Q[state])
    direction = VALID_ACTIONS[action]
    #print(direction)

    rows, cols, ch = cv_image.shape
    fx = 1920/ float(cols)
    fy = 1080/ float(rows)
    small = cv2.resize(cv_image, (1920, 1080))
    p = (int((req.x)*fx), int((req.y)*fy))

    if action == 0 or action == 2:
        t = 60
    else:
        t = 30


    df =  req.orientation
    x1 = int( p[0] - ( np.cos( (direction + df) * np.pi/180 ) * t ) )
    y1 = int( p[1] + ( np.sin( (direction + df) * np.pi/180 ) * t ) )


    x2 = int( p[0] - ( np.cos( (direction + df) * np.pi/180 ) * (t + 75) ) )
    y2 = int( p[1] + ( np.sin( (direction + df) * np.pi/180 ) * (t + 75) ) )

    #small = cv2.circle(small, p, 5, (0,0,0), 3)
    #small = cv2.circle(small, (x1, y1), 3, (0,0,0), 3)
    #small = cv2.circle(small, (x2, y2), 3, (0,0,0), 3)
    #cv2.imshow("small", small)
    #cv2.waitKey()

    res_x1 = int(x1/fx)
    res_y1 = int(y1/fy)
    res_x2 = int(x2/fx)
    res_y2 = int(y2/fy)

    theta = df + direction

    print("Direction:")
    print(direction)


    print("Theta before:")
    print(theta)

    if theta >= 0 and theta <= 180:
       theta = 180 - theta
    elif theta > 180 and theta <= 270:
       theta = 360 - (theta-180)
    else:
       theta = 270 - (theta - 270)

    print("Theta after:")
    print(theta)
	

    theta = theta * np.pi / 180
    

    return PushDebrisResponse(res_x1, res_y1, res_x2, res_y2, theta)


def push_debris_server():
    rospy.init_node('push_debris_server')
    s = rospy.Service('push_debris', PushDebris, predict_push)
    print("predict the push direction")
    rospy.spin()


if __name__ == "__main__":
    push_debris_server()
