#!/usr/bin/env python

import sys
import rospy
from push_debris.srv import *
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def push_debris_client(x, y, orientation, rgb):
  rospy.wait_for_service('push_debris')
  try:
    push_debris = rospy.ServiceProxy('push_debris', PushDebris)
    resp1 = push_debris(x, y, orientation, rgb)
    return resp1.push_direction
  except:
    print("Service call failed")

if __name__ == "__main__":
  x = 1394 
  y = 1415 
  orientation = 180-125.03 
  rgb = cv2.imread("/home/marios/Desktop/cap.png", 1)
  #print(rgb.shape)
  #cv2.imshow("rgb", rgb)
  #cv2.waitKey()
  bridge = CvBridge()
  image_message = bridge.cv2_to_imgmsg(rgb, "bgr8")
  print(push_debris_client(x, y, orientation, image_message))
