#!/usr/bin/env python3
import io
import numpy as np
import time

import rospy

from cv_bridge       import CvBridge
from sensor_msgs.msg import Image
import cv2 as cv
from delivery_robot_serial_handler.msg import Pos

bridge = CvBridge()
image = None 

P1 =0
P2 =0
P1_pre = 0
P2_pre = 0
def enc_callback(msg):

    global P1, P2   
    P1 = msg.P1
    P2 = msg.P2
frame_count = 0
def callback(msg):
        global image,P1, P1_pre, P2, P2_pre,frame_count
        image =bridge.imgmsg_to_cv2(msg)
        cv.putText(image, 'P1 = ' + str(P1), (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    2)
        cv.putText(image, 'P2 = ' + str(P2), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    2)
        cv.imwrite("./frame_{}.jpg".format(frame_count), image)

        # tăng số frame đã đọc được lên 1
        frame_count += 1
        #cv.imshow("Video", image)
        
rospy.init_node("imagetimer111", anonymous=True)        
rospy.Subscriber("/automobile/image_raw",Image,callback)

time.sleep(2)
rospy.Subscriber('Pos', Pos, enc_callback)
while not rospy.is_shutdown():
    #print(image)
    time.sleep(0.1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
