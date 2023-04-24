#!/usr/bin/env python3
import io
import numpy as np
import time

import rospy

from cv_bridge       import CvBridge
from sensor_msgs.msg import Image
import cv2 as cv

#cap = cv.VideoCapture("/home/pi/Desktop/test.mp4")
cap = cv.VideoCapture("/home/pi/Desktop/backup_9_1/video_record31.h264")
cap = cv.VideoCapture(0)
cap.set(3,512)
cap.set(4,288)
print(cap.isOpened())
bridge = CvBridge()
def talker():
    pub = rospy.Publisher("/automobile/image_raw",Image, queue_size =1)
    rospy.init_node('Cam_Pub',anonymous = False)
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            time.sleep(2)
        
        msg = bridge.cv2_to_imgmsg(frame,"bgr8")
        pub.publish(msg)
        rate.sleep()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()

if __name__ =='__main__':
    talker()