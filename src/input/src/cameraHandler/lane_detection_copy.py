#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import warnings

warnings.filterwarnings("ignore")

array = [
    "phai", "thang", "trai",
    "thang", "thang", "thang",
    "trai", "thang", "trai",  
    "phai"
]

# Canny for image
def CannyImage(image):
    # Step 1: Gray convert
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Remove noise Gaussian Filter
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Step 3: Canny Detect Edges
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Create polygon
def region_of_interest(image):
    height = 480
    polygons = np.array([
        [(0, height-200), (0, height), (640, height), (640, height-240), (620, 200)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Lines 
def display_lines(image, lines):
    try:
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image
    except:
        pass

def make_coordiantes(image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1*6/7)
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])
    except:
        pass

def averaged_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    try:
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]
                if slope > 0:
                    right_fit.append((slope, intercept))
            right_fit_avr = np.average(right_fit, axis=0)
            right_line = make_coordiantes(image, right_fit_avr)
            return np.array([right_line])
    except:
        pass

def start(Image):
    try:
        global pre_x_mid, x2r, x1r, combo_image, check
        angle_pub = rospy.Publisher('Update_angle', Float32, queue_size = 100)
        speed_pub = rospy.Publisher('Update_speed', Float32, queue_size = 100) 

        print("This line excuted - 1")

        lane_image = np.copy(Image)
        canny = CannyImage(lane_image)
        cropped_image = region_of_interest(canny)

        print("This line excuted - 2")

        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=5)
    
        avr_lines = averaged_slope_intercept(lane_image, lines)

        line_image = display_lines(lane_image, avr_lines)

        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

        print("This line excuted - 3")
    
        check = True
        right = []
        if avr_lines is not None:
            right = avr_lines.reshape(4)
        print(right)
        x_mid = int((right[0]+right[2])/2)
        print("+++++++++++++++++++++++++++++")
        print(x_mid)
     
        x1r = right[0]
        x2r = right[2]
        x_mid = int((x1r + x2r)/2)

        if x_mid >= 615 and x2r >= 440:
            speed = 20
            angle = 15
            angle_pub.publish(angle)
            speed_pub.publish(speed)
        elif x2r < 0:
            speed = 20
            angle = 20
            angle_pub.publish(angle)
            speed_pub.publish(speed)
        elif x_mid < 520 or x2r in range (0, 420):
            speed = 20
            angle = -20
            angle_pub.publish(angle)
            speed_pub.publish(speed)    
        elif x_mid in range (520, 580):
            speed = 20
            angle = 0
            angle_pub.publish(angle)
            speed_pub.publish(speed)  
        elif x1r > 1000:
            speed = 0
            angle = 0
            angle_pub.publish(angle)
            speed_pub.publish(speed)
            turn, pos = graph.callback(array)
        pre_x_mid = x_mid
        print("This line excuted - 4")
    except:
        print("Exception Occured")
        check = False
    finally:
        if check == False:
            if pre_x_mid >= 615 and x2r >= 440:
                speed = 20
                angle = 15
                angle_pub.publish(angle)
                speed_pub.publish(speed)
            elif x2r < 0:
                speed = 20
                angle = 20
                angle_pub.publish(angle)
                speed_pub.publish(speed)
            elif pre_x_mid < 520 or x2r in range (0, 420):
                speed = 20
                angle = -20
                angle_pub.publish(angle)
                speed_pub.publish(speed)    
            elif pre_x_mid in range (520, 550):
                speed = 20
                angle = 0
                angle_pub.publish(angle)
                speed_pub.publish(speed)  
            elif pre_x_mid < 0 or pre_x_mid > 640:
                speed = 0
                angle = 0
                angle_pub.publish(angle)
                speed_pub.publish(speed)
                #turn, pos = graph.callback(array)
            check = True

        cv2.imshow("Frame preview", combo_image)
        cv2.imshow("Poly", cropped_image)
        key = cv2.waitKey(1)
        print("This line is always executed")

        print("This line excuted - 5")


class CameraHandler():
    # ===================================== INIT==========================================
    def __init__(self):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        rospy.init_node('CAMnod', anonymous=True)
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.callback)
        rospy.spin()

    def callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        start(self.cv_image)
    
            
if __name__ == '__main__':
    try:
        nod = CameraHandler()
    except rospy.ROSInterruptException:
        pass
