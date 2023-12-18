#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import rospy
import cv2
import os
import numpy as np
import datetime
import time
import sys
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String, Float64
#from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
import math

import image_converter as ic
import const
from config import Config
import config 

config = Config.data_collection
# if config['vehicle_name'] == 'fusion':
#     from fusion.msg import Control
# elif config['vehicle_name'] == 'rover':
#     from rover.msg import Control
# else:
#     exit(config['vehicle_name'] + 'not supported vehicle.')

# global first_time_step, vel_prev, time_prev
first_time_step = True
vel_prev = 0
time_prev = 0

class AccelerationCalculation():
    def __init__(self):
        
        # self.steering = 0
        # self.throttle = 0
        # self.brake = 0
        rospy.init_node('acceleration')
        self.vel_x = self.vel_y = self.vel_z = 0
        self.vel = 0
        # self.pos_x = self.pos_y = self.pos_z = 0
        self.accel = 0
        # self.accel_x = self.accel_y = self.accel_z = 0
        # self.img_cvt = ic.ImageConverter()
        
        # self.vel_prev = 0
        self.vel_curr = 0 
        # self.time_prev = 0
        self.time_curr = 0
        
        ##
        # data will be saved in a location specified with rosparam path_to_e2e_data

        # create csv data file
        # name_datatime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # #path = '../data/' + sys.argv[1] + '/' + name_datatime + '/'
        # path = rospy.get_param('path_to_e2e_data', 
        #                 './e2e_data') + '/' + sys.argv[1] + '/' + name_datatime + '/'
        # if os.path.exists(path):
        #     print('path exists. continuing...')
        # else:
        #     print('new folder created: ' + path)
        #     os.makedirs(path)

        # self.text = open(str(path) + name_datatime + const.DATA_EXT, "w+")
        # self.path = path


    def calc_velocity(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    def calc_accel(self, curr_v, curr_t, prev_v, prev_t ):

        curr_accel = (curr_v - prev_v)/(curr_t - prev_t)
        # accel = math.sqrt(x**2 + y**2) # + z**2) # ignore the z acceleration for now
        # if x < 0.0 :#or y <0.0:
        #     accel = -1 * accel
        # else:
        #     accel = accel
        return curr_accel

    # def steering_throttle_cb(self, value):
    #     self.throttle = value.throttle
    #     self.steering = value.steer
    #     self.brake = value.brake

    def pos_vel_cb(self, value):
        # global first_time_step
        global first_time_step, vel_prev, time_prev
        if first_time_step:
            self.vel_x = value.twist.twist.linear.x 
            self.vel_y = value.twist.twist.linear.y
            self.vel_z = value.twist.twist.linear.z
            self.vel = self.calc_velocity(self.vel_x, self.vel_y, self.vel_z)
            vel_prev = self.vel
            unix_time = time.time()
            time_prev = unix_time
            self.accel = 0
            first_time_step = False
        else:
            self.vel_x = value.twist.twist.linear.x 
            self.vel_y = value.twist.twist.linear.y
            self.vel_z = value.twist.twist.linear.z
            self.vel = self.calc_velocity(self.vel_x, self.vel_y, self.vel_z)
            unix_time = time.time()
            self.time_curr = unix_time
            self.vel_curr = self.vel
            self.accel = self.calc_accel(self.vel_curr, self.time_curr, vel_prev, time_prev)
            vel_prev = self.vel_curr
            time_prev = self.time_curr
       

    # def imu_cb(self, value):
    #     self.accel_x = -1 * value.linear_acceleration.x
    #     self.accel_y = -1 * value.linear_acceleration.y
    #     self.accel_z = -1 * value.linear_acceleration.z - 9.8 #subtracting gravity
    #     self.accel = self.calc_accel(self.accel_x, self.accel_y, self.accel_z)
    #     # self.accel_y = value.angular_velocity.z
    #     # if config['version'] >= 0.92:
    #     #     line = "{},{}\r\n".format(self.accel_x, self.accel_y)
    #     # self.text_accel.write(line)

    # def recorder_cb(self, data):
    #     img = self.img_cvt.imgmsg_to_opencv(data)

    #     # no more cropping in data collection - new strategy    
    #     # # crop
    #     if config['crop'] is True: # this is for old datasets
    #         cropped = img[config['image_crop_y1']:config['image_crop_y2'],
    #                       config['image_crop_x1']:config['image_crop_x2']]

    #     unix_time = time.time()
    #     time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    #     file_full_path = str(self.path) + str(time_stamp) + const.IMAGE_EXT
    #     if config['crop'] is True:
    #         cv2.imwrite(file_full_path, cropped)
    #     else:
    #         cv2.imwrite(file_full_path, img)
    #     # sys.stdout.write(file_full_path + '\r')
    #     # sys.stdout.flush()
    #     if config['version'] >= 0.92:
    #         line = "{}{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\r\n".format(time_stamp, const.IMAGE_EXT, 
    #                                                     self.steering, 
    #                                                     self.throttle,
    #                                                     self.brake,
    #                                                     unix_time,
    #                                                     self.vel,
    #                                                     self.vel_x,
    #                                                     self.vel_y,
    #                                                     self.vel_z,
    #                                                     self.pos_x,
    #                                                     self.pos_y,
    #                                                     self.pos_z,
    #                                                     self.accel,
    #                                                     self.accel_x,
    #                                                     self.accel_y,
    #                                                     self.accel_z)
    #     else:
    #         line = "{}{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\r\n".format(time_stamp, const.IMAGE_EXT, 
    #                                                     self.steering, 
    #                                                     self.throttle,
    #                                                     unix_time,
    #                                                     self.vel,
    #                                                     self.vel_x,
    #                                                     self.vel_y,
    #                                                     self.vel_z,
    #                                                     self.pos_x,
    #                                                     self.pos_y,
    #                                                     self.pos_z,
    #                                                     self.accel,
    #                                                     self.accel_x,
    #                                                     self.accel_y,
    #                                                     self.accel_z)
    #     self.text.write(line)
    #     cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}     \t{4:.3f}\r'.format(self.steering, 
    #                       self.throttle, self.brake, self.vel, self.accel)

    #     sys.stdout.write(cur_output)
    #     sys.stdout.flush()                                                 
        # cur_output = '{0:.3f}'.format(self.vel)
        # sys.stdout.write(cur_output + '\r')
        # sys.stdout.flush()

def main():

    accel_cal = AccelerationCalculation()

    
    # rospy.Subscriber(config['vehicle_control_topic'], Control, accel_cal.steering_throttle_cb)
    rospy.Subscriber(config['base_pose_topic'], Odometry, accel_cal.pos_vel_cb)
    # rospy.Subscriber(config['imu'], Imu, accel_cal.imu_cb)
    # rospy.Subscriber(config['camera_image_topic'], Image, accel_cal.recorder_cb)
    accel_pub = rospy.Publisher('acceleration', Float64, queue_size=1)
    accel_data = Float64()
    
    while not rospy.is_shutdown():
        accel_data.data = accel_cal.accel
        accel_pub.publish(accel_data)
        rospy.sleep(1)
 
    # try:
    #     rospy.spin()
        
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     print("\nBye...")    


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print('Usage: ')
        exit('$ rosrun acceleration calc_acceleration.py')

    main()