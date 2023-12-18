#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:23:14 2023
This allows to synchronize messages between 
the data_collection node and the acceleration node 
using the message_filters package

@author: aws
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
from std_msgs.msg import String
from acceleration.msg import acceleration_msg
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber

import math

import image_converter as ic
import const
from config import Config
import config 

config = Config.data_collection
if config['vehicle_name'] == 'fusion':
    from fusion.msg import Control
elif config['vehicle_name'] == 'rover':
    from rover.msg import Control
else:
    exit(config['vehicle_name'] + 'not supported vehicle.')


class DataCollection():
    def __init__(self):
        #control
        self.steering = 0
        self.throttle = 0
        self.brake = 0
        # counter to know the amount fo data collected.
        self.counter = 0
        #velocity
        self.vel_x = self.vel_y = self.vel_z = 0
        self.vel = 0
        #position
        self.pos_x = self.pos_y = self.pos_z = 0
        #imu
        self.accel_x = self.accel_y = self.accel_z = 0
        #img
        self.img_cvt = ic.ImageConverter()
        #acceleration
        self.accel_calc = 0 # calculated acceleration from velocity and time.
        self.accel_curr_time = 0
        self.last_accel_time = None # used to ensure our data have the right timestamps
        #heading
        self.yaw_rate_deg = 0
        self.heading_initial = 0.0  # Initial heading (in radians)
        self.heading_current = self.heading_initial
        self.last_heading_time = None

        # Create subscribers for synchronized messages
        self.control_sub = Subscriber(config['vehicle_control_topic'], Control)
        self.odom_sub = Subscriber(config['base_pose_topic'], Odometry)
        self.imu_sub = Subscriber(config['imu'], Imu)
        self.accel_sub = Subscriber(config['accel'], acceleration_msg)
        self.image_sub = Subscriber(config['camera_image_topic'], Image)

        # Use ApproximateTimeSynchronizer to synchronize messages, including control messages
        self.sync_subs = [self.control_sub, self.image_sub, self.odom_sub, self.imu_sub, self.accel_sub]
        self.ts = ApproximateTimeSynchronizer(self.sync_subs, queue_size=1, slop=0.1)
        self.ts.registerCallback(self.synced_callback)
        
        
        # data will be saved in a location specified with rosparam path_to_e2e_data
        # create csv data file
        name_datatime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        #path = '../data/' + sys.argv[1] + '/' + name_datatime + '/'
        path = rospy.get_param('path_to_e2e_data', 
                        './e2e_data') + '/' + sys.argv[1] + '/' + name_datatime + '/'
        if os.path.exists(path):
            print('path exists. continuing...')
        else:
            print('new folder created: ' + path)
            os.makedirs(path)
        self.text = open(str(path) + name_datatime + const.DATA_EXT, "w+")
        self.path = path


    def calc_velocity(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    
    def synced_callback(self, control_msg, image_msg, odom_msg, imu_msg, accel_msg):

        #print timestamps for debugging
        # -----------------------------------------------------------------------------------------------
        # #uncomment this block for debugging
        # print("*************************************************")
        # print("Control Timestamp:", control_msg.header.stamp.to_sec())
        # print("Image Timestamp:", image_msg.header.stamp.to_sec())
        # print("Odom Timestamp:", odom_msg.header.stamp.to_sec())
        # print("IMU Timestamp:", imu_msg.header.stamp.to_sec())
        # print("Accel Timestamp:", accel_msg.header.stamp.to_sec())
        # print("*************************************************")
        # -----------------------------------------------------------------------------------------------

        
        # Extract data from synchronized messages
        # -----------------------------------------------------------------------------------------------
        #control
        ###################################################################################
        self.steering = control_msg.steer
        self.throttle = control_msg.throttle
        self.brake = control_msg.brake
        
        #postion
        ###################################################################################
        self.pos_x = odom_msg.pose.pose.position.x
        self.pos_y = odom_msg.pose.pose.position.y
        self.pos_z = odom_msg.pose.pose.position.z
        
        #velocity
        ###################################################################################
        self.vel_x = odom_msg.twist.twist.linear.x
        self.vel_y = odom_msg.twist.twist.linear.y
        self.vel_z = odom_msg.twist.twist.linear.z
        self.vel = self.calc_velocity(self.vel_x, self.vel_y, self.vel_z)
        
        #imu
        ###################################################################################
        self.accel_x = imu_msg.linear_acceleration.x
        self.accel_y = imu_msg.linear_acceleration.y
        self.accel_z = imu_msg.linear_acceleration.z
        
        #heading
        ###################################################################################
        # Get yaw rate (angular velocity around the z-axis) from the IMU
        yaw_rate = imu_msg.angular_velocity.z  # Assuming z-axis is the yaw axis
        self.yaw_rate_deg = yaw_rate#*(180/math.pi) # Yaw rate in degrees per second
        # Get the timestamp from the IMU message
        current_heading_time = imu_msg.header.stamp
        if self.last_heading_time is not None:
            # Calculate time difference (delta_t) between current and previous IMU messages
            delta_t = (current_heading_time - self.last_heading_time).to_sec()
            # Integrate yaw rate to calculate change in heading (Δθ)
            delta_heading = yaw_rate * delta_t
            # Update the current heading
            self.heading_current += delta_heading #(in radians)
            # Ensure the heading stays within the [0, 2π) range
            self.heading_current = self.heading_current % (2 * math.pi)
            #(convert to degrees)
            self.heading_current *= 180/math.pi 
        # Update the last heading time for the next iteration
        self.last_heading_time = current_heading_time
        
        #acceleration
        ###################################################################################
        self.accel_calc = accel_msg.acceleration
        self.accel_curr_time = accel_msg.current_time.to_sec() # from nsec to sec

        #image
        ###################################################################################
        img = self.img_cvt.imgmsg_to_opencv(image_msg)
        # no more cropping in data collection - new strategy    
        # # crop
        if config['crop'] is True: # this is for old datasets
            cropped = img[config['image_crop_y1']:config['image_crop_y2'],
                          config['image_crop_x1']:config['image_crop_x2']]
        # -----------------------------------------------------------------------------------------------

        # Record data in the csv file
        # -----------------------------------------------------------------------------------------------
        unix_time = time.time()
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        file_full_path = str(self.path) + str(time_stamp) + const.IMAGE_EXT

        if config['version'] >= 0.92:
            line = "{}{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\r\n".format(time_stamp, const.IMAGE_EXT, 
                                                        self.steering, 
                                                        self.throttle,
                                                        self.brake,
                                                        unix_time,
                                                        self.vel,
                                                        self.vel_x,
                                                        self.vel_y,
                                                        self.vel_z,
                                                        self.pos_x,
                                                        self.pos_y,
                                                        self.pos_z,
                                                        self.accel_x,
                                                        self.accel_y,
                                                        self.accel_z,
                                                        self.yaw_rate_deg,
                                                        self.heading_current,
                                                        self.accel_calc,
                                                        self.accel_curr_time)
        else:
            line = "{}{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\r\n".format(time_stamp, const.IMAGE_EXT, 
                                                        self.steering, 
                                                        self.throttle,
                                                        unix_time,
                                                        self.vel,
                                                        self.vel_x,
                                                        self.vel_y,
                                                        self.vel_z,
                                                        self.pos_x,
                                                        self.pos_y,
                                                        self.pos_z,
                                                        self.accel_x,
                                                        self.accel_y,
                                                        self.accel_z,
                                                        self.yaw_rate_deg,
                                                        self.heading_current,
                                                        self.accel_calc,
                                                        self.accel_curr_time)

        # To avoid outliers in the data, we will ignore the data line when there is a spike in the acceleration
        if abs(self.accel_calc) > 3.0:
            pass
        else:
            # save the image
            if config['crop'] is True:
                cv2.imwrite(file_full_path, cropped)
            else:
                cv2.imwrite(file_full_path, img)
            # sys.stdout.write(file_full_path + '\r')
            # sys.stdout.flush()

            # Update last_accel_time
            self.text.write(line)
            self.last_accel_time = self.accel_curr_time
            
            # update counter
            self.counter += 1

            # Print out the data
            cur_output = '{0:d} \t{1:.3f} \t{2:.3f} \t{3:.3f} \t{4:.3f}     \t{5:.3f}     \t{6:.3f}     \t{7:.3f}\r'.format(self.counter, self.steering, 
                          self.throttle, self.brake, self.vel, self.accel_calc, self.heading_current, self.yaw_rate_deg) 

            
            sys.stdout.write(cur_output)
            sys.stdout.flush()
        # -----------------------------------------------------------------------------------------------


def main():
    try:
        rospy.init_node('data_collection')
        dc = DataCollection()
        print("Data collection node is running")
        print('count \tsteer \tthrt \tbrake \tvelocity \tacceleration \theading \tyaw')
        rospy.spin()        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print("An error occurred: ", str(e))
    finally:
        print("Shutting down data collection node")

if __name__ == '__main__':
    #check if the usage is correct
    if len(sys.argv) < 2:
        print('Usage: ')
        exit('$ rosrun data_collection data_collection.py your_data_id')
    #call the main function
    main()