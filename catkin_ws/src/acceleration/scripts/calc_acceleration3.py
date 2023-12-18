#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import rospy
import os
import numpy as np
import datetime
import time
import sys
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Header#, Float64
from nav_msgs.msg import Odometry
from acceleration.msg import acceleration_msg

import math
from config import Config
import config 

config = Config.data_collection
# vel_prev = 0
# time_prev = 0

class AccelerationCalculation():
    def __init__(self):
        self.sub = rospy.Subscriber(config['base_pose_topic'], Odometry, self.pos_vel_cb)
        self.pub = rospy.Publisher('acceleration', acceleration_msg, queue_size=20)
        self.accel_data = acceleration_msg()
        # self.time_current_data = Float64()
        self.vel_x = self.vel_y = self.vel_z = 0
        self.vel = 0
        self.accel = 0
        self.vel_prev = 0 
        self.time_curr = 0
        # self.first_time_step = True
        self.last_vel_time = None  # Initialize the timestamp

    def calc_velocity(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    def calc_accel(self, curr_v, curr_t, prev_v, prev_t):
        if curr_t != prev_t and curr_v != prev_v:
            curr_accel = (curr_v - prev_v) / (curr_t - prev_t)
        else:
            curr_accel = 0  # Avoid division by zero
        return curr_accel


    def pos_vel_cb(self, value):
        # unix_time = time.time()
        self.vel_x = value.twist.twist.linear.x
        self.vel_y = value.twist.twist.linear.y
        self.vel_z = value.twist.twist.linear.z
        self.vel = self.calc_velocity(self.vel_x, self.vel_y, self.vel_z)

        # Extract the timestamp from the incoming Odometry message
        vel_time = value.header.stamp
        
        if self.last_vel_time is not None :

            if vel_time > self.last_vel_time:
                # Calculate acceleration using the time from the last velocity message
                self.accel = self.calc_accel(self.vel, vel_time.to_sec(), self.vel_prev, self.last_vel_time.to_sec())
                # print("vel_time > last_vel_time")
                # Publish acceleration and current time using your custom message
                self.accel_data.acceleration = self.accel
                self.accel_data.current_time = vel_time #rospy.Time.now()  # Use the time from the velocity message
                self.accel_data.header = Header()
                self.accel_data.header.stamp = vel_time
                self.pub.publish(self.accel_data)

                self.vel_prev = self.vel
                self.last_vel_time = vel_time

        elif self.last_vel_time is None:
            # Handle the case when there isn't a previous velocity message
            self.vel_curr = self.vel
            self.last_vel_time = vel_time
            self.accel_data.acceleration = 0
            self.accel_data.current_time = vel_time # rospy.Time.now()  # Use the time from the velocity message
            self.accel_data.header = Header()
            self.accel_data.header.stamp = vel_time
            self.pub.publish(self.accel_data)
        


if __name__ == '__main__':
    rospy.init_node('acceleration')
    # rate = rospy.Rate(100)
    accel_calc = AccelerationCalculation()
    rospy.spin()
    # rate.sleep()