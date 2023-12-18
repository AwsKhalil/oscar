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
from std_msgs.msg import String, Float64
from nav_msgs.msg import Odometry
import math
from config import Config
import config 

config = Config.data_collection
first_time_step = True
vel_prev = 0
time_prev = 0

class AccelerationCalculation():
    def __init__(self):
        self.sub = rospy.Subscriber(config['base_pose_topic'], Odometry, self.pos_vel_cb)
        self.pub = rospy.Publisher('acceleration', Float64, queue_size=20)
        
        self.vel_x = self.vel_y = self.vel_z = 0
        self.vel = 0
        self.accel = 0
        self.vel_curr = 0 
        self.time_curr = 0

    def calc_velocity(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    def calc_accel(self, curr_v, curr_t, prev_v, prev_t ):
        curr_accel = (curr_v - prev_v)/(curr_t - prev_t)
        return curr_accel

    def pos_vel_cb(self, value):
        global first_time_step, vel_prev, time_prev, accel_data 
        accel_data = Float64()
        if first_time_step:
            unix_time = time.time()
            self.vel_x = value.twist.twist.linear.x 
            self.vel_y = value.twist.twist.linear.y
            self.vel_z = value.twist.twist.linear.z
            self.vel = self.calc_velocity(self.vel_x, self.vel_y, self.vel_z)
            vel_prev = self.vel
            time_prev = unix_time
            self.accel = 0
            first_time_step = False
            accel_data.data = self.accel
            self.pub.publish(accel_data)
        else:
            unix_time = time.time()
            self.vel_x = value.twist.twist.linear.x 
            self.vel_y = value.twist.twist.linear.y
            self.vel_z = value.twist.twist.linear.z
            self.vel = self.calc_velocity(self.vel_x, self.vel_y, self.vel_z)
            self.time_curr = unix_time
            self.vel_curr = self.vel
            self.accel = self.calc_accel(self.vel_curr, self.time_curr, vel_prev, time_prev)
            accel_data.data = self.accel
            self.pub.publish(accel_data)
            vel_prev = self.vel_curr
            time_prev = self.time_curr

            cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}\r'.format(self.vel_curr, self.time_curr, vel_prev, time_prev)

            sys.stdout.write(cur_output)
            sys.stdout.flush()
            # print(self.vel_curr, self.time_curr, vel_prev, time_prev)


if __name__ == '__main__':
    rospy.init_node('acceleration')
    accel_calc = AccelerationCalculation()
    rospy.spin()