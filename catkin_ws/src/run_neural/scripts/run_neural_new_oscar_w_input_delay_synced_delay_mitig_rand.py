#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import threading 
import cv2
import time
import rospy
import numpy as np
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from acceleration.msg import acceleration_msg
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Header
import random
import csv
import math

import gpu_options

import sys
import os
import datetime

import const
from image_converter import ImageConverter
from drive_run_new_oscar_delay_mitig import DriveRun
from config_delay_mitig import Config
from image_process import ImageProcess

config = Config.neural_net
config_rn = Config.run_neural
config_dc = Config.data_collection

if config_dc['vehicle_name'] == 'fusion':
    from fusion.msg import Control
elif config_dc['vehicle_name'] == 'rover':
    from geometry_msgs.msg import Twist
    from rover.msg import Control
else:
    exit(config_dc['vehicle_name'] + 'not supported vehicle.')


# velocity = 0
# acceleration = 0

class NeuralControl:
    def __init__(self, weight_file_name, base_model_path):
        
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.rate = rospy.Rate(30) #30
        self.drive = DriveRun(weight_file_name, base_model_path)

        # Publisher for vehicle control
        self.joy_pub = rospy.Publisher(config_dc['vehicle_control_topic'], Control, queue_size=10)
        # Publisher for delta values
        self.delta_pub = rospy.Publisher(config_dc['delta_value'], Int32, queue_size=10)

        # Create subscribers for synchronized messages
        pose_sub = Subscriber(config_dc['base_pose_topic'], Odometry)
        accel_sub = Subscriber(config_dc['accel'], acceleration_msg)
        image_sub = Subscriber(config_dc['camera_image_topic'], Image)
        # vel_sub = rospy.Subscriber(config_dc['velocity_topic'], YOUR_VELOCITY_MSG_TYPE)  # Replace with the actual message type

        # Use ApproximateTimeSynchronizer to synchronize messages
        sync_subs = [image_sub, pose_sub, accel_sub]
        self.ts = ApproximateTimeSynchronizer(sync_subs, queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synced_callback)

        self.image = None
        self.velocity = None
        self.acceleration = None
        self.image_processed = False
        self.braking = False
        self.delta = config_rn['delta']

    def synced_callback(self, image_msg, odom_msg, accel_msg):
        # Extract data from synchronized messages
        # image
        img = self.ic.imgmsg_to_opencv(image_msg)
        cropped = img[config_dc['image_crop_y1']:config_dc['image_crop_y2'],
                    config_dc['image_crop_x1']:config_dc['image_crop_x2']]
        img = cv2.resize(cropped, (config['input_image_width'],
                                config['input_image_height']))
        self.image = self.image_process.process(img)
        # velocity
        vel_x = odom_msg.twist.twist.linear.x 
        vel_y = odom_msg.twist.twist.linear.y
        vel_z = odom_msg.twist.twist.linear.z
        self.velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        # acceleration
        self.acceleration = accel_msg.acceleration  

        # This is for CNN-LSTM net models
        if config['lstm'] is True:
            self.image = np.array(self.image).reshape(1,
                                config['input_image_height'],
                                config['input_image_width'],
                                config['input_image_depth'])
        self.image_processed = True

        ##############
        
    def _timer_cb(self):
        self.braking = False

    def apply_brake(self):
        self.braking = True
        timer = threading.Timer(config_rn['brake_apply_sec'], self._timer_cb) 
        timer.start()
        
def main(weight_file_name, base_model_path):

    gpu_options.set()

    # ready for neural network
    neural_control = NeuralControl(weight_file_name, base_model_path)
    
    # ready for /bolt topic publisher
    # joy_pub = rospy.Publisher(config_dc['vehicle_control_topic'], Control, queue_size = 1)
    joy_data = Control()
    # joy_data.header.frame_id = ""
    # if config_rn['delay'] == True:
    #     delayed_input = 
    if config_dc['vehicle_name'] == 'rover':
        joy_pub4mavros = rospy.Publisher(Config.config['mavros_cmd_vel_topic'], Twist, queue_size=20)

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    print('steer \tthrt \tbrake \tvelocity \tdelta')

    # use_predicted_throttle = True if config['num_outputs'] == 2 else False
    
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    # exec_time_file_path = '/home/aws/Action_modulation/' + str(time_stamp) + '.csv'
    # file = open(exec_time_file_path, "w+")
    #/home/aws/oscar/e2e_fusion_data/delay_mitig_ai/delay_mitigation_MLP_delta_all_normalize_steer0_true_shuffle_data_true_1st_try/'

 
    in_delay_buffer = []
    delay_print_flag = 1
    # delta = config_rn['delta']
    delta_list = [9, 12, 15, 18, 21]
    frame_counter = 0
    delta_values = []

    while not rospy.is_shutdown():
        start_time = time.time()
        if neural_control.image_processed is False:
            continue
        
        input_tuple = (neural_control.image, neural_control.velocity, neural_control.acceleration)
        
        # Check if random delta is enabled
        if config_rn.get('delta_rand', False):
            # Generate a new random delta between 9 and 18 every 60 frames (approx every one second - we have 60-62 FPS)
            if frame_counter % 60 == 0: 
                neural_control.delta = random.randint(9, 21)
                delta_values.append(neural_control.delta)
        else:
            neural_control.delta = config_rn['delta']
        
        if config_rn['delay'] and neural_control.delta > 0: 
            in_delay_buffer.append(input_tuple)
            if len(in_delay_buffer) > neural_control.delta + 1:
                in_delay_buffer.pop(0)
            
            if len(in_delay_buffer) > neural_control.delta:
                if delay_print_flag == 1:
                    print("Delay triggered! ...")
                    delay_print_flag = 0
                input_tuple = in_delay_buffer.pop(0)
        #     else:
        #         print("nodelay1 ...")
        # else:
        #     print("nodelay2 ...")

        if config['num_inputs'] == 3:
            if config_rn['delay'] == False:
                prediction = neural_control.drive.run(input_tuple)
                joy_data.steer = float(prediction[0][0])
                throttle_brake = float(prediction[1][0])
                if throttle_brake >= 0.0:
                    joy_data.throttle = throttle_brake
                    joy_data.brake = 0.0
                else:
                    joy_data.throttle = 0.0
                    joy_data.brake = -1 * throttle_brake
            else:
                if config_rn['mlp_all'] == True:
                    prediction, prediction_base = neural_control.drive.run(input_tuple)

                    if neural_control.delta == 0 or neural_control.velocity < config_rn['velocity_0']:
                        joy_data.steer = float(prediction_base[0][0])
                    elif neural_control.delta in delta_list:
                        index = delta_list.index(neural_control.delta)
                        joy_data.steer = float(prediction[index][0])
                    else:
                        # Perform interpolation
                        lower_delta = max(d for d in delta_list if d < neural_control.delta)
                        upper_delta = min(d for d in delta_list if d > neural_control.delta)
                        lower_index = delta_list.index(lower_delta)
                        upper_index = delta_list.index(upper_delta)
                        lower_prediction = float(prediction[lower_index][0])
                        upper_prediction = float(prediction[upper_index][0])
                        joy_data.steer = lower_prediction + (neural_control.delta - lower_delta) / (upper_delta - lower_delta) * (upper_prediction - lower_prediction)
                
        elif config['num_inputs'] == 2:
            prediction = neural_control.drive.run((neural_control.image, neural_control.velocity))
            if config['num_outputs'] == 2:
                # prediction is [ [] ] numpy.ndarray
                joy_data.steer = prediction[0][0]
                joy_data.throttle = prediction[0][1]
            else: # num_outputs is 1
                joy_data.steer = prediction[0][0]
        else: # num_inputs is 1
            prediction = neural_control.drive.run((neural_control.image, ))
            if config['num_outputs'] == 2:
                # prediction is [ [] ] numpy.ndarray
                joy_data.steer = prediction[0][0]
                joy_data.throttle = prediction[0][1]
            else: # num_outputs is 1
                joy_data.steer = prediction[0][0]
            
        #############################
        ## very very simple controller
        ## 

        is_sharp_turn = False
        # if brake is not already applied and sharp turn
        if neural_control.braking is False: 
            if neural_control.velocity < config_rn['velocity_0']: # too slow then no braking
                joy_data.throttle = config_rn['throttle_default'] # apply default throttle
                joy_data.brake = 0
            elif abs(joy_data.steer) > config_rn['sharp_turn_min']:
                is_sharp_turn = True
            
            if is_sharp_turn or neural_control.velocity > config_rn['max_vel']: 
                # joy_data.throttle = config_rn['throttle_sharp_turn']
                joy_data.brake = config_rn['brake_val']
                joy_data.throttle = 0
                neural_control.apply_brake()
            else:
                # if use_predicted_throttle is False:
                joy_data.throttle = config_rn['throttle_default']
                joy_data.brake = 0
        else:
            joy_data.throttle = 0
        ################################

        neural_control.delta_pub.publish(neural_control.delta)
        neural_control.joy_pub.publish(joy_data) 

        # file.write(str(neural_control.delta)+'\r\n')
        ## print out
        # if config_rn['delay'] == False:
        #     cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}\r'.format(joy_data.steer, 
        #                   joy_data.throttle, joy_data.brake, velocity)
        # else:
        #     cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}\r'.format(delayed_joy_data.steer, 
        #                   delayed_joy_data.throttle, delayed_joy_data.brake, velocity)
        cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f} \t{4:.0f}msec\r'.format(joy_data.steer, 
                          joy_data.throttle, joy_data.brake, neural_control.velocity, neural_control.delta)
        sys.stdout.write(cur_output)
        sys.stdout.flush()
        
        # Increment frame counter
        frame_counter += 1

        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()
        
    # # After rosshutdown, write delay_values to a CSV file
    # model_folder_path = os.path.dirname(weight_file_name.rstrip('/'))
    # delta_values_file_path = model_folder_path + '/delay_mitig_drive_data/' + str(time_stamp) + '.csv'
    # with open(delta_values_file_path, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     # csv_writer.writerow(['Delay Values'])
    #     for value in delta_values:
    #         csv_writer.writerow([value])
    # print('delay values saved!')

if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            exit('Usage:\n$ rosrun run_neural run_neural.py weight_file_name base_model_path')

        main(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
        
