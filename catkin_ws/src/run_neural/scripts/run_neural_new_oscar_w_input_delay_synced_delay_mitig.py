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

        self.joy_pub = rospy.Publisher(config_dc['vehicle_control_topic'], Control, queue_size = 10)

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
    print('steer \tthrt: \tbrake \tvelocity')

    # use_predicted_throttle = True if config['num_outputs'] == 2 else False
    
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    exec_time_file_path = '/home/aws/Action_modulation/' + str(time_stamp) + '.csv'
    file = open(exec_time_file_path, "w+")


    in_delay_buffer = []
    delay_print_flag = 1
    while not rospy.is_shutdown():
        start_time = time.time()
        if neural_control.image_processed is False:
            continue
        
        sim_time = rospy.Time.now()
        # When the simulation starts the sim_time is around 08:01:00 and when the first lap 
        # is complete the sim time is around 08:04:00 or 08:05:00
        # we want the delay to start after the first lap.
        delay_time_trigger_in_secs = 8 * 60 * 60 + 1 * 60 
        input_tuple = (neural_control.image, neural_control.velocity, neural_control.acceleration)
        if config_rn['delay'] == True and config_rn['delta'] > 0: 
            # if delay_print_flag == 1:
            #     print("Delay triggered! ...")
            #     delay_print_flag = 0
            ## ADDING INPUT DELAY based on timestamps not atual time delay (frames not seconds)
            # Append the current input to the in_delay_buffer
            # Also the delay is only applied after 10 minutes of simulation because the 
            # in the first lap the vehicle is slow
            in_delay_buffer.append(input_tuple)
            # make sure the delay length buffer does not exceed delta
            if len(in_delay_buffer) > config_rn['delta']+1:
                in_delay_buffer.pop(0)
            # If delay_buffer size exceeds delta, pop the oldest element
            if len(in_delay_buffer) > config_rn['delta']:# and neural_control.velocity >= config_rn['velocity_0']: # sim_time.to_sec() > delay_time_trigger_in_secs:
                if delay_print_flag == 1:
                    print("Delay triggered! ...")
                    delay_print_flag = 0
                # pop the first element in the array which represents joy_data(t-delta)
                input_tuple = in_delay_buffer.pop(0)
            else:
                # this is for the first delta(number) frames so that we don't add zeros in the delay buffer.
                print("nodelay1 ...")
                input_tuple = input_tuple     
        else:
            print("nodelay2 ...")
            input_tuple = input_tuple
            

        # predicted action_data from an input image with/without velocity and acceleration
        if config['num_inputs'] == 3:
            if config_rn['delay'] == False:
                prediction = neural_control.drive.run(input_tuple)
                joy_data.steer = float(prediction[0][0])
                throttle_brake = float(prediction[1][0])
                if throttle_brake >=0.0:
                    joy_data.throttle = throttle_brake
                    joy_data.brake = 0.0
                else:
                    joy_data.throttle = 0.0
                    joy_data.brake = -1*throttle_brake
            else:
                if config_rn['mlp_all'] == True:
                    # set default throttle & brake (these will only be needed when speed is between min and max)
                    # otherwise the simple controller below will take care of throttle and brake.
                    # joy_data.throttle = config_rn['throttle_default']
                    # joy_data.brake = 0.0
                    # steering
                    prediction,prediction_base = neural_control.drive.run(input_tuple)

                    if config_rn['delta'] == 0 or neural_control.velocity < config_rn['velocity_0']:
                        joy_data.steer = float(prediction_base[0][0])
                    elif config_rn['delta'] == 9:
                        joy_data.steer = float(prediction[0][0])
                    elif config_rn['delta'] == 12:
                        joy_data.steer = float(prediction[1][0])
                    elif config_rn['delta'] == 15:
                        joy_data.steer = float(prediction[2][0])
                    elif config_rn['delta'] == 18:
                        joy_data.steer = float(prediction[3][0])
                    elif config_rn['delta'] == 21:
                        joy_data.steer = float(prediction[4][0])
                
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

        # to prevent the vehicle from stopping completely or driving at very low speed.
        # driving at very low speed will cancel out the delay effect.
        # if velocity < 18.0:
        #     joy_data.throttle = 1.0
        #     joy_data.brake = 0.0
            
        # if config_rn['delay'] == False:
        #     # time.sleep(0.1)
        #     joy_pub.publish(joy_data)     
        # else:g
        #     ## ADDING DELAY based on timestamps not atual time delay (frames not seconds)
        #     # Append the current joy_data to the delay_buffer
        #     delay_buffer.append(joy_data)
        #     # If delay_buffer size exceeds delta, pop the oldest element
        #     if len(delay_buffer) > config_rn['delta']:
        #         # pop the first element in the array which represents joy_data(t-delta)
        #         delayed_joy_data = delay_buffer.pop(0)
        #         # Publish the delayed joy_data
        #         joy_pub.publish(delayed_joy_data)
        #     else:
        #         # this is for the first delta(number) frames so that we don't add zeros in the delay buffer.
        #         joy_pub.publish(joy_data)  
        neural_control.joy_pub.publish(joy_data) 
        ##############################    
        ## publish mavros control topic
                
        if config_dc['vehicle_name'] == 'rover':
            joy_data4mavros = Twist()
            if neural_control.braking is True:
                joy_data4mavros.linear.x = 0
                joy_data4mavros.linear.y = 0
            else: 
                joy_data4mavros.linear.x = joy_data.throttle*config_rn['scale_factor_throttle']
                joy_data4mavros.linear.y = joy_data.steer*config_rn['scale_factor_steering']

            joy_pub4mavros.publish(joy_data4mavros)

        execution_time = time.time() - start_time
        file.write(str(execution_time)+'\r\n')
        
        ## print out
        # if config_rn['delay'] == False:
        #     cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}\r'.format(joy_data.steer, 
        #                   joy_data.throttle, joy_data.brake, velocity)
        # else:
        #     cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}\r'.format(delayed_joy_data.steer, 
        #                   delayed_joy_data.throttle, delayed_joy_data.brake, velocity)
        cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}\r'.format(joy_data.steer, 
                          joy_data.throttle, joy_data.brake, neural_control.velocity)
        sys.stdout.write(cur_output)
        sys.stdout.flush()
        
        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()
        


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            exit('Usage:\n$ rosrun run_neural run_neural.py weight_file_name base_model_path')

        main(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
        
