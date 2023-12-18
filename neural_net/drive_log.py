#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:07:31 2019
History:
11/28/2020: modified for OSCAR 
12/17/2023: modified by Aws
"""

import cv2
import numpy as np
#import keras
#import sklearn
#import resnet
from progressbar import ProgressBar
import matplotlib.pyplot as plt

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess

###############################################################################
#

def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

def is_min_max_normalized(data):
    return np.all((data >= 0) & (data <= 1))

class DriveLog:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
       
    def __init__(self, model_path, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash+1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path

        csv_path = data_path + '/' + model_name + const.DATA_EXT   
        
        self.data_path = data_path
        self.data = DriveData(csv_path)
        self.data = self.data
        self.test_generator = None
        
        self.num_test_samples = 0        
        #self.config = Config()
        
        self.net_model = NetModel(model_path)
        self.net_model.load()
        self.model_path = model_path
        
        self.image_process = ImageProcess()

        self.steering_measurements = []
        self.steering_predictions = []
        self.steering_differences = []
        self.steering_squared_differences = []

        # if Config.neural_net['accel']:
        self.throttle_measurements = []
        self.throttle_predictions = []
        self.throttle_differences = []
        self.throttle_squared_differences = []

        self.brake_measurements = []
        self.brake_predictions = []
        self.brake_differences = []
        self.brake_squared_differences = []


    ###########################################################################
    #
    def _prepare_data(self):
        
        self.data.read(normalize = False)

        # Normalize velocity and acceleration
        self.data.velocities = min_max_scaling(self.data.velocities)
        if is_min_max_normalized(self.data.velocities):
            print("Velocity data is Min-Max Normalized")
        self.data.calculated_accelerations = min_max_scaling(self.data.calculated_accelerations)
        if is_min_max_normalized(self.data.calculated_accelerations):
            print("Acceleration data is Min-Max Normalized")

        self.test_data = list(zip(self.data.image_names, self.data.velocities, self.data.calculated_accelerations, 
                                  self.data.actions))
        self.num_test_samples = len(self.test_data)
        
        print('Test samples: {0}'.format(self.num_test_samples))

    
   ###########################################################################
    #
    def _savefigs(self, plt, filename):
        plt.savefig(filename + '.png', dpi=150)
        plt.savefig(filename + '.pdf', dpi=150)
        print('Saved ' + filename + '.png & .pdf.')


    ###########################################################################
    #
    def _plot_results(self):
        # plt.figure()
        # # Plot a histogram of the prediction errors
        # num_bins = 25
        # hist, bins = np.histogram(self.steering_differences, num_bins)
        # center = (bins[:-1]+ bins[1:]) * 0.5
        # plt.bar(center, hist, width=0.05)
        # #plt.title('Historgram of Predicted Errors')
        # # if Config.neural_net['accel']:


        # all three histograms on one figure
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        num_bins = 25
        # Plot steering angle histogram with transparent bars
        hist, bins = np.histogram(self.steering_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, color='b', label='Steering')
        # Plot throttle histogram with transparent bars
        hist, bins = np.histogram(self.throttle_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, color='g', label='Throttle')
        # Plot brake histogram with transparent bars
        hist, bins = np.histogram(self.brake_differences, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05, alpha=0.5, color='r', label='Brake')
        # title and labels
        ax1.set_title("Histograms of Prediction Errors")
        ax1.set_xlabel('Values')
        ax1.set_ylabel('Samples')
        ax1.legend()
        ax1.grid(True)
        # plt.show()
        self._savefigs(plt, self.data_path + '_str_thr_brk_err_hist')



        #######################################################################################
        # STEERING
        #######################################################################################

        # plt.xlabel('Steering Angle')
        # plt.ylabel('Number of Predictions')
        # plt.xlim(-1.0, 1.0)
        # plt.plot(np.min(self.differences), np.max(self.differences))
        # plt.tight_layout()
        # self._savefigs(plt, self.data_path + '_err_hist')

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.steering_measurements, self.steering_predictions)
        #plt.title('Scatter Plot of Errors')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.steering_measurements)
        plt.plot(self.steering_predictions)
        mean = sum(self.steering_differences)/len(self.steering_differences)
        variance = sum([((x - mean) ** 2) for x in self.steering_differences]) / len(self.steering_differences) 
        std = variance ** 0.5
        # print("mean:", mean)
        # print("std:", std)
        # plt.title('MAE: {0:.3f}, STDEV: {1:.3f}'.format(mean, std))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([-1.0, 1.0])
        plt.xlabel('Time Step')
        plt.ylabel('Steering Angle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_steering_comparison')

        #######################################################################################
        # THROTTLE
        #######################################################################################

        # plt.xlabel('Throttle')
        # plt.ylabel('Number of Predictions')
        # plt.xlim(0.0, 1.0)
        # plt.plot(np.min(self.throttle_differences), np.max(self.throttle_differences))
        # plt.tight_layout()
        # self._savefigs(plt, self.data_path + '_thr_err_hist')

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.throttle_measurements, self.throttle_predictions)
        plt.title('Scatter Plot of Throttle Errors')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot([0.0, 1.0], [0.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_thr_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.throttle_measurements)
        plt.plot(self.throttle_predictions)
        mean_thr = sum(self.throttle_differences)/len(self.throttle_differences)
        variance_thr = sum([((x - mean_thr) ** 2) for x in self.throttle_differences]) / len(self.throttle_differences) 
        std_thr = variance_thr ** 0.5
        # plt.title('thr_MAE: {0:.3f}, thr_STDEV: {1:.3f}'.format(mean_thr, std_thr))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([0.0, 1.0])
        plt.xlabel('Time Step')
        plt.ylabel('Throttle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_thr_comparison')

        #######################################################################################
        # BRAKE
        #######################################################################################

        # plt.xlabel('Brake')
        # plt.ylabel('Number of Predictions')
        # plt.xlim(0.0, 1.0)
        # plt.plot(np.min(self.brake_differences), np.max(self.brake_differences))
        # plt.tight_layout()
        # self._savefigs(plt, self.data_path + '_brk_err_hist')

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.brake_measurements, self.brake_predictions)
        plt.title('Scatter Plot of Brake Errors')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot([0.0, 1.0], [0.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_brk_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.brake_measurements)
        plt.plot(self.brake_predictions)
        mean_brk = sum(self.brake_differences)/len(self.brake_differences)
        variance_brk = sum([((x - mean_brk) ** 2) for x in self.brake_differences]) / len(self.brake_differences) 
        std_brk = variance_brk ** 0.5
        # plt.title('brk_MAE: {0:.3f}, brk_STDEV: {1:.3f}'.format(mean_brk, std_brk))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([0.0, 1.0])
        plt.xlabel('Time Step')
        plt.ylabel('Brake')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_brk_comparison')

            
        # show all figures
        plt.show()


   ###########################################################################
    #
    def run(self):
        
        self._prepare_data()
        #fname = self.data_path + const.LOG_EXT
        fname = self.data_path + const.LOG_EXT # use model name to save log
        
        file = open(fname, 'w')

        #print('image_name', 'label', 'predict', 'abs_error')
        bar = ProgressBar()

        # log file header
        if Config.neural_net['num_inputs'] == 3:
            file.write('image_name, velocity, acceleration, \
                    label_steering_angle, predicted_steering_angle, abs_steering_error, squared_steering_error,\
                    label_throttle, predicted_throttle, abs_throttle_error, squared_throttle_error,\
                    label_brake, predicted_brake, abs_brake_error, squared_brake_error \n')
        elif Config.neural_net['num_inputs'] == 2:
            file.write('image_name, velocity, \
                    label_steering_angle, predicted_steering_angle, abs_steering_error, squared_steering_error,\
                    label_throttle, predicted_throttle, abs_throttle_error, squared_throttle_error,\
                    label_brake, predicted_brake, abs_brake_error, squared_brake_error \n')

        
        # if Config.neural_net['str_out']:
        #     file.write('image_name, velocity, throttle, brake, label_steering_angle, pred_steering_angle, abs_error, squared_error\n')
        # elif Config.neural_net['thr_out']:
        #     file.write('image_name, velocity, steering, brake, label_throttle, pred_throttle, abs_error, squared_error\n')
        # elif Config.neural_net['brk_out']:
        #     file.write('image_name, velocity, steering, throttle, label_brake, pred_brake, abs_error, squared_error\n')
        # elif Config.neural_net['accel']:
        #     file.write('image_name, velocity, acceleration, label_throttle, label_brake, \
        #                pred_throttle, pred_brake, thr_abs_error, thr_squared_error, brk_abs_error, brk_squared_error\n')

        if Config.neural_net['lstm'] is True:
            images = []
            #images_names = []
            cnt = 1

            for image_name, velocity, measurement in bar(self.test_data):   
                image_fname = self.data_path + '/' + image_name
                image = cv2.imread(image_fname)

                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if Config.data_collection['crop'] is not True:
                    image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

                image = cv2.resize(image, (Config.neural_net['input_image_width'],
                                        Config.neural_net['input_image_height']))
                image = self.image_process.process(image)

                images.append(image)
                #images_names.append(image_name)
                
                if cnt >= Config.neural_net['lstm_timestep']:
                    trans_image = np.array(images).reshape(-1, Config.neural_net['lstm_timestep'], 
                                                Config.neural_net['input_image_height'],
                                                Config.neural_net['input_image_width'],
                                                Config.neural_net['input_image_depth'])                    

                    predict = self.net_model.model.predict(trans_image)
                    pred_steering_angle = predict[0][0]
                    pred_steering_angle = pred_steering_angle / Config.neural_net['steering_angle_scale']
                
                    if Config.neural_net['num_outputs'] == 2:
                        pred_throttle = predict[0][1]
                    
                    label_steering_angle = measurement[0] # labeled steering angle
                    self.measurements.append(label_steering_angle)
                    self.predictions.append(pred_steering_angle)
                    diff = abs(label_steering_angle - pred_steering_angle)
                    self.differences.append(diff)
                    self.squared_differences.append(diff*2)
                    log = image_name+','+str(label_steering_angle)+','+str(pred_steering_angle)\
                                    +','+str(diff)\
                                    +','+str(diff**2)

                    file.write(log+'\n')
                    # delete the 1st element
                    del images[0]
        else:
            for image_name, velocity, acceleration, action in bar(self.test_data):   
                image_fname = self.data_path + '/' + image_name
                image = cv2.imread(image_fname)

                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if Config.data_collection['crop'] is not True:
                    image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

                image = cv2.resize(image, (Config.neural_net['input_image_width'],
                                        Config.neural_net['input_image_height']))
                image = self.image_process.process(image)
                
                npimg = np.expand_dims(image, axis=0)
                steering_angle, throttle, brake, throttle_brake = action 
                # print('steering type')
                # print(type(steering_angle))
                np_velocity = np.array(velocity).reshape(-1, 1)
                np_acceleration = np.array(acceleration).reshape(-1,1)
                # np_steering_angle = np.array(steering_angle).reshape(-1, 1)
                np_throttle = np.array(throttle).reshape(-1, 1)
                np_brake = np.array(brake).reshape(-1, 1)
                if Config.neural_net['num_inputs'] == 3:
                    predict = self.net_model.model.predict([npimg, np_velocity, np_acceleration])
                elif Config.neural_net['num_inputs'] == 2:
                    predict = self.net_model.model.predict([npimg, np_velocity])
                else:
                    predict = self.net_model.model.predict(npimg)

 
                pred_steering_angle = predict[0][0]
                # print('pred_steering_angle type')
                # print(type(pred_steering_angle))
                if Config.neural_net['num_outputs'] == 3:
                    pred_throttle_brake = predict[1][0]

                pred_throttle=[]
                pred_brake=[]
                for i in range(len(pred_throttle_brake)):
                    if pred_throttle_brake[i] >=0.0:
                        pred_throttle.append(pred_throttle_brake[i])
                        pred_brake.append(0.0)
                    else:
                        pred_throttle.append(0.0)
                        pred_brake.append(-1*pred_throttle_brake[i])
                # Convert lists to numpy arrays for element-wise operations
                pred_throttle = np.array(pred_throttle)
                pred_brake = np.array(pred_brake)

                # label_steering_angle = (steering_angle+1.0)/2.0    
                label_steering_angle = np.array(steering_angle)
                label_throttle = throttle
                label_brake = brake
                
                self.steering_measurements.append(label_steering_angle)
                self.steering_predictions.append(pred_steering_angle)
                diff_steering = abs(label_steering_angle - pred_steering_angle)
                self.steering_differences.append(diff_steering[0])
                # print(self.steering_differences)
                self.steering_squared_differences.append(diff_steering**2)
                
                self.throttle_measurements.append(label_throttle)
                self.throttle_predictions.append(pred_throttle)
                diff_thr = abs(label_throttle - pred_throttle)
                self.throttle_differences.append(diff_thr)
                self.throttle_squared_differences.append(diff_thr**2)
                
                self.brake_measurements.append(label_brake)
                self.brake_predictions.append(pred_brake)
                diff_brk = abs(label_brake - pred_brake)
                self.brake_differences.append(diff_brk)
                self.brake_squared_differences.append(diff_brk**2)

                # file.write('image_name, velocity, acceleration, \
                #     label_steering_angle, predicted_steering_angle, abs_steering_error, squared_steering_error,\
                #     label_throttle, predicted_throttle, abs_throttle_error, squared_throttle_error,\
                #     label_brake, predicted_brake, abs_brake_error, squared_brake_error \n')
                                                
                if Config.neural_net['num_inputs'] == 3:
                    log = image_name+','+ str(velocity) +','+ str(acceleration) + ',' \
                            + str(label_steering_angle) + ',' + str(pred_steering_angle) +',' + str(diff_steering) + ',' + str(diff_steering**2) + ',' \
                            + str(label_throttle) + ',' + str(pred_throttle) +',' + str(diff_thr) + ',' + str(diff_thr**2) + ',' \
                            + str(label_brake) + ',' + str(pred_brake) +',' + str(diff_brk) + ',' + str(diff_brk**2) 
                elif Config.neural_net['num_inputs'] == 2:
                    log = image_name+','+ str(velocity) +',' \
                            + str(label_steering_angle) + ',' + str(pred_steering_angle) +',' + str(diff_steering) + ',' + str(diff_steering**2) + ',' \
                            + str(label_throttle) + ',' + str(pred_throttle) +',' + str(diff_thr) + ',' + str(diff_thr**2) + ',' \
                            + str(label_brake) + ',' + str(pred_brake) +',' + str(diff_brk) + ',' + str(diff_brk**2) 
                file.write(log+'\n')
                
        file.close()
        print('Saved ' + fname + '.')

        self._plot_results()



# from drive_log_acc import DriveLog


###############################################################################
#       
def main(weight_name, data_folder_name):
    drive_log = DriveLog(weight_name, data_folder_name) 
    drive_log.run() # data folder path to test
       

###############################################################################
#       
if __name__ == '__main__':
    import sys

    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python {} weight_name data_folder_name'.format(sys.argv[0]))
        
        main(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
