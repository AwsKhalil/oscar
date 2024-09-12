#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:20:14 2023
History:
This is a copy of net_model.py modified for NDPIA project.

@author: Aws
# -*- coding: utf-8 -*-
"""
import sys
from keras.models import Sequential, Model, model_from_json
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, LSTM, Concatenate,concatenate 
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, BatchNormalization, Input
from keras.layers import Add, Subtract, Multiply, RepeatVector, Reshape, TimeDistributed
from keras import losses, optimizers, metrics
import keras.backend as K
import tensorflow as tf

import const_new_oscar_delay_mitig as const
from config_delay_mitig import Config


config = Config.neural_net


def str_thr_brk():
    """
    This model takes the current frame image as input
    (you can add velocity and/or acceleration too) 
    and gives the action values as output (steering, throttle, brake)
    I am trying to use this model to cancel the simple controller used in the 
    run_neural.py file. (catkin_ws/src/run_neural/run_neural.py)

    input: x(t) --> current frame image, velocity, and acceleration
    output: a(t) --> where a(t) = [str(t), thr(t), brk(t)]
    """
    ######### variables ####################
    droput_rate = 0.3

    ######### inputs shape ####################
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    vel_shape = (1,) # velocity
    acc_shape = (1,) # acceleration
    
    ######### img model ####################
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2))(lamb)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2))(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2))(conv_2)
    conv_4 = Conv2D(64, (3, 3))(conv_3)
    conv_5 = Conv2D(64, (3, 3), name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    ###### vel model #######
    vel_input = Input(shape=vel_shape)
    vel_vec = Dense(16*3*3, name='fc_vel')(vel_input)
    # rep_vel = RepeatVector(16*3*3, name='rep_vel')(vel_input)
    # vel_vec = Reshape((16*3*3,), name='rshp_vel')(rep_vel)
    ###### acc model #######
    acc_input = Input(shape=acc_shape)
    acc_vec = Dense(16*3*3, name='fc_acc')(acc_input)
    # rep_acc = RepeatVector(64*3*3, name='rep_acc')(acc_input)
    # acc_vec = Reshape((64*3*3,), name='rshp_acc')(rep_acc)

    # concatenate the inputs
    if config['num_inputs'] == 3:
        concat_inputs = concatenate([flat,vel_vec, acc_vec]) # I mistakenly used concatenate instead of Concatenate but the model is already trained and saved so I did not change it.
    elif config['num_inputs'] == 2:
        concat_inputs = concatenate([flat,vel_vec])
    else:
        concat_inputs = flat

    fc_1 = Dense(512, name='fc_1', activation='relu')(concat_inputs)
    drop0 = Dropout(droput_rate)(fc_1)
    fc_2 = Dense(100, name='fc_2', activation='relu')(drop0)
    drop1 = Dropout(droput_rate)(fc_2)
    fc_3 = Dense(50, name='fc_3', activation='relu')(drop1)
    drop2 = Dropout(droput_rate)(fc_3)
    fc_4 = Dense(10, name='fc_4', activation='relu')(drop2)
    # fc_last = Dense(3, name='fc_action', activation='sigmoid')(fc_4) # 3 --> str, thr, brk
    # fc_str = Dense(1, activation='tanh', name='fc_str')(fc_4) # Steering
    # fc_thr = Dense(1, activation='sigmoid', name='fc_thr')(fc_4) # Throttle
    # fc_brk = Dense(1, activation='sigmoid', name='fc_brk')(fc_4) # Brake
    # action = concatenate([fc_str, fc_thr,fc_brk])

    fc_str = Dense(1, activation='tanh', name='fc_str')(fc_4) # Steering
    fc_thr_brk = Dense(1, activation='tanh', name='fc_thr_brk')(fc_4) # Throttle_brake

    if config['num_inputs'] == 3:
        model = Model(inputs=[img_input,vel_input, acc_input], output=[fc_str, fc_thr_brk])
    elif config['num_inputs'] ==2:
        model = Model(inputs=[img_input,vel_input], output=[fc_str, fc_thr_brk])
    else:
        model = Model(inputs=img_input, output=[fc_str, fc_thr_brk])

    return model

def pre_trained_str_thr_brk(base_model_path):
    base_weightsfile = base_model_path+'.h5'
    base_modelfile   = base_model_path+'.json'
    
    base_json_file = open(base_modelfile, 'r')
    base_loaded_model_json = base_json_file.read()
    base_json_file.close()
    base_model = model_from_json(base_loaded_model_json)
    base_model.load_weights(base_weightsfile)
    # Set the model to be non-trainable
    base_model.trainable = False
    # print(base_model.get_layer('conv2d_3').output)
    return base_model

def multistep_lstm_delay_mitigation(base_model_path):
    """
    This model takes the current frame image as input
    (you can add velocity and/or acceleration too) 
    and gives the action values as output (steering, throttle, brake)
    I am trying to use this model to cancel the simple controller used in the 
    run_neural.py file. (catkin_ws/src/run_neural/run_neural.py)

    input: x(t) --> current frame image, velocity, and acceleration
    output: a(t) --> where a(t) = [str(t), thr(t), brk(t)]
    """
    ######### variables ####################
    # droput_rate = 0.3

    ######### inputs shape ####################
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    vel_shape = (1,) # velocity
    acc_shape = (1,) # acceleration

    ######model#######
    img_input = Input(shape=img_shape)
    vel_input = Input(shape=vel_shape)
    gvel_input = Input(shape=acc_shape)
    
    base_model1 = pre_trained_str_thr_brk(base_model_path)
    base_model2 = pre_trained_str_thr_brk(base_model_path)
    base_model3 = pre_trained_str_thr_brk(base_model_path)
    pretrained_model_last = Model(base_model1.input, base_model1.get_layer('fc_out').output, name='base_model_output')
    pretrained_model_conv3 = Model(base_model2.input, base_model2.get_layer('conv2d_3').output, name='base_model_conv2d_3')
    pretrained_model_conv5 = Model(base_model3.input, base_model3.get_layer('conv2d_last').output, name='base_model_conv2d_last')
    # if config['style_train'] is True:
    pretrained_model_conv3.trainable = False
    pretrained_model_conv5.trainable = False
    pretrained_model_last.trainable = False
        
    base_model_last_output = pretrained_model_last([img_input, vel_input])
    base_model_conv3_output = pretrained_model_conv3([img_input, vel_input])
    base_model_conv5_output = pretrained_model_conv5([img_input, vel_input])
    
    add_base_layer = Add()([base_model_conv3_output, base_model_conv5_output])
    fc_vel = Dense(100, activation='relu', name='fc_vel')(vel_input)
    fc_gvel = Dense(100, activation='relu', name='fc_gvel')(gvel_input)
    fc_base_out = Dense(100, activation='relu', name='fc_base_out')(base_model_last_output)
    flat = Flatten()(add_base_layer)
    fc_1 = Dense(500, activation='relu', name='fc_1')(flat)
    conc = Concatenate()([fc_base_out, fc_1, fc_vel, fc_gvel])
    fc_2 = Dense(200, activation='relu', name='fc_2')(conc)
    drop = Dropout(rate=0.2)(fc_2)
    fc_3 = Dense(100, activation='relu', name='fc_3')(drop)
    
    # print(base_model_last_output[0].shape)
    if config['only_thr_brk'] is True: 
        fc_out = Dense(config['num_outputs']-1, name='fc_out')(fc_3)
    else:
        fc_out = Dense(config['num_outputs'], name='fc_out')(fc_3)
    # fc_str = Dense(1, name='fc_str')(base_str)
    # fc_thr = Dense(1, name='fc_thr')(fc_3)
    # fc_brk = Dense(1, name='fc_brk')(fc_3)
    
    model = Model(inputs=[img_input, vel_input, gvel_input], outputs=[fc_out])
    # model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_thr, fc_brk])
    return model

def conditional_MLP_delay_mitigation(base_model_path):
    """
    Conditional Imitation learning model.
    This model takes the following as input:
    [the current frame image, the current frame velocity, the current frame acceleration]
    The pretrained base model gives the following output (steering, throttle, brake)
    Then the base model output along with the concatenated image vector, velocity vector, and acceleration vector
    are fed to the delay mitigation block which will predict future control values. 
    I am trying to use this model to mitigate the input delay.

    input: x(t) --> current frame image, velocity, and acceleration
    output: a(t+delta) --> where a(t+delta) = [str(t+delta), thr(t+delta), brk(t+delta)]
    """
    ######### variables ####################
    # droput_rate = 0.3

    ######### inputs shape ####################
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    vel_shape = (1,) # velocity
    acc_shape = (1,) # acceleration

    ######model#######
    img_input = Input(shape=img_shape)
    vel_input = Input(shape=vel_shape)
    acc_input = Input(shape=acc_shape)
    
    base_model1 = pre_trained_str_thr_brk(base_model_path)
    base_model2 = pre_trained_str_thr_brk(base_model_path)
    base_model3 = pre_trained_str_thr_brk(base_model_path)
    
    pretrained_model_concat = Model(base_model1.input, base_model1.get_layer('concatenate_1').output, name='base_model_concat')
    pretrained_model_str = Model(base_model2.input, base_model2.get_layer('fc_str').output, name='base_model_out_str')
    pretrained_model_thr_brk = Model(base_model3.input, base_model3.get_layer('fc_thr_brk').output, name='base_model_out_thr_brk')
    # if config['style_train'] is True:
    pretrained_model_concat.trainable = False
    pretrained_model_str.trainable = False
    pretrained_model_thr_brk.trainable = False
        
    base_model_concat_output = pretrained_model_concat([img_input, vel_input, acc_input])
    base_model_str_output = pretrained_model_str([img_input, vel_input, acc_input])
    base_model_thr_brk_output = pretrained_model_thr_brk([img_input, vel_input, acc_input])
    
    # add_base_layer = Add()([base_model_conv3_output, base_model_conv5_output])
    # fc_vel = Dense(100, activation='relu', name='fc_vel')(vel_input)
    # fc_gvel = Dense(100, activation='relu', name='fc_gvel')(gvel_input)
    fc_base_concat = Dense(500, activation='relu', name='fc_base_concat')(base_model_concat_output)
    fc_base_str = Dense(100, activation='relu', name='fc_base_str')(base_model_str_output)
    fc_base_thr_brk = Dense(100, activation='relu', name='fc_base_thr_brk')(base_model_thr_brk_output)
    # flat = Flatten()(add_base_layer)
    # fc_1 = Dense(500, activation='relu', name='fc_1')(flat)
    conc = Concatenate()([fc_base_concat, fc_base_str, fc_base_thr_brk])
    fc_2 = Dense(200, activation='relu', name='fc_2')(conc)
    drop = Dropout(rate=0.2)(fc_2)
    fc_3 = Dense(100, activation='relu', name='fc_3')(drop)
    
    fc_out_steer = Dense(1, name='fc_out_steer')(fc_3)
    fc_out_thr_brk = Dense(1, name='fc_out_thr_brk')(fc_3)
    
    model = Model(inputs=[img_input, vel_input, acc_input], outputs=[fc_out_steer, fc_out_thr_brk])
    # model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_thr, fc_brk])
    return model

def conditional_MLP_delay_mitigation_all(base_model_path):
    """
    Conditional Imitation learning model.
    The difference between this and the one above is:
    1) no longer need acc as input to the delay mitigation block
    2) the delay mitigation block will be trained to predict steering only 
    3) the delay mitigation block output will have separate heads, one for each delta value.
    This model takes the following as input:
    [the current frame image, the current frame velocity, the current frame acceleration]
    The pretrained base model gives the following output (steering, throttle, brake)
    Then the base model steering output along with the concatenated image vector, and velocity vector
    are fed to the delay mitigation block which will predict future control values. 
    I am trying to use this model to mitigate the input delay.

    input: x(t) --> current frame image, velocity, and acceleration
    output: str(t+delta) --> where str(t+delta) = [str(t+9), str(t+12), str(t+15), str(t+18), ...]
    """
    ######### variables ####################
    # droput_rate = 0.3

    ######### inputs shape ####################
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    vel_shape = (1,) # velocity
    acc_shape = (1,) # acceleration

    ######model#######
    img_input = Input(shape=img_shape)
    vel_input = Input(shape=vel_shape)
    if config['num_inputs']==3:
        acc_input = Input(shape=acc_shape)
    
    base_model1 = pre_trained_str_thr_brk(base_model_path)
    base_model2 = pre_trained_str_thr_brk(base_model_path)
    base_model3 = pre_trained_str_thr_brk(base_model_path)
    
    pretrained_model_img_vec = Model(base_model1.input, base_model1.get_layer('flatten_1').output, name='base_model_img_vec')
    pretrained_model_vel = Model(base_model3.input, base_model3.get_layer('fc_vel').output, name='base_model_vel_vec')
    pretrained_model_str = Model(base_model2.input, base_model2.get_layer('fc_str').output, name='base_model_out_str')
    
    # if config['style_train'] is True:
    pretrained_model_img_vec.trainable = False
    pretrained_model_vel.trainable = False
    pretrained_model_str.trainable = False
        
    if config['num_inputs']==3:
        base_model_img_vec = pretrained_model_img_vec([img_input, vel_input, acc_input])
        base_model_vel_vec = pretrained_model_vel([img_input, vel_input, acc_input])
        base_model_str_output = pretrained_model_str([img_input, vel_input, acc_input])
    elif config['num_inputs']==2:
        base_model_img_vec = pretrained_model_img_vec([img_input, vel_input])
        base_model_vel_vec = pretrained_model_vel([img_input, vel_input])
        base_model_str_output = pretrained_model_str([img_input, vel_input])

    fc_base_img = Dense(500, activation='relu', name='fc_base_img')(base_model_img_vec)
    fc_base_vel = Dense(100, activation='relu', name='fc_base_vel')(base_model_vel_vec)
    fc_base_str = Dense(100, activation='relu', name='fc_base_str')(base_model_str_output)

    conc = Concatenate()([fc_base_img, fc_base_vel, fc_base_str])

    # delta = 0
    # then use the base model steering output    

    # delta = 9
    fc_2 = Dense(200, activation='relu', name='fc_2')(conc)
    drop1 = Dropout(rate=0.3)(fc_2)
    fc_3 = Dense(100, activation='relu', name='fc_3')(drop1)
    drop_d9 = Dropout(rate=0.3)(fc_3)
    fc_d9 = Dense(50, activation='tanh', name='fc_d9')(drop_d9)
    fc_out_steer9 = Dense(1, activation='tanh', name='fc_out_steer9')(fc_d9)

    # delta = 12
    fc_4 = Dense(200, activation='relu', name='fc_4')(conc)
    drop2 = Dropout(rate=0.3)(fc_4)
    fc_5 = Dense(100, activation='relu', name='fc_5')(drop2)
    drop_d12 = Dropout(rate=0.3)(fc_5)
    fc_d12 = Dense(50, activation='tanh', name='fc_d12')(drop_d12)
    fc_out_steer12 = Dense(1, activation='tanh', name='fc_out_steer12')(fc_d12)

    # delta = 15
    fc_6 = Dense(200, activation='relu', name='fc_6')(conc)
    drop3 = Dropout(rate=0.3)(fc_6)
    fc_7 = Dense(100, activation='relu', name='fc_7')(drop3)
    drop_d15 = Dropout(rate=0.3)(fc_7)
    fc_d15 = Dense(50, activation='tanh', name='fc_d15')(drop_d15)
    fc_out_steer15 = Dense(1, activation='tanh', name='fc_out_steer15')(fc_d15)

    # delta = 18
    fc_8 = Dense(200, activation='relu', name='fc_8')(conc)
    drop4 = Dropout(rate=0.3)(fc_8)
    fc_9 = Dense(100, activation='relu', name='fc_9')(drop4)
    drop_d18 = Dropout(rate=0.3)(fc_9)
    fc_d18 = Dense(50, activation='tanh', name='fc_d18')(drop_d18)
    fc_out_steer18 = Dense(1, activation='tanh', name='fc_out_steer18')(fc_d18)

    # delta = 21
    fc_10 = Dense(200, activation='relu', name='fc_10')(conc)
    drop5 = Dropout(rate=0.3)(fc_10)
    fc_11 = Dense(100, activation='relu', name='fc_11')(drop5)
    drop_d21 = Dropout(rate=0.3)(fc_11)
    fc_d21 = Dense(50, activation='tanh', name='fc_d21')(drop_d21)
    fc_out_steer21 = Dense(1, activation='tanh', name='fc_out_steer21')(fc_d21)
    
    # model = Model(inputs=[img_input, vel_input, acc_input], 
    #               outputs=[base_model_str_output, fc_out_steer9, fc_out_steer12, 
    #                        fc_out_steer15, fc_out_steer18, fc_out_steer21])
    if config['num_inputs']==3:
        model = Model(inputs=[img_input, vel_input, acc_input], 
                      outputs=[fc_out_steer9, fc_out_steer12, 
                      fc_out_steer15, fc_out_steer18, fc_out_steer21])
    elif config['num_inputs']==2:
        model = Model(inputs=[img_input, vel_input], 
                      outputs=[fc_out_steer9, fc_out_steer12, 
                      fc_out_steer15, fc_out_steer18, fc_out_steer21])

    return model

def conditional_MLP_delay_mitigation_all_2(base_model_path): 
    """
    # this model use separate concatenated input to the mitigation block (each delta head gets its own concatenated input)
    Conditional Imitation learning model.
    The difference between this and the one above is:
    1) no longer need acc as input to the delay mitigation block
    2) the delay mitigation block will be trained to predict steering only 
    3) the delay mitigation block output will have separate heads, one for each delta value.
    This model takes the following as input:
    [the current frame image, the current frame velocity, the current frame acceleration]
    The pretrained base model gives the following output (steering, throttle, brake)
    Then the base model steering output along with the concatenated image vector, and velocity vector
    are fed to the delay mitigation block which will predict future control values. 
    I am trying to use this model to mitigate the input delay.

    input: x(t) --> current frame image, velocity, and acceleration
    output: str(t+delta) --> where str(t+delta) = [str(t+9), str(t+12), str(t+15), str(t+18), ...]
    """
    ######### variables ####################
    # droput_rate = 0.3

    ######### inputs shape ####################
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    vel_shape = (1,) # velocity
    acc_shape = (1,) # acceleration

    ######model#######
    img_input = Input(shape=img_shape)
    vel_input = Input(shape=vel_shape)
    acc_input = Input(shape=acc_shape)
    
    base_model1 = pre_trained_str_thr_brk(base_model_path)
    base_model2 = pre_trained_str_thr_brk(base_model_path)
    base_model3 = pre_trained_str_thr_brk(base_model_path)
    
    pretrained_model_img_vec = Model(base_model1.input, base_model1.get_layer('flatten_1').output, name='base_model_img_vec')
    pretrained_model_vel = Model(base_model3.input, base_model3.get_layer('fc_vel').output, name='base_model_vel_vec')
    pretrained_model_str = Model(base_model2.input, base_model2.get_layer('fc_str').output, name='base_model_out_str')
    
    # if config['style_train'] is True:
    pretrained_model_img_vec.trainable = False
    pretrained_model_vel.trainable = False
    pretrained_model_str.trainable = False
        
    base_model_img_vec = pretrained_model_img_vec([img_input, vel_input, acc_input])
    base_model_vel_vec = pretrained_model_vel([img_input, vel_input, acc_input])
    base_model_str_output = pretrained_model_str([img_input, vel_input, acc_input])
    
    # fc_base_img = Dense(500, activation='relu', name='fc_base_img')(base_model_img_vec)
    # fc_base_vel = Dense(100, activation='relu', name='fc_base_vel')(base_model_vel_vec)
    # fc_base_str = Dense(100, activation='relu', name='fc_base_str')(base_model_str_output)

    # conc = Concatenate()([fc_base_img, fc_base_vel, fc_base_str])

    # delta = 0
    # then use the base model steering output    

    # delta = 9
    fc_base_img_d9 = Dense(500, activation='relu', name='fc_base_img_d9')(base_model_img_vec)
    fc_base_vel_d9 = Dense(100, activation='relu', name='fc_base_vel_d9')(base_model_vel_vec)
    fc_base_str_d9 = Dense(100, activation='relu', name='fc_base_str_d9')(base_model_str_output)
    conc_d9 = Concatenate()([fc_base_str_d9, fc_base_img_d9, fc_base_vel_d9])
    fc_2 = Dense(200, activation='relu', name='fc_2')(conc_d9)
    drop1 = Dropout(rate=0.3)(fc_2)
    fc_3 = Dense(100, activation='relu', name='fc_3')(drop1)
    drop_d9 = Dropout(rate=0.3)(fc_3)
    fc_d9 = Dense(50, activation='tanh', name='fc_d9')(drop_d9)
    fc_out_steer9 = Dense(1, activation='tanh', name='fc_out_steer9')(fc_d9)

    # delta = 12
    fc_base_img_d12 = Dense(500, activation='relu', name='fc_base_img_d12')(base_model_img_vec)
    fc_base_vel_d12 = Dense(100, activation='relu', name='fc_base_vel_d12')(base_model_vel_vec)
    fc_base_str_d12 = Dense(100, activation='relu', name='fc_base_str_d12')(base_model_str_output)
    conc_d12 = Concatenate()([fc_base_str_d12, fc_base_img_d12, fc_base_vel_d12])
    fc_4 = Dense(200, activation='relu', name='fc_4')(conc_d12)
    drop2 = Dropout(rate=0.3)(fc_4)
    fc_5 = Dense(100, activation='relu', name='fc_5')(drop2)
    drop_d12 = Dropout(rate=0.3)(fc_5)
    fc_d12 = Dense(50, activation='tanh', name='fc_d12')(drop_d12)
    fc_out_steer12 = Dense(1, activation='tanh', name='fc_out_steer12')(fc_d12)

    # delta = 15
    fc_base_img_d15 = Dense(500, activation='relu', name='fc_base_img_d15')(base_model_img_vec)
    fc_base_vel_d15 = Dense(100, activation='relu', name='fc_base_vel_d15')(base_model_vel_vec)
    fc_base_str_d15 = Dense(100, activation='relu', name='fc_base_str_d15')(base_model_str_output)
    conc_d15 = Concatenate()([fc_base_str_d15, fc_base_img_d15, fc_base_vel_d15])
    fc_6 = Dense(200, activation='relu', name='fc_6')(conc_d15)
    drop3 = Dropout(rate=0.3)(fc_6)
    fc_7 = Dense(100, activation='relu', name='fc_7')(drop3)
    drop_d15 = Dropout(rate=0.3)(fc_7)
    fc_d15 = Dense(50, activation='tanh', name='fc_d15')(drop_d15)
    fc_out_steer15 = Dense(1, activation='tanh', name='fc_out_steer15')(fc_d15)

    # delta = 18
    fc_base_img_d18 = Dense(500, activation='relu', name='fc_base_img_d18')(base_model_img_vec)
    fc_base_vel_d18 = Dense(100, activation='relu', name='fc_base_vel_d18')(base_model_vel_vec)
    fc_base_str_d18 = Dense(100, activation='relu', name='fc_base_str_d18')(base_model_str_output)
    conc_d18 = Concatenate()([fc_base_str_d18, fc_base_img_d18, fc_base_vel_d18])
    fc_8 = Dense(200, activation='relu', name='fc_8')(conc_d18)
    drop4 = Dropout(rate=0.3)(fc_8)
    fc_9 = Dense(100, activation='relu', name='fc_9')(drop4)
    drop_d18 = Dropout(rate=0.3)(fc_9)
    fc_d18 = Dense(50, activation='tanh', name='fc_d18')(drop_d18)
    fc_out_steer18 = Dense(1, activation='tanh', name='fc_out_steer18')(fc_d18)

    # delta = 21
    fc_base_img_d21 = Dense(500, activation='relu', name='fc_base_img_d21')(base_model_img_vec)
    fc_base_vel_d21 = Dense(100, activation='relu', name='fc_base_vel_d21')(base_model_vel_vec)
    fc_base_str_d21 = Dense(100, activation='relu', name='fc_base_str_d21')(base_model_str_output)
    conc_d21 = Concatenate()([fc_base_str_d21, fc_base_img_d21, fc_base_vel_d21])
    fc_10 = Dense(200, activation='relu', name='fc_10')(conc_d21)
    drop5 = Dropout(rate=0.3)(fc_10)
    fc_11 = Dense(100, activation='relu', name='fc_11')(drop5)
    drop_d21 = Dropout(rate=0.3)(fc_11)
    fc_d21 = Dense(50, activation='tanh', name='fc_d21')(drop_d21)
    fc_out_steer21 = Dense(1, activation='tanh', name='fc_out_steer21')(fc_d21)
    
    # model = Model(inputs=[img_input, vel_input, acc_input], 
    #               outputs=[base_model_str_output, fc_out_steer9, fc_out_steer12, 
    #                        fc_out_steer15, fc_out_steer18, fc_out_steer21])
    model = Model(inputs=[img_input, vel_input, acc_input], 
                  outputs=[fc_out_steer9, fc_out_steer12, 
                           fc_out_steer15, fc_out_steer18, fc_out_steer21])
    # model = Model(inputs=[img_input, vel_input], outputs=[fc_str, fc_thr, fc_brk])

    return model

def conditional_LSTM_MLP_delay_mitigation_all(base_model_path):
    """
    Conditional Imitation learning model with LSTM layer for each head.

    input: x(t) --> current frame image, velocity, and acceleration
    output: str(t+delta) --> where str(t+delta) = [str(t+9), str(t+12), str(t+15), str(t+18), ...]
    """
    ######### variables ####################
    # droput_rate = 0.3

    ######### inputs shape ####################
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    vel_shape = (1,) # velocity
    acc_shape = (1,) # acceleration

    ######model#######
    img_input = Input(shape=img_shape)
    vel_input = Input(shape=vel_shape)
    acc_input = Input(shape=acc_shape)
    
    base_model1 = pre_trained_str_thr_brk(base_model_path)
    base_model2 = pre_trained_str_thr_brk(base_model_path)
    base_model3 = pre_trained_str_thr_brk(base_model_path)
    
    pretrained_model_img_vec = Model(base_model1.input, base_model1.get_layer('flatten_1').output, name='base_model_img_vec')
    pretrained_model_vel = Model(base_model3.input, base_model3.get_layer('fc_vel').output, name='base_model_vel_vec')
    pretrained_model_str = Model(base_model2.input, base_model2.get_layer('fc_str').output, name='base_model_out_str')
    
    # if config['style_train'] is True:
    pretrained_model_img_vec.trainable = False
    pretrained_model_vel.trainable = False
    pretrained_model_str.trainable = False
        
    base_model_img_vec = pretrained_model_img_vec([img_input, vel_input, acc_input])
    base_model_vel_vec = pretrained_model_vel([img_input, vel_input, acc_input])
    base_model_str_output = pretrained_model_str([img_input, vel_input, acc_input])
    
    fc_base_img = Dense(500, activation='relu', name='fc_base_img')(base_model_img_vec)
    fc_base_vel = Dense(100, activation='relu', name='fc_base_vel')(base_model_vel_vec)
    fc_base_str = Dense(100, activation='relu', name='fc_base_str')(base_model_str_output)

    conc = Concatenate()([fc_base_img, fc_base_vel, fc_base_str])
    reshaped_conc = Reshape((1, -1))(conc)

    # delta = 0
    # then use the base model steering output    

    # delta = 9
    fc_2 = TimeDistributed(Dense(200, activation='elu'), name='fc_2')(reshaped_conc)
    lstm9 = LSTM(100, activation='elu', return_sequences=True)(fc_2)
    dropout_lstm9 = Dropout(rate=0.3)(lstm9)
    fc_d9 = TimeDistributed(Dense(50, activation='elu'), name='fc_d9')(dropout_lstm9)
    fc_out_steer9 = Dense(1, activation='tanh', name='fc_out_steer9')(fc_d9)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer9 = Reshape((1,))(fc_out_steer9)

    # delta = 12
    fc_4 = TimeDistributed(Dense(200, activation='elu'), name='fc_4')(reshaped_conc)
    lstm12 = LSTM(100, activation='elu', return_sequences=True)(fc_4)
    dropout_lstm12 = Dropout(rate=0.3)(lstm12)
    fc_d12 = TimeDistributed(Dense(50, activation='elu'), name='fc_d12')(dropout_lstm12)
    fc_out_steer12 = Dense(1, activation='tanh', name='fc_out_steer12')(fc_d12)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer12 = Reshape((1,))(fc_out_steer12)

    # delta = 15
    fc_6 = TimeDistributed(Dense(200, activation='elu'), name='fc_6')(reshaped_conc)
    lstm15 = LSTM(100, activation='elu', return_sequences=True)(fc_6)
    dropout_lstm15 = Dropout(rate=0.3)(lstm15)
    fc_d15 = TimeDistributed(Dense(50, activation='elu'), name='fc_d15')(dropout_lstm15)
    fc_out_steer15 = Dense(1, activation='tanh', name='fc_out_steer15')(fc_d15)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer15 = Reshape((1,))(fc_out_steer15)

    # delta = 18
    fc_8 = TimeDistributed(Dense(200, activation='elu'), name='fc_8')(reshaped_conc)
    lstm18 = LSTM(100, activation='elu', return_sequences=True)(fc_8)
    dropout_lstm18 = Dropout(rate=0.3)(lstm18)
    fc_d18 = TimeDistributed(Dense(50, activation='elu'), name='fc_d18')(dropout_lstm18)
    fc_out_steer18 = Dense(1, activation='tanh', name='fc_out_steer18')(fc_d18)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer18 = Reshape((1,))(fc_out_steer18)

    # delta = 21
    fc_10 = TimeDistributed(Dense(200, activation='relu'), name='fc_10')(reshaped_conc)
    lstm21 = LSTM(100, activation='elu', return_sequences=True)(fc_10)
    dropout_lstm21 = Dropout(rate=0.3)(lstm21)
    fc_d21 = TimeDistributed(Dense(50, activation='tanh'), name='fc_d21')(dropout_lstm21)
    fc_out_steer21 = Dense(1, activation='tanh', name='fc_out_steer21')(fc_d21)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer21 = Reshape((1,))(fc_out_steer21)
    
    model = Model(inputs=[img_input, vel_input, acc_input], 
                  outputs=[fc_out_steer9, fc_out_steer12, 
                           fc_out_steer15, fc_out_steer18, fc_out_steer21])

    return model

def conditional_LSTM_MLP_delay_mitigation_all_2(base_model_path):
    """
    Conditional Imitation learning model with LSTM layer for each head.

    input: x(t) --> current frame image, velocity, and acceleration
    output: str(t+delta) --> where str(t+delta) = [str(t+9), str(t+12), str(t+15), str(t+18), ...]
    """
    ######### variables ####################
    # droput_rate = 0.3

    ######### inputs shape ####################
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    vel_shape = (1,) # velocity
    acc_shape = (1,) # acceleration

    ######model#######
    img_input = Input(shape=img_shape)
    vel_input = Input(shape=vel_shape)
    acc_input = Input(shape=acc_shape)
    
    base_model1 = pre_trained_str_thr_brk(base_model_path)
    base_model2 = pre_trained_str_thr_brk(base_model_path)
    base_model3 = pre_trained_str_thr_brk(base_model_path)
    
    pretrained_model_img_vec = Model(base_model1.input, base_model1.get_layer('flatten_1').output, name='base_model_img_vec')
    pretrained_model_vel = Model(base_model3.input, base_model3.get_layer('fc_vel').output, name='base_model_vel_vec')
    pretrained_model_str = Model(base_model2.input, base_model2.get_layer('fc_str').output, name='base_model_out_str')
    
    # if config['style_train'] is True:
    pretrained_model_img_vec.trainable = False
    pretrained_model_vel.trainable = False
    pretrained_model_str.trainable = False
        
    base_model_img_vec = pretrained_model_img_vec([img_input, vel_input, acc_input])
    base_model_vel_vec = pretrained_model_vel([img_input, vel_input, acc_input])
    base_model_str_output = pretrained_model_str([img_input, vel_input, acc_input])
    
    fc_base_img = Dense(500, activation='relu', name='fc_base_img')(base_model_img_vec)
    fc_base_vel = Dense(100, activation='relu', name='fc_base_vel')(base_model_vel_vec)
    fc_base_str = Dense(100, activation='relu', name='fc_base_str')(base_model_str_output)

    conc = Concatenate()([fc_base_img, fc_base_vel, fc_base_str])
    reshaped_conc = Reshape((1, -1))(conc)

    # delta = 0
    # then use the base model steering output    

    # delta = 9
    lstm9_1 = LSTM(200, activation='tanh', return_sequences=True)(reshaped_conc) # default activation is tanh
    lstm9_2 = LSTM(100, activation='tanh', return_sequences=True)(lstm9_1)
    dropout_lstm9 = Dropout(rate=0.3)(lstm9_2)
    fc_d9 = TimeDistributed(Dense(50, activation='tanh'), name='fc_d9')(dropout_lstm9)
    fc_out_steer9 = Dense(1, activation='tanh', name='fc_out_steer9')(fc_d9)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer9 = Reshape((1,), name='rshp_steer9')(fc_out_steer9)

    # delta = 12
    lstm12_1 = LSTM(200, activation='tanh', return_sequences=True)(reshaped_conc)
    lstm12_2 = LSTM(100, activation='tanh', return_sequences=True)(lstm12_1)
    dropout_lstm12 = Dropout(rate=0.3)(lstm12_2)
    fc_d12 = TimeDistributed(Dense(50, activation='tanh'), name='fc_d12')(dropout_lstm12)
    fc_out_steer12 = Dense(1, activation='tanh', name='fc_out_steer12')(fc_d12)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer12 = Reshape((1,), name='rshp_steer12')(fc_out_steer12)

    # delta = 15
    lstm15_1 = LSTM(200, activation='tanh', return_sequences=True)(reshaped_conc)
    lstm15_2 = LSTM(100, activation='tanh', return_sequences=True)(lstm15_1)
    dropout_lstm15 = Dropout(rate=0.3)(lstm15_2)
    fc_d15 = TimeDistributed(Dense(50, activation='tanh'), name='fc_d15')(dropout_lstm15)
    fc_out_steer15 = Dense(1, activation='tanh', name='fc_out_steer15')(fc_d15)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer15 = Reshape((1,), name='rshp_steer15')(fc_out_steer15)

    # delta = 18
    lstm18_1 = LSTM(200, activation='tanh', return_sequences=True)(reshaped_conc)
    lstm18_2 = LSTM(100, activation='tanh', return_sequences=True)(lstm18_1)
    dropout_lstm18 = Dropout(rate=0.3)(lstm18_2)
    fc_d18 = TimeDistributed(Dense(50, activation='tanh'), name='fc_d18')(dropout_lstm18)
    fc_out_steer18 = Dense(1, activation='tanh', name='fc_out_steer18')(fc_d18)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer18 = Reshape((1,), name='rshp_steer18')(fc_out_steer18)

    # delta = 21
    lstm21_1 = LSTM(200, activation='tanh', return_sequences=True)(reshaped_conc)
    lstm21_2 = LSTM(100, activation='tanh', return_sequences=True)(lstm21_1)
    dropout_lstm21 = Dropout(rate=0.3)(lstm21_2)
    fc_d21 = TimeDistributed(Dense(50, activation='tanh'), name='fc_d21')(dropout_lstm21)
    fc_out_steer21 = Dense(1, activation='tanh', name='fc_out_steer21')(fc_d21)
    # Reshape the output to be 2D instead of 3D
    fc_out_steer21 = Reshape((1,), name='rshp_steer21')(fc_out_steer21)
    
    model = Model(inputs=[img_input, vel_input, acc_input], 
                  outputs=[fc_out_steer9, fc_out_steer12, 
                           fc_out_steer15, fc_out_steer18, fc_out_steer21])

    return model

def rover_multiclass_str():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    return Sequential([
        Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape),
        Conv2D(24, (5, 5), strides=(2,2), activation='relu'),
        Conv2D(36, (5, 5), strides=(2,2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2,2), activation='relu'),
        # Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_last'),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(3, name='steer_class')]) # If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).

def model_ce491():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    return Sequential([
        Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape),
        Conv2D(24, (5, 5), strides=(2,2), activation='relu'),
        Conv2D(36, (5, 5), strides=(2,2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2,2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_last'),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(config['num_outputs'])]) # If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).

def model_jaerock():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    return Sequential([
        Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape),
        Conv2D(24, (5, 5), strides=(2,2)),
        Conv2D(36, (5, 5), strides=(2,2)),
        Conv2D(48, (5, 5), strides=(2,2)),
        Conv2D(64, (3, 3)),
        Conv2D(64, (3, 3), name='conv2d_last'),
        Flatten(),
        Dense(1000),
        Dense(100),
        Dense(50),
        Dense(10),
        Dense(config['num_outputs'])])

    
def model_jaerock_vel():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    vel_shape = 1
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2))(lamb)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2))(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2))(conv_2)
    conv_4 = Conv2D(64, (3, 3))(conv_3)
    conv_5 = Conv2D(64, (3, 3), name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(1000, name='fc_1')(flat)
    fc_2 = Dense(100, name='fc_2')(fc_1)
    
    ######vel model#######
    vel_input = Input(shape=[vel_shape])
    fc_vel = Dense(50, name='fc_vel')(vel_input)
    
    ######concat##########
    concat_img_vel = concatenate([fc_2, fc_vel])
    fc_3 = Dense(50, name='fc_3')(concat_img_vel)
    fc_4 = Dense(10, name='fc_4')(fc_3)
    fc_last = Dense(2, name='fc_str')(fc_4)
    
    model = Model(inputs=[img_input, vel_input], output=fc_last)

    return model

def model_convlstm():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    vel_shape = (None, 1)
    
    input_img = Input(shape=img_shape, name='input_image')
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb_img')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), strides=(2,2)), name='conv_1')(lamb)
    conv_2    = TimeDistributed(Convolution2D(36, (5, 5), strides=(2,2)), name='conv_2')(conv_1)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), strides=(2,2)), name='conv_3')(conv_2)
    conv_4    = TimeDistributed(Convolution2D(64, (3, 3)), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Convolution2D(64, (3, 3)), name='conv2d_last')(conv_4)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    fc_1      = TimeDistributed(Dense(1000, activation='relu'), name='fc_1')(flat)
    fc_2      = TimeDistributed(Dense(100, activation='relu' ), name='fc_2')(fc_1)
    
    if config['num_inputs'] == 1:
        lstm      = LSTM(10, return_sequences=False, name='lstm')(fc_2)
        fc_3      = Dense(50, activation='relu', name='fc_3')(lstm)
        fc_4      = Dense(10, activation='relu', name='fc_4')(fc_3)
        fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_4)
    
        model = Model(inputs=input_img, outputs=fc_last)
        
    elif config['num_inputs'] == 2:
        input_velocity = Input(shape=vel_shape, name='input_velocity')
        lamb      = TimeDistributed(Lambda(lambda x: x / 38), name='lamb_vel')(input_velocity)
        fc_vel_1  = TimeDistributed(Dense(50, activation='relu'), name='fc_vel')(lamb)
        concat    = concatenate([fc_2, fc_vel_1], name='concat')
        lstm      = LSTM(10, return_sequences=False, name='lstm')(concat)
        fc_3      = Dense(50, activation='relu', name='fc_3')(lstm)
        fc_4      = Dense(10, activation='relu', name='fc_4')(fc_3)
        fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_4)

        model = Model(inputs=[input_img, input_velocity], outputs=fc_last)
    
    return model

def model_jaerock_shift():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    steering_shape = 1
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2))(lamb)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2))(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2))(conv_2)
    conv_4 = Conv2D(64, (3, 3))(conv_3)
    conv_5 = Conv2D(64, (3, 3), name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(1000, name='fc_1')(flat)
    fc_2 = Dense(100, name='fc_2')(fc_1)
    
    ######steering model#######
    steering_input = Input(shape=[steering_shape])
    fc_steering = Dense(50, name='fc_steering')(steering_input)
    
    ######concat##########
    concat_img_steering = concatenate([fc_2, fc_steering])
    fc_3 = Dense(50, name='fc_3')(concat_img_steering)
    fc_4 = Dense(10, name='fc_4')(fc_3)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_4)
    
    model = Model(inputs=[img_input, steering_input], output=fc_last)

    return model

def model_convlstm_img_out():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    steering_shape = (None, 1)
    
    input_img = Input(shape=img_shape, name='input_image')
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb_img')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), strides=(2,2)), name='conv_1')(lamb)
    conv_2    = TimeDistributed(Convolution2D(36, (5, 5), strides=(2,2)), name='conv_2')(conv_1)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), strides=(2,2)), name='conv_3')(conv_2)
    conv_4    = TimeDistributed(Convolution2D(64, (3, 3)), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Convolution2D(64, (3, 3)), name='conv2d_last')(conv_4)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    fc_1      = TimeDistributed(Dense(1000, activation='relu'), name='fc_1')(flat)
    fc_2      = TimeDistributed(Dense(100, activation='relu' ), name='fc_2')(fc_1)

    input_steering = Input(shape=steering_shape, name='input_steering')
    lamb      = TimeDistributed(Lambda(lambda x: x / 38), name='lamb_vel')(input_steering)
    fc_steer_1  = TimeDistributed(Dense(50, activation='relu'), name='fc_steer_1')(lamb)
    concat    = concatenate([fc_2, fc_steer_1], name='concat')
    lstm      = LSTM(10, return_sequences=False, name='lstm')(concat)
    fc_3      = Dense(50, activation='relu', name='fc_3')(lstm)
    fc_4      = Dense(10, activation='relu', name='fc_4')(fc_3)
    fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_4)

    model = Model(inputs=[input_img, input_steering], outputs=fc_last)
    
    return model

class NetModel:
    def __init__(self, model_path, base_model_path = None):
        self.model = None
        self.base_model = None
        model_name = model_path[model_path.rfind('/'):] # get folder name
        self.name = model_name.strip('/')

        self.model_path = model_path
        #self.config = Config()
        self.base_model_path = base_model_path

        # to address the error:
        #   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # K.tensorflow_backend.set_session(sess)

        self._model(base_model_path=base_model_path)

    ###########################################################################
    #
    def _model(self, base_model_path=None):
        if config['network_type'] == const.NET_TYPE_ROVER_MULTICLASS_STR:
            self.model = rover_multiclass_str()
        elif config['network_type'] == const.NET_TYPE_STR_THR_BRK:
            self.model = str_thr_brk()
        elif config['network_type'] == const.NET_TYPE_LSTM_DELAY_MITIGATION:
            self.base_model = str_thr_brk()
            self.model = multistep_lstm_delay_mitigation(base_model_path)
        elif config['network_type'] == const.NET_TYPE_CONDITIONAL_MLP_DELAY_MITIGATION:
            self.base_model = str_thr_brk()
            self.model = conditional_MLP_delay_mitigation(base_model_path)
        elif config['network_type'] == const.NET_TYPE_CONDITIONAL_MLP_DELAY_MITIGATION_ALL:
            self.base_model = str_thr_brk()
            self.model = conditional_MLP_delay_mitigation_all(base_model_path)
        elif config['network_type'] == const.NET_TYPE_CONDITIONAL_MLP_DELAY_MITIGATION_ALL_2:
            self.base_model = str_thr_brk()
            self.model = conditional_MLP_delay_mitigation_all_2(base_model_path)
        elif config['network_type'] == const.NET_TYPE_CONDITIONAL_LSTM_MLP_DELAY_MITIGATION_ALL:
            self.base_model = str_thr_brk()
            self.model = conditional_LSTM_MLP_delay_mitigation_all(base_model_path)
        elif config['network_type'] == const.NET_TYPE_CONDITIONAL_LSTM_MLP_DELAY_MITIGATION_ALL_2:
            self.base_model = str_thr_brk()
            self.model = conditional_LSTM_MLP_delay_mitigation_all_2(base_model_path)
        elif config['network_type'] == const.NET_TYPE_JAEROCK:
            self.model = model_jaerock()
        elif config['network_type'] == const.NET_TYPE_JAEROCK_VEL:
            self.model = model_jaerock_vel()
        elif config['network_type'] == const.NET_TYPE_JAEROCK_SHIFT:
            self.model = model_jaerock_shift()    
        elif config['network_type'] == const.NET_TYPE_CE491:
            self.model = model_ce491()
        elif config['network_type'] == const.NET_TYPE_CONVLSTM:
            self.model = model_convlstm()
        elif config['network_type'] == const.NET_TYPE_CONVLSTM_IMG_OUT:
            self.model = model_convlstm_img_out()
        else:
            exit('ERROR: Invalid neural network type.')

        self.summary()
        self._compile()


    ###########################################################################
    #

    def _custom_loss_cosine_similarity(self, y_true, y_pred):
        # Compute the cosine similarity between y_true and y_pred
        dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
        magnitude_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1))
        magnitude_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1))
        
        cosine_sim = dot_product / (magnitude_true * magnitude_pred + 1e-7)  # Adding a small epsilon for numerical stability
        
        # Define the penalty factor
        penalty = 2.0  # You can adjust this penalty factor as needed
        
        # Apply the custom logic element-wise
        condition = tf.logical_and(tf.greater(y_pred[:, 0], 0), tf.greater(y_pred[:, 1], 0))
        loss = tf.where(condition, (1.0 - cosine_sim) * penalty, 1.0 - cosine_sim)
        
        return tf.reduce_mean(loss)  # Take the mean of the losses across the batch


    def _custom_loss_mse(self, y_true, y_pred):
        # mse = ()
        penalty = 100
        # ypred_np = K.eval(y_pred)
        # print('********************************************')#, output_stream=sys.stderr)
        # print(K.eval(y_pred))#[0], [1], [1]))#tf.slice(y_pred,[0], [1]))#, output_stream=sys.stderr)
        # print('********************************************')
        # actual = 0.1 and pred = -0.05 should be penalized a lot more than actual = 0.1 and pred = 0.05
        
        condition = tf.logical_and(tf.greater(y_pred[:,1], 0.0), tf.less(y_pred[:,2], 0.0))
        mse_value = losses.mse(y_true, y_pred)
        # loss = tf.cond(condition, lambda:  mse_value * penalty, lambda: mse_value)
        
        #actual = 0.1 and pred = 0.15 slightly more penalty than actual = 0.1 and pred = 0.05
        # loss = tf.cond(tf.greater(y_pred, y_true),
        #                 lambda: loss * penalty / 2,
        #                 lambda: loss * penalty / 3)

        loss = tf.where(condition, mse_value * penalty, mse_value)

        return tf.reduce_mean(loss)
    
    def _custom_loss_mse_3_out(self, y_true, y_pred):
        penalty = 100

        # Extract the individual predictions for throttle, brake, and steering
        steering_pred, throttle_pred, brake_pred = tf.unstack(y_pred, axis=-1)

        # Calculate the MSE for throttle and brake
        mse_throttle_brake = losses.mse(y_true[:, :2], y_pred[:, :2])

        # Calculate the MSE for steering
        mse_steering = losses.mse(y_true[:, 0], steering_pred)

        # Apply the penalty to the combined loss
        condition = tf.logical_and(tf.greater(throttle_pred, 0.0), tf.less(brake_pred, 0.0))
        loss = tf.where(condition, mse_throttle_brake * penalty, mse_throttle_brake)

        # Add the MSE for steering to the loss
        loss += mse_steering

        return tf.reduce_mean(loss)
    
    def _custom_loss_mse_2_out(self, y_true, y_pred):
        penalty = 100

        # Extract the individual predictions for throttle, brake, and steering
        throttle_pred, brake_pred = tf.unstack(y_pred, axis=-1)

        # Calculate the MSE for throttle and brake
        mse_throttle_brake = losses.mse(y_true[:, :2], y_pred[:, :2])

        # Apply the penalty to the combined loss
        condition = tf.logical_and(tf.greater(throttle_pred, 0.0), tf.less(brake_pred, 0.0))
        loss = tf.where(condition, mse_throttle_brake * penalty, mse_throttle_brake)

        return tf.reduce_mean(loss)

    def _r_squared(self, y_true, y_pred):
        """
        Compute the R-squared (coefficient of determination) metric.

        Parameters:
        y_true -- True labels (ground truth)
        y_pred -- Predicted labels

        Returns:
        r2 -- R-squared value
        """
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        r2 = 1 - (SS_res / (SS_tot + K.epsilon()))
        return r2
    

    def _cosine_similarity(self, y_true, y_pred):
        # Compute the dot product between y_true and y_pred
        dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # Compute the magnitude of y_true and y_pred
        magnitude_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1))
        magnitude_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1))
        
        # Calculate cosine similarity
        cosine_sim = dot_product / (magnitude_true * magnitude_pred + 1e-7)  # Adding a small epsilon for numerical stability
        
        # Take the mean cosine similarity across the batch
        mean_cosine_sim = tf.reduce_mean(cosine_sim)
        
        return mean_cosine_sim
    
    def _rmse_w_penalty(self, y_true, y_pred):
        # mse = ()
        penalty = 100
        # ypred_np = K.eval(y_pred)
        # print('********************************************')#, output_stream=sys.stderr)
        # print(K.eval(y_pred))#[0], [1], [1]))#tf.slice(y_pred,[0], [1]))#, output_stream=sys.stderr)
        # print('********************************************')
        # actual = 0.1 and pred = -0.05 should be penalized a lot more than actual = 0.1 and pred = 0.05
        condition = tf.logical_and(tf.greater(y_pred[:,0], 0.0), tf.less(y_pred[:,1], 0.0))
        mse_value = losses.mse(y_true, y_pred)
        rmse = tf.sqrt(mse_value)
        # loss = tf.cond(condition, lambda:  mse_value * penalty, lambda: mse_value)
        
        #actual = 0.1 and pred = 0.15 slightly more penalty than actual = 0.1 and pred = 0.05
        # loss = tf.cond(tf.greater(y_pred, y_true),
        #                 lambda: loss * penalty / 2,
        #                 lambda: loss * penalty / 3)

        loss = tf.where(condition, rmse * penalty, rmse)

        return tf.reduce_mean(loss)

    def _compile(self):
        
        if config['lstm'] is True:
            learning_rate = config['lstm_lr']
        else:
            learning_rate = config['cnn_lr']

        decay = config['decay']
        decay_lstm = config['decay_lstm']

        if config['delay_mitig']:

            self.base_model.compile(loss=losses.mean_squared_error,
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay, clipvalue=1), 
                        metrics=[self._r_squared])
            
            self.model.compile(loss=losses.mean_squared_error,
                    optimizer=optimizers.Adam(lr=learning_rate, decay=decay, clipvalue=1), 
                    metrics=[self._r_squared])
            
        else:
            self.model.compile(loss=[losses.mean_squared_error, losses.mean_squared_error], 
                        optimizer=optimizers.Adam(lr=learning_rate, decay=decay, clipvalue=1), 
                        metrics=[self._r_squared])


        # if config['custom_loss']:
        #     self.model.compile(loss=[losses.mean_squared_error, self._custom_loss_mse_2_out],
        #                 optimizer=optimizers.Adam(lr=learning_rate), 
        #                 metrics=[self._r_squared])
        #                 # metrics=['accuracy'])
        #                 # metrics=[self._rmse_w_penalty])
        # else:
        #     self.model.compile(loss=[losses.mean_squared_error, losses.mean_squared_error], 
        #                 optimizer=optimizers.Adam(lr=learning_rate), 
        #                 metrics=[self._r_squared])
                        # metrics=['accuracy'])
        # if config['steering_angle_tolerance'] == 0.0:
        #     self.model.compile(loss=losses.mean_squared_error,
        #               optimizer=optimizers.Adam(),
        #               metrics=['accuracy'])
        # else:
        #     self.model.compile(loss=losses.mean_squared_error,
        #               optimizer=optimizers.Adam(),
        #               metrics=['accuracy', self._mean_squared_error])


    ###########################################################################
    #
    # save model
    def save(self, model_name):

        json_string = self.model.to_json()
        #weight_filename = self.model_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type'])
        open(model_name+'.json', 'w').write(json_string)
        self.model.save_weights(model_name+'.h5', overwrite=True)


    ###########################################################################
    # model_path = '../data/2007-09-22-12-12-12.
    def load(self):

        # from keras.models import model_from_json
        # if config['delay_mitig']:
        #     self.base_model
        # self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')

        if config['delay_mitig'] is True:
            self.base_model.load_weights(self.base_model_path+'.h5')

        self._compile()

    ###########################################################################
    #
    # show summary
    def summary(self):
        # if config['delay_miti']:
        #     self.base_model.summary()
        self.model.summary()


if __name__ == "__main__":
    try:

        if config['delay_mitig']==True:
            if len(sys.argv) !=3:
                exit('Delay Mitigation is True in the config file: please add the base model path')
            netmodel = NetModel(sys.argv[1],sys.argv[2]) # model_path, base_model_path
        else:
            if len(sys.argv) < 2:
                exit('Usage:\n$ rosrun run_neural run_neural.py weight_file_name')
            netmodel = NetModel(sys.argv[1]) # model_path
        # netmodel.summary()

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')