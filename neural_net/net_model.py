#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:20:14 2023
History:
12/17/2023 modified by Aws.

@author: Aws
# -*- coding: utf-8 -*-
"""
import sys
from keras.models import Sequential, Model
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, concatenate
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, BatchNormalization, Input
from keras.layers import Add, Subtract, Multiply, RepeatVector, Reshape
from keras import losses, optimizers, metrics
import keras.backend as K
import tensorflow as tf

import const
from config import Config


config = Config.neural_net
config_rn = Config.run_neural


def str_thr_brk():
    """
    This model takes the current frame image as input
    (you can add velocity and/or acceleration too) 
    and gives the action values as output (steering, throttle, brake)
    I am trying to use this model to cancel the simple controller used in the 
    run_neural.py file. (catkin_ws/src/run_neural/run_neural.py)

    input: x(t), v(t), acc(t) --> current frame image, velocity, and acceleration
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
        concat_inputs = concatenate([flat,vel_vec, acc_vec])
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
    def __init__(self, model_path):
        self.model = None
        model_name = model_path[model_path.rfind('/'):] # get folder name
        self.name = model_name.strip('/')

        self.model_path = model_path
        #self.config = Config()

        # to address the error:
        #   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # K.tensorflow_backend.set_session(sess)

        self._model()

    ###########################################################################
    #
    def _model(self):
        if config['network_type'] == const.NET_TYPE_ROVER_MULTICLASS_STR:
            self.model = rover_multiclass_str()
        elif config['network_type'] == const.NET_TYPE_STR_THR_BRK:
            self.model = str_thr_brk()
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



    # ###########################################################################
    # #
    # def _mean_squared_error(self, y_true, y_pred):
    #     diff = K.abs(y_true - y_pred)
    #     if (diff < config['steering_angle_tolerance']) is True:
    #         diff = 0
    #     return K.mean(K.square(diff))

    ###########################################################################
    #


    
    def _custom_loss_mse_3_out(self, y_true, y_pred):
        """
        use this loss if your final model layer is dense(3, ...)
        It adds a penalty in case the throttle and brake were used simultaneously 
        Do not use if combined the throttle and brake in on data column and consider them one variable.
        """
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
        """
        use this loss if your model has two output heads
        one for steering - one for throttle, and brake
        It adds a penalty in case the throttle and brake were used simultaneously 
        Do not use if combined the throttle and brake in on data column and consider them one variable.
        """
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
    

    def _compile(self):
        
        if config['lstm'] is True:
            learning_rate = config['lstm_lr']
        else:
            learning_rate = config['cnn_lr']

        if config['custom_loss']:
            self.model.compile(loss=[losses.mean_squared_error, self._custom_loss_mse_2_out],
                        optimizer=optimizers.Adam(lr=learning_rate), 
                        metrics=[self._r_squared])
                        # metrics=['accuracy'])
                        # metrics=[self._rmse_w_penalty])
        else:
            self.model.compile(loss=[losses.mean_squared_error, losses.mean_squared_error], 
                        optimizer=optimizers.Adam(lr=learning_rate), 
                        metrics=[self._r_squared])
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

        from keras.models import model_from_json

        self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        self._compile()

    ###########################################################################
    #
    # show summary
    def summary(self):
        self.model.summary()

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            exit('Usage:\n$ rosrun run_neural run_neural.py weight_file_name')

        netmodel = NetModel(sys.argv[1]) #model_path
        # netmodel.summary()

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
