from Models.BaseModel import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import Add, MaxPooling2D, Lambda, PReLU
from utils.losses import *

import tensorflow as tf
import os

"""
Baseline Model
"""

class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.build_model()
        self.model.summary()

    def build_model(self):
        def conv_block(net, depth_ksize=3, depth_strides=1, conv_filters=16, conv_ksize=1,
                       conv_strides=1):
            shortcut = net

            net = DepthwiseConv2D(kernel_size=depth_ksize, strides=depth_strides,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01), padding='same')(net)
            net = Conv2D(filters=conv_filters, kernel_size=conv_ksize, strides=conv_strides,
                         kernel_regularizer=tf.keras.regularizers.l2(0.01), padding='same')(net)

            net = Add()([shortcut, net])
            net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            return net

        def branch_block(net, depth_ksize=3, depth_strides=2, conv_filters=16, conv_ksize=1,
                         conv_strides=1, pad=True):
            branch_1 = DepthwiseConv2D(kernel_size=depth_ksize, strides=depth_strides, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01))(net)
            branch_1 = Conv2D(filters=conv_filters, kernel_size=conv_ksize, strides=conv_strides, padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(branch_1)

            branch_2 = MaxPooling2D(pool_size=2)(net)
            if pad:
                branch_2 = tf.pad(branch_2, paddings=[[0, 0], [0, 0], [0, 0], [0, int(conv_filters/2)]], mode='CONSTANT', constant_values=0)

            net = Add()([branch_1, branch_2])
            net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            return net

        inputs = Input(shape=(self.config.img_size, self.config.img_size, self.config.n_channel), name='input') # (192, 192, 3)

        net = Conv2D(filters=16, kernel_size=3, strides=2, padding='same')(inputs) # (96, 96, 16)
        net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

        net = conv_block(net, conv_filters=16)
        net = conv_block(net, conv_filters=16)

        net = branch_block(net, depth_strides=2, conv_filters=32) # (48, 48, 32)

        net = conv_block(net, conv_filters=32)
        net = conv_block(net, conv_filters=32)

        net = branch_block(net, depth_strides=2, conv_filters=64) # (24, 24, 64)

        net = conv_block(net, conv_filters=64)
        net = conv_block(net, conv_filters=64)

        net = branch_block(net, depth_strides=2, conv_filters=128) # (12, 12, 128)

        net = conv_block(net, conv_filters=128)
        net = conv_block(net, conv_filters=128)

        net = branch_block(net, depth_strides=2, conv_filters=128, pad=False) # (6, 6, 128)

        net = conv_block(net, conv_filters=128)
        net = conv_block(net, conv_filters=128)

        '''split branch CONTOUR'''
        contour_net = branch_block(net, depth_strides=2, conv_filters=128, pad=False) # (3,
        # 3, 128)

        contour_net = conv_block(contour_net, conv_filters=128)
        contour_net = conv_block(contour_net, conv_filters=128)

        contour_net = Conv2D(filters=32, kernel_size=1, padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(0.01))(contour_net) # (3, 3, 32)
        contour_net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(contour_net)

        contour_net = conv_block(contour_net, conv_filters=32)

        # last conv CONTOUR
        contour_net = Conv2D(filters=48, kernel_size=3, padding='valid',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01), name='contour_last_conv')(contour_net) # (1, 1, N_CONTOURS)
        # reshape result
        landmarks = tf.reshape(contour_net, [-1, 24, 2], name='contour_tf')

        '''split branch POSE'''
        pose_net = branch_block(net, depth_strides=2, conv_filters=128, pad=False) # (3, 3, 128)

        pose_net = conv_block(pose_net, conv_filters=128)
        pose_net = conv_block(pose_net, conv_filters=128)

        pose_net = Conv2D(filters=32, kernel_size=1, padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(pose_net) # (3, 3, 32)
        pose_net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(pose_net)

        pose_net = conv_block(pose_net, conv_filters=32)

        # last conv POSE
        pose_net = Conv2D(filters=3, kernel_size=3, padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                          name='pose_last_conv')(pose_net) # (1, 1, 3)
        # reshape result
        pose = tf.reshape(pose_net, [-1, 3], name='pose_tf')
        '''end of model'''

        # compile model
        self.model = tf.keras.Model(inputs=inputs, outputs=[landmarks, pose])

        self.model.compile(loss=[dist_squared, 'mse'], optimizer='adam')

        # # load model if exists
        # if len(os.listdir(self.config.checkpoint_dir)) > 0:
        #     checkpoints = os.listdir(self.config.checkpoint_dir)
        #     checkpoints.sort()
        #     self.load(os.path.join(self.config.checkpoint_dir, checkpoints[-1]))
