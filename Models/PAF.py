from Models.BaseModel import BaseModel
import tensorflow.keras.layers as layers
from utils.losses import *

import tensorflow as tf


"""
PAF module
"""
class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)

        self.dropout_rate = 0.10
        self.build_model()

    def build_model(self):

        def paf_generator(input_img):
            # per image standardization
            img = layers.Lambda(lambda x: tf.image.per_image_standardization(x))(input_img)

            # downsample a bit
            net = layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l1(0.01))(img)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)
            # (48 48 16)
            net = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)
            # (24 24 32)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Conv2D(64, kernel_size=1, strides=1, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            # net = layers.Add()([shortcut, net])
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            # net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Add()([shortcut, net])
            net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Add()([shortcut, net])
            net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Add()([shortcut, net])
            net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            # shortcut = net
            # net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
            #                              kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            # net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same',
            #                     kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            # net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            # net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # net = layers.Dropout(rate=self.dropout_rate)(net)
            #
            # shortcut = net
            # net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
            #                              kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            # net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same',
            #                     kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            # net = layers.Add()([shortcut, net])
            # # net = layers.ReLU()(net)
            # net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # net = layers.Dropout(rate=self.dropout_rate)(net)

            # transposes
            net = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                         output_padding=1, use_bias=False)(net)
            net = layers.BatchNormalization(momentum=0.99)(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # net = layers.ReLU()(net)

            net = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                                         output_padding=1, use_bias=False)(net)
            net = layers.BatchNormalization(momentum=0.99)(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            out = layers.Conv2D(2, kernel_size=1, strides=1, padding='same')(net)
            # out = kernel_convolution(out)

            return out

        input_img = layers.Input((self.config.img_size, self.config.img_size, self.config.n_channel))
        paf = paf_generator(input_img)
        self.paf_gen = tf.keras.Model(inputs=input_img, outputs=paf, name='paf_generator')

        self.paf_gen.summary()





