from Models.BaseModel import BaseModel
import tensorflow.keras.layers as layers
from utils.losses import *

import tensorflow as tf

"""
Boundary-Aware Face Alignment
"""

class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.num_landmarks = 60
        self.num_boundaries = 11
        self.dropout_rate = 0.10

        self.kernel = self.create_kernel(self.config.dot_size)
        self.build_model()

    def create_kernel(self, dot_size):
        alpha = 1.0
        x_axis = tf.linspace(alpha, -alpha, dot_size)[:, None]
        y_axis = tf.linspace(alpha, -alpha, dot_size)[None, :]

        template = tf.sqrt(x_axis ** 2 + y_axis ** 2)
        template = tf.reduce_max(template) - template
        template = template / tf.reduce_max(template)

        kernel = tf.reshape([template] * self.num_boundaries, (self.num_boundaries, dot_size, dot_size, 1))
        kernel = tf.transpose(kernel, [1, 2, 0, 3])
        kernel = tf.cast(kernel, tf.float32)
        return tf.Variable(kernel, trainable=False)

    def build_model(self):
        def kernel_convolution(net):
            out = tf.nn.depthwise_conv2d(net, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
            out = layers.Activation('sigmoid', name='local')(out)
            return out

        def conv_block(net, depth_ksize=3, depth_strides=1, conv_filters=16, conv_ksize=1, conv_strides=1):
            shortcut = net

            net = layers.DepthwiseConv2D(kernel_size=depth_ksize, strides=depth_strides,
                                  kernel_regularizer=tf.keras.regularizers.l1(0.01), padding='same')(net)
            net = layers.Conv2D(filters=conv_filters, kernel_size=conv_ksize, strides=conv_strides,
                         kernel_regularizer=tf.keras.regularizers.l1(0.01), padding='same')(net)

            net = layers.Add()([shortcut, net])
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            return net

        def branch_block(net, depth_ksize=3, depth_strides=2, conv_filters=16, conv_ksize=1, conv_strides=1, pad=True):
            branch_1 = layers.DepthwiseConv2D(kernel_size=depth_ksize, strides=depth_strides, padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l1(0.01))(net)
            branch_1 = layers.Conv2D(filters=conv_filters, kernel_size=conv_ksize, strides=conv_strides, padding='same',
                              kernel_regularizer=tf.keras.regularizers.l1(0.01))(branch_1)

            branch_2 = layers.MaxPooling2D(pool_size=2)(net)
            if pad:
                branch_2 = tf.pad(branch_2, paddings=[[0, 0], [0, 0], [0, 0], [0, int(conv_filters/2)]], mode='CONSTANT', constant_values=0)

            net = layers.Add()([branch_1, branch_2])
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            return net

        def boundary_generator(input_img):
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

            out = layers.Conv2D(self.num_boundaries, kernel_size=1, strides=1, padding='same')(net)
            # out = kernel_convolution(out)

            return out


        def landmark_regressor(inputs):
            dropout_rate = 0.5
            net = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs) # (48, 48, 32)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            net = layers.Dropout(rate=dropout_rate)(net)

            net = conv_block(net, conv_filters=32)
            net = branch_block(net, depth_strides=2, conv_filters=64) # (24, 24, 64)
            net = layers.Dropout(rate=dropout_rate)(net)

            net = conv_block(net, conv_filters=64)
            net = branch_block(net, depth_strides=2, conv_filters=128) # (12, 12, 128)
            net = layers.Dropout(rate=dropout_rate)(net)

            net = conv_block(net, conv_filters=128)
            net = branch_block(net, depth_strides=2, conv_filters=128, pad=False) # (6, 6, 128)
            net = layers.Dropout(rate=dropout_rate)(net)

            net = conv_block(net, conv_filters=128)
            net = branch_block(net, depth_strides=2, conv_filters=128, pad=False) # (3, 3, 128)
            net = layers.Dropout(rate=dropout_rate)(net)

            net = layers.GlobalAveragePooling2D()(net) # (128)
            net = layers.Dense(128, activation='relu')(net)
            net = layers.Dense(128, activation='relu')(net)
            net = layers.Dense(self.num_landmarks * 2, activation='linear')(net)
            return tf.reshape(net, [-1, self.num_landmarks, 2], name='landmark')

        def effectiveness_discriminator(input_img):
            # (96, 96, 11)
            net = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input_img) # (48, 48, 32)
            net = layers.LeakyReLU()(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            net = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(net) # (24, 24, 64)
            net = layers.LeakyReLU()(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            net = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(net) # (12, 12, 64)
            net = layers.LeakyReLU()(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            net = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(net) # (6, 6, 64)
            net = layers.LeakyReLU()(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            net = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(net) # (3, 3, 64)
            net = layers.LeakyReLU()(net)
            net = layers.Dropout(rate=self.dropout_rate)(net)

            net = layers.Conv2D(filters=64, kernel_size=3)(net) # (1, 1, 64)
            net = layers.Flatten()(net) # (64)

            logit = layers.Dense(1, activation='sigmoid', name='logit')(net) # (1)

            return logit

        def input_image_fusion(x, bm_):
            x_s = tf.reduce_sum(x, -1)
            temp = tf.tile(tf.reshape(x_s, (-1, self.config.img_size, self.config.img_size, 1)), [1, 1, 1, 11])
            fused_img_ = temp * tf.image.resize(bm_, (self.config.img_size, self.config.img_size))
            x_s = tf.reshape(x_s, (-1, self.config.img_size, self.config.img_size, 1))
            fused_img_ = tf.concat([x_s, fused_img_], -1)
            return fused_img_

        def input_coordinate_fusion(bm_):
            coord = tf.linspace(0.0, self.config.img_size - 1.0, self.config.img_size)
            x_coord = tf.tile(tf.reshape(coord, [1, 1, self.config.img_size, 1]), [self.config.batch_size, self.config.img_size, 1, 11])
            y_coord = tf.tile(tf.reshape(coord, [1, self.config.img_size, 1, 1]), [self.config.batch_size, 1, self.config.img_size, 11])

            x_coord_bm_ = bm_ * x_coord
            y_coord_bm_ = bm_ * y_coord

            return x_coord_bm_, y_coord_bm_

        input_img = layers.Input((self.config.img_size, self.config.img_size, self.config.n_channel))
        boundary = boundary_generator(input_img)
        self.generator = tf.keras.Model(inputs=input_img, outputs=boundary, name='boundary_generator')

        input_boundary = layers.Input((self.config.img_size, self.config.img_size, self.num_boundaries))
        fused_img = input_image_fusion(input_img, input_boundary)
        landmark = landmark_regressor(fused_img)
        self.regressor = tf.keras.Model(inputs=[input_img, input_boundary], outputs=landmark, name='landmark_regressor')

        predicted_boundary = layers.Input((self.config.img_size, self.config.img_size, self.num_boundaries))
        logit = effectiveness_discriminator(predicted_boundary)
        self.discriminator = tf.keras.Model(inputs=predicted_boundary, outputs=logit, name='effectiveness_discriminator')

        logit = self.discriminator(boundary)
        self.gen_and_disc = tf.keras.Model(inputs=input_img, outputs=logit, name='generator_and_discriminator')

        self.generator.summary()
        self.regressor.summary()
        self.discriminator.summary()
