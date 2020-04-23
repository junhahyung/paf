from Models.BaseModel import BaseModel
import tensorflow.keras.layers as layers
from utils.losses import *

import tensorflow as tf
import os

"""
Heatmap Regression Model
"""

class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.kernel = self.create_kernel(self.config.dot_size)
        self.build_model()
        self.model.summary()

    def create_kernel(self, dot_size):
        alpha = 1.0
        x_axis = tf.linspace(alpha, -alpha, dot_size)[:, None]
        y_axis = tf.linspace(alpha, -alpha, dot_size)[None, :]

        template = tf.sqrt(x_axis ** 2 + y_axis ** 2)
        template = tf.reduce_max(template) - template
        template = template / tf.reduce_max(template)

        kernel = tf.reshape([template] * self.config.num_landmarks, (self.config.num_landmarks, 5, 5, 1))
        kernel = tf.transpose(kernel, [1, 2, 0, 3])
        kernel = tf.cast(kernel, tf.float32)
        return tf.Variable(kernel, trainable=False)

    def build_model(self):
        def kernel_convolution(net):
            out = tf.nn.depthwise_conv2d(net, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
            out = layers.Activation('sigmoid', name='local')(out)
            return out

        def local_context_subnet():
            input_img = layers.Input((self.config.img_size, self.config.img_size, self.config.n_channel))

            # downsample a bit
            net = layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same')(input_img)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # (48 48 16)
            net = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # (24 24 32)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net)
            net = layers.Conv2D(64, kernel_size=1, strides=1, padding='same')(net)
            # net = layers.Add()([shortcut, net])
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(net)
            # net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(net)
            net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(net)
            net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(net)
            net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(net)
            net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(net)
            net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            shortcut = net
            net = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(net)
            net = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(net)
            net = layers.Add()([shortcut, net])
            # net = layers.ReLU()(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

            net = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                                         output_padding=1, use_bias=False)(net)
            net = layers.BatchNormalization(momentum=0.99)(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # net = layers.ReLU()(net)

            net = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',
                                         output_padding=1, use_bias=False)(net)
            net = layers.BatchNormalization(momentum=0.99)(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # net = layers.ReLU()(net)

            label = layers.Conv2D(self.config.num_landmarks, kernel_size=1, strides=1, padding='same')(net)
            local_out = kernel_convolution(label)

            model = tf.keras.Model(inputs=input_img, outputs=local_out, name='local_context_subnet')

            return model

        def regression_subnet(shape):
            input_img = layers.Input(shape)

            # downsample a bit
            net = layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same')(input_img)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # (48 48 16)

            net = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # (24 24 32)

            net = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # (12 12 64)

            net = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # (6 6 128)

            net = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(net)
            net = layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)
            # (3 3 256)

            net = layers.Conv2D(filters=self.config.num_landmarks * 2, kernel_size=1, strides=1, padding='valid')(net)
            out = tf.reshape(net, [-1, self.config.num_landmarks, 2], name='landmarks')

            model = tf.keras.Model(inputs=input_img, outputs=out, name='regression_subnet')
            return model


        input_img = layers.Input((self.config.img_size, self.config.img_size, self.config.n_channel))

        local_model = local_context_subnet()
        local_out = local_model(input_img)

        if self.config.finetune:
            model_name = '0327-local-subnet-ds5-wce10'
            model_dir = os.path.join('experiments', model_name, 'checkpoints')
            checkpoints = os.listdir(model_dir)
            checkpoints.remove('training.log')
            checkpoints.sort()
            print("Loading model checkpoint {} ...\n".format(os.path.join(model_dir, checkpoints[-1])))
            local_model.load_weights(os.path.join(model_dir, checkpoints[-1]))
            print("Model loaded")

        shape = local_out.shape
        # regression_model = regression_subnet(shape)
        # out = regression_model(local_out)

        # dilated convolutional layer
        net = layers.Conv2D(128, kernel_size=3, strides=1, dilation_rate=4, padding='same')(local_out)
        net = layers.Conv2D(128, kernel_size=3, strides=1, dilation_rate=4, padding='same')(net)
        net = layers.Conv2D(128, kernel_size=3, strides=1, dilation_rate=4, padding='same')(net)
        net = layers.Conv2D(128, kernel_size=3, strides=1, dilation_rate=4, padding='same')(net)
        #
        global_out = layers.Conv2D(self.config.num_landmarks, kernel_size=1, strides=1, padding='same', name='global')(net)
        out = layers.Add(name='out')([local_out, global_out])

        self.model = tf.keras.Model(inputs=input_img, outputs=out)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.model.compile(loss=weighted_cross_entropy(beta=10), optimizer=optimizer)

        # # load model if exists
        # if len(os.listdir(self.config.checkpoint_dir)) > 0:
        #     checkpoints = os.listdir(self.config.checkpoint_dir)
        #     checkpoints.sort()
        #     self.load(os.path.join(self.config.checkpoint_dir, checkpoints[-1]))