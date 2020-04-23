from Models.BaseModel import BaseModel
import tensorflow.keras.layers as layers
from utils.losses import *

import tensorflow as tf

"""
Face Alignment Model 
https://github.com/1adrianb/face-alignment/blob/master/face_alignment/models.py
"""

class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.build_model()
        self.model.summary()

    def build_model(self):
        def conv3x3(filters, stride=1, padding='same', bias=False):
            return layers.Conv2D(filters=filters,
                                 kernel_size=(3, 3),
                                 strides=(stride, stride),
                                 padding=padding,
                                 use_bias=bias)

        def conv_block(net, out_filters):
            in_filters = net.shape[-1]

            residual = net

            net = layers.BatchNormalization()(net)
            net = layers.ReLU()(net)
            out1 = conv3x3(out_filters // 2)(net)

            net = layers.BatchNormalization()(out1)
            net = layers.ReLU()(net)
            out2 = conv3x3(out_filters // 4)(net)

            net = layers.BatchNormalization()(out2)
            net = layers.ReLU()(net)
            out3 = conv3x3(out_filters // 4)(net)

            out = layers.Concatenate()([out1, out2, out3])

            if in_filters != out_filters:
                residual = layers.BatchNormalization()(residual)
                residual = layers.ReLU()(residual)
                residual = layers.Conv2D(out_filters, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(residual)

            out = layers.Add()([out, residual])

            return out

        def hourglass(net, depth):

            def forward(level, net):
                up1 = net
                up1 = conv_block(up1, 128)
                low1 = layers.AveragePooling2D()(net)
                low1 = conv_block(low1, 128)

                if level > 1:
                    low2 = forward(level - 1, low1)
                else:
                    low2 = low1
                    low2 = conv_block(low2, 128)

                low3 = low2
                low3 = conv_block(low3, 128)
                up2 = layers.UpSampling2D(size=(2, 2))(low3)

                return layers.Add()([up1, up2])

            return forward(depth, net)

        num_modules = 2
        inputs = layers.Input(shape=(self.config.img_size, self.config.img_size, self.config.n_channel),
                              name='input_img')

        conv1 = layers.Conv2D(32, kernel_size=7, strides=2, padding='same')(inputs)
        bn1 = layers.BatchNormalization(axis=1,momentum=0.1)(conv1)
        act1 = layers.Activation('relu')(bn1)

        conv2 = conv_block(act1, 64)
        avgP1 = layers.AveragePooling2D(2, strides=2)(conv2)

        conv3 = conv_block(avgP1, 64)
        conv4 = conv_block(conv3, 128)

        previous = conv4
        outputs = []

        for i in range(num_modules):
            hg = hourglass(previous, depth=3)
            ll = hg
            ll = conv_block(ll, 128)
            ll = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(ll)
            bn_temp = layers.BatchNormalization()(ll)
            act_temp = layers.ReLU()(bn_temp)

            tmp_out = layers.Conv2D(68, kernel_size=1, strides=1, padding='same')(act_temp)
            outputs.append(tmp_out)

            if i < num_modules - 1:
                ll = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(act_temp)
                tmp_out_ = layers.Conv2D(128, kernel_size=1, strides=1, padding='same')(tmp_out)
                previous = layers.Add()([previous, ll, tmp_out_])

        net = layers.Add()(outputs)

        net = layers.Conv2DTranspose(filters=68, kernel_size=3, strides=2, padding='same', output_padding=1, use_bias=False)(net)
        net = layers.ReLU()(net)
        net = layers.Conv2DTranspose(filters=68, kernel_size=3, strides=2, padding='same', output_padding=1, use_bias=False)(net)

        output = layers.Conv2D(68, kernel_size=1, strides=1, activation='softmax', padding='same')(net)
        '''end of model'''

        # compile model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
        self.model.compile(loss=combination_loss, metrics=['mse', weighted_cross_entropy], optimizer='adam')

        # # load model if exists
        # if len(os.listdir(self.config.checkpoint_dir)) > 0:
        #     checkpoints = os.listdir(self.config.checkpoint_dir)
        #     checkpoints.sort()
        #     self.load(os.path.join(self.config.checkpoint_dir, checkpoints[-1]))
