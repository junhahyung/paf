import tensorflow as tf
if tf.__version__ >= '2.0.0':
    import tensorflow.keras.layers as nn
else:
    import keras.layers as nn

class Block(object):
    def __init__(self, squeeze_filters, expanded_filters, name):
        self.conv = nn.Conv2D(filters=squeeze_filters,
                           kernel_size=1, strides=1, padding='same',
                           kernel_initializer='glorot_normal', name=f'{name}/sq-1x1conv')

        self.ex_conv = nn.Conv2D(filters=expanded_filters,
                            kernel_size=1, strides=1, padding='same',
                            kernel_initializer='glorot_normal', name=f'{name}/ex-1x1conv')
        # self.squeeze = SqueezeNet(net, in_filters * expansion, name=name+'/squeeze')
        self.ex_conv2 = nn.Conv2D(filters=expanded_filters,
                               kernel_size=3, strides=1, padding='same',
                               kernel_initializer='glorot_normal', name=f'{name}/ex-3x3conv')

    def __call__(self, net):
        sq = self.conv(net)
        sq = nn.ReLU()(sq)

        exp = self.ex_conv(net)
        exp = nn.ReLU()(exp)
        exp = self.ex_conv2(exp)
        exp = nn.ReLU()(exp)

        return nn.Concatenate(axis=-1)([sq, exp])
