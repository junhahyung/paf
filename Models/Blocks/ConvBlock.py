import tensorflow as tf
if tf.__version__ >= '2.0.0':
    import tensorflow.keras.layers as nn
else:
    import keras.layers as nn

class Block(object):
    def __init__(self, in_filters, out_filters, stride, name, expansion=1):
        self.conv = nn.Conv2D(filters=in_filters * expansion,
                           kernel_size=1, strides=1, padding='same',
                           kernel_initializer='glorot_normal', name=f'{name}/k3conv')

        self.conv2 = nn.Conv2D(filters=out_filters,
                           kernel_size=1, strides=stride, padding='same',
                           kernel_initializer='glorot_normal', name=f'{name}/k3conv2')

        self.bn = nn.BatchNormalization(epsilon=1e-3, momentum=0.99, name=f'{name}/bn')
        self.bn2 = nn.BatchNormalization(epsilon=1e-3, momentum=0.99, name=f'{name}/bn2')

        self.activation = nn.ReLU()

        self.skip_connection = True if (stride == 1) & (in_filters == out_filters) else False

    def __call__(self, net):
        shortcut = net

        net = self.conv(net)
        net = self.bn(net)
        net = self.activation(net)

        net = self.conv2(net)
        net = self.bn2(net)

        if self.skip_connection:
            net = nn.Add()([shortcut, net])

        return net
