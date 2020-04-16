import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def points_in_circle_tf(radius):
    a = tf.range(radius+1)
    x = tf.where(tf.expand_dims(a,1)**2 + a**2 <= radius**2)
    x = tf.cast(x, tf.int32)
    x1 = tf.multiply(x, tf.constant([1, -1]))
    x2 = tf.multiply(x, tf.constant([-1, -1]))
    x3 = tf.multiply(x, tf.constant([-1, 1]))
    x = tf.concat([x,x1,x2,x3],axis=0)
    x = tf.constant(np.unique(x, axis=0))
    #yield from set(((x, y), (x, -y), (-x, y), (-x, -y),))
    return x

a = tf.constant([[1,2],[2,3.]])
b = tf.constant([[2,3],[3,4.]])
c = tf.linspace(a,b,3)
