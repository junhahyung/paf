import tensorflow.keras.backend as K
import tensorflow as tf


def dist(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.math.square(y_pred - y_true), axis=-1)))


def dist_squared(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))


def dist2(y_true, y_pred):
    """
    :param y_true: todo test if i could pass landscape
    :param y_pred: heatmap shape (96, 96, 68)
    :return: euclidean distance between points
    """
    def apply_threshold(pred, threshold=0.8):
        out = tf.cast(pred > threshold, pred.dtype) * pred
        return out

    def get_xy_coord(pred):
        size = 96
        temp = tf.reshape([tf.linspace(0.0, size - 1.0, size)] * size, (size, size))
        x = tf.reduce_sum(temp * pred) / tf.reduce_sum(pred)
        y = tf.reduce_sum(tf.transpose(temp) * pred) / tf.reduce_sum(pred)
        return x, y

    pred_x, pred_y = [], []

    c = y_pred.shape[-1]
    y_pred = apply_threshold(y_pred, 0.70)

    for c in range(c):
        pred = y_pred[:, :, :, c]
        x, y = get_xy_coord(pred)

        # replace nan values todo how??????????????????????????? 이거부터 해
        x = tf.clip_by_value(x, -1.0, 192.0)
        y = tf.clip_by_value(y, -1.0, 192.0)

        pred_x.append(x)
        pred_y.append(y)

    y_pred = tf.concat([tf.reshape(pred_x, (-1, 1)), tf.reshape(pred_y, (-1, 1))], axis=-1)

    return dist(y_true, y_pred)


def normalized_mean_error(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true - y_pred)) / (tf.reduce_sum(tf.abs(tf.reduce_mean(y_pred) - y_pred)))


def convert_to_logits(y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    return tf.math.log(y_pred / (1 - y_pred))

def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=beta)
    return tf.reduce_mean(loss)

def weighted_cross_entropy(beta):
    """
    WCE is a variant of CE where all positive examples get weighted by some coefficient.
    It can be used in the case of class imbalance.

    set beta > 1 to decrease FALSE NEGATIVES
    set beta < 1 to decrease FALSE POSITIVES
    """
    beta = beta + tf.keras.backend.epsilon()

    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=beta)
        return tf.reduce_mean(loss)

    return loss


def balanced_cross_entropy(beta):
    """
     BCE is similar to WCE, but also weights negative examples.

    set beta > 1 to decrease FALSE NEGATIVES
    set beta < 1 to decrease FALSE POSITIVES
    """
    beta = beta + tf.keras.backend.epsilon()

    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=pos_weight)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))

    return loss


def dice_loss(y_true, y_pred):
    """
    Overlap measure - similar to IoU
    Dice Coefficient = 2TP / (2TP + FP + FN)

    TP : true positive
    FP : false positive
    FN : false negative
    """
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(tf.square(y_true) + tf.square(y_pred), axis=-1)

    return 1 - (numerator + 1) / (denominator + 1)


def combination_loss(y_true, y_pred):
    beta = 10
    return weighted_cross_entropy(beta)(y_true, y_pred) + tf.keras.losses.MSE(y_true, y_pred)

beta = 10
custom_losses = {
    'dist': dist,
    'dist_squared': dist_squared,
    'dist2': dist2,
    'normalized_mean_error': normalized_mean_error,
    'weighted_cross_entropy': weighted_cross_entropy(10),
    'balanced_cross_entropy': balanced_cross_entropy(10),
    'dice_loss': dice_loss,
    'combination_loss': combination_loss,
    'convert_to_logits': convert_to_logits,
    'loss': loss,
    'beta': beta
}