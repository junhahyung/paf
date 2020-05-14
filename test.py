import tensorflow as tf
import numpy as np
import os

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

feature_description = {
    'image': tf.io.VarLenFeature(tf.string),
    'landmarks': tf.io.FixedLenFeature([68*2], tf.float32),
    'headpose': tf.io.FixedLenFeature([3], tf.float32),
    'paf_x': tf.io.VarLenFeature(tf.string),
    'paf_y': tf.io.VarLenFeature(tf.string),
}

def _parse_features(example_raw):
    return tf.io.parse_single_example(example_raw, feature_description)

@tf.function
def preprocess(record):
    @tf.function
    def rotation_matrix(theta):
        a = tf.cos(theta/2.)
        b = -1 * tf.sin(theta / 2.0)
        c = 0 * tf.sin(theta / 2.0)
        d = 0 * tf.sin(theta / 2.0)
        return [[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]]

    def _add_random_noise_each(img):
        def _add_gaussian_noise(img, std):
            return img + tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=std, dtype=tf.float32)

        img = _add_gaussian_noise(img, 0.035)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.7, 1.0)
        img = tf.image.random_hue(img, 0.1)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        return tf.clip_by_value(img, 0.0, 1.0)

    def get_random_size():
        size_x = tf.random.normal([1], 216, 12) # 240, 24
        size_x = tf.clip_by_value(size_x, 192, 240) #192, 240
        size_x = tf.cast(size_x, tf.int32)
        size_y = tf.random.normal([1], 216, 12) # 240, 24
        size_y = tf.clip_by_value(size_y, 192, 240)
        size_y = tf.cast(size_y, tf.int32)
        size2d = tf.reshape([size_y, size_x], [2])
        return size_x, size_y, size2d

    def get_random_offset(size_x, size_y):
        max_offset_x = tf.cast(size_x - 192, tf.float32)
        max_offset_y = tf.cast(size_y - 192, tf.float32)
        center_x = (max_offset_x - 0.0) / 4
        center_y = (max_offset_y - 0.0) / 4

        offset_x = tf.random.normal([1], center_x, center_x / 3)
        offset_y = tf.random.normal([1], center_y, center_y / 3)
        offset_x = tf.clip_by_value(offset_x, 0.0, max_offset_x)
        offset_y = tf.clip_by_value(offset_y, 0.0, max_offset_y)
        return offset_x, offset_y

    def get_n_landmarks(landmarks):
        jaw = landmarks[0:17] # 17
        eyebrows = landmarks[17:27] # 10
        nose = landmarks[27:36] # 9
        re = landmarks[36:42] # 6
        le = landmarks[42:48] # 6
        mouth_exterior = landmarks[48:60] # 12
        mouth_interior = landmarks[60:] # 8

        return tf.concat([jaw, eyebrows, nose, re, le, mouth_exterior], axis=0)


    def parse_boundary_heatmap(record):

        jaw = record['jaw'].values[0]
        jaw = tf.image.decode_jpeg(jaw)
        jaw = tf.image.convert_image_dtype(jaw, tf.float32)

        right_eyebrow = record['right_eyebrow'].values[0]
        right_eyebrow = tf.image.decode_jpeg(right_eyebrow)
        right_eyebrow = tf.image.convert_image_dtype(right_eyebrow, tf.float32)

        left_eyebrow = record['left_eyebrow'].values[0]
        left_eyebrow = tf.image.decode_jpeg(left_eyebrow)
        left_eyebrow = tf.image.convert_image_dtype(left_eyebrow, tf.float32)

        nose_vert = record['nose_vert'].values[0]
        nose_vert = tf.image.decode_jpeg(nose_vert)
        nose_vert = tf.image.convert_image_dtype(nose_vert, tf.float32)

        nose_hori = record['nose_hori'].values[0]
        nose_hori = tf.image.decode_jpeg(nose_hori)
        nose_hori = tf.image.convert_image_dtype(nose_hori, tf.float32)

        re_upper = record['re_upper'].values[0]
        re_upper = tf.image.decode_jpeg(re_upper)
        re_upper = tf.image.convert_image_dtype(re_upper, tf.float32)

        re_lower = record['re_lower'].values[0]
        re_lower = tf.image.decode_jpeg(re_lower)
        re_lower = tf.image.convert_image_dtype(re_lower, tf.float32)

        le_upper = record['le_upper'].values[0]
        le_upper = tf.image.decode_jpeg(le_upper)
        le_upper = tf.image.convert_image_dtype(le_upper, tf.float32)

        le_lower = record['le_lower'].values[0]
        le_lower = tf.image.decode_jpeg(le_lower)
        le_lower = tf.image.convert_image_dtype(le_lower, tf.float32)

        mouth_upper = record['mouth_upper'].values[0]
        mouth_upper = tf.image.decode_jpeg(mouth_upper)
        mouth_upper = tf.image.convert_image_dtype(mouth_upper, tf.float32)

        mouth_lower = record['mouth_lower'].values[0]
        mouth_lower = tf.image.decode_jpeg(mouth_lower)
        mouth_lower = tf.image.convert_image_dtype(mouth_lower, tf.float32)
        boundary = tf.stack([jaw,
                             right_eyebrow, left_eyebrow,
                             nose_vert, nose_hori,
                             re_upper, re_lower,
                             le_upper, le_lower,
                             mouth_upper, mouth_lower])
        boundary = tf.reshape(boundary, (11, 240, 240))
        boundary = tf.transpose(boundary, [1, 2, 0])
        return boundary

    record = _parse_features(record)

    # get image
    img = record['image'].values[0]
    img = tf.image.decode_jpeg(img)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = _add_random_noise_each(img)

    # get landmarks
    landmarks = record['landmarks']
    landmarks = tf.reshape(landmarks, (-1, 2))
    landmarks = get_n_landmarks(landmarks)

    # random noises

    theta = tf.random.uniform([], -3.14159265, 3.14159265)
    R = tf.reshape(((tf.cos(theta), -tf.sin(theta)),
                    (tf.sin(theta), tf.cos(theta))), (2, 2))
    # size_x, size_y, size2d = get_random_size()
    size_x, size_y, size2d = 192, 192, tf.reshape([192, 192], [2])
    offset_x, offset_y = get_random_offset(size_x, size_y)
    offset_x = tf.reshape(offset_x, ())
    offset_y = tf.reshape(offset_y, ())

    # landmarks

    resized_landmark = landmarks * tf.cast(tf.reshape([size_x, size_y], [2]), tf.float32)
    offseted_landmark = resized_landmark - tf.reshape([offset_x, offset_y], [2])
    normed_landmark = offseted_landmark / 192.0
    landmark = normed_landmark * 240

    # image
    resized = tf.image.resize(img, size2d)
    img = tf.image.crop_to_bounding_box(resized,
                                        tf.cast(offset_y, tf.int32),
                                        tf.cast(offset_x, tf.int32),
                                        192, 192)
    img = tf.image.resize(img, (240, 240))

    img = tf.image.per_image_standardization(img)
    paf_x = record['paf_x'].values[0]
    paf_y = record['paf_y'].values[0]
    paf_x = tf.image.decode_jpeg(paf_x)
    paf_x = tf.image.convert_image_dtype(paf_x, tf.float32)
    paf_y = tf.image.decode_jpeg(paf_y)
    paf_y = tf.image.convert_image_dtype(paf_y, tf.float32)
    paf = tf.stack([paf_x, paf_y], axis=-1)
    paf = tf.linalg.normalize(paf, axis=-1)[0]
    paf = tf.squeeze(paf)
    paf = tf.where(tf.math.is_nan(paf), tf.zeros_like(paf), paf)

    return img, (landmark, paf)



tfrecord_path = "/home/liam/mario/data-archive/paf/train/"
no_order_option = tf.data.Options()
no_order_option.experimental_deterministic = False
tfrecord_data = tf.data.Dataset.list_files(os.path.join(tfrecord_path, '*.tfrecords'), shuffle=False) \
        .with_options(no_order_option) \
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .apply(tf.data.experimental.ignore_errors()) \
        .batch(32, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE)

d = iter(tfrecord_data)
x,y = next(d)
#print(y[-1][0]==127)
paf = y[1][2]
for i in range(240):
    print(paf[i])
