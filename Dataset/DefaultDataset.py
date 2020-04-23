from tensorflow.compat.v2.data import Dataset
from tensorflow.python.ops import random_ops
import tensorflow as tf
import tensorflow_addons as tfa
import os

file_shuffle_buffer = 100
frame_shuffle_buffer = 100
batch_shuffle_buffer = 80
interleave_cycle = 20

class TFDataset():
    def __init__(self, config, phase):
        self.feature_description = {
            'image': tf.io.VarLenFeature(tf.string),
            'landmarks': tf.io.FixedLenFeature([68*2], tf.float32),
            'headpose': tf.io.FixedLenFeature([3], tf.float32)
        }

        self.config = config
        self.phase = phase
        self.build_dataset()

    def __call__(self):
        temp = self.dataset.repeat(self.config.num_epochs)
        return temp

    def __len__(self):
        # todo: this is way too slow
        temp = self.dataset
        iter = temp.as_numpy_iterator()
        counter = 0
        for x in iter:
            counter += 1
        return counter

    def build_dataset(self):
        tfrecord_path = os.path.join(self.config.landmark_path, self.phase)
        self.dataset = self.read_tfrecord(tfrecord_path)

    # Read TFRecord Files in Directory
    def read_tfrecord(self, tfrecord_path):
        no_order_option = tf.data.Options()
        no_order_option.experimental_deterministic = False

        # Make TFRecord File list and shuffle
        tfrecord_data = tf.data.Dataset.list_files(os.path.join(tfrecord_path, '*.tfrecords'))\
            .with_options(no_order_option)\
            .interleave(tf.data.TFRecordDataset, cycle_length=interleave_cycle, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .shuffle(frame_shuffle_buffer)\
            .batch(self.config.batch_size, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        return tfrecord_data

    def _parse_features(self, example_raw):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_raw, self.feature_description)

    # todo: essentially this is the only part that needs to change
    @tf.function
    def preprocess(self, record):
        def eul2rot(theta):
            R = tf.cast([
                [tf.cos(theta[1]) * tf.cos(theta[2]),
                 tf.sin(theta[0]) * tf.sin(theta[1]) * tf.cos(theta[2]) - tf.sin(theta[2]) * tf.cos(theta[0]),
                 tf.sin(theta[1]) * tf.cos(theta[0]) * tf.cos(theta[2]) + tf.sin(theta[0]) * tf.sin(theta[2])],
                [tf.sin(theta[2]) * tf.cos(theta[1]),
                 tf.sin(theta[0]) * tf.sin(theta[1]) * tf.sin(theta[2]) + tf.cos(theta[0]) * tf.cos(theta[2]),
                 tf.sin(theta[1]) * tf.sin(theta[2]) * tf.cos(theta[0]) - tf.sin(theta[0]) * tf.cos(theta[2])],
                [-tf.sin(theta[1]),
                 tf.sin(theta[0]) * tf.cos(theta[1]),
                 tf.cos(theta[0]) * tf.cos(theta[1])]],
                tf.float32)
            return R

        def rot2eul(R):
            sy = tf.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            x = tf.atan2(R[2, 1], R[2, 2])
            y = tf.atan2(-R[2, 0], sy)
            z = tf.atan2(R[1, 0], R[0, 0])
            return [z, y, x]

        @tf.function
        def rotation_matrix(theta):
            # axis = axis/tf.sqrt(tf.tensordot(axis, axis, 1))
            a = tf.cos(theta/2.)
            b = -1 * tf.sin(theta / 2.0)
            c = 0 * tf.sin(theta / 2.0)
            d = 0 * tf.sin(theta / 2.0)
            # b, c, d = -axis * tf.sin(theta / 2.)
            return [[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                    [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                    [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]]

        def _add_random_noise_each(img):
            def _add_gaussian_noise(img, std):
                return img + tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=std, dtype=tf.float32)

            # img = _add_gaussian_noise(img, 0.01)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.7, 1.0)
            img = tf.image.random_hue(img, 0.1)
            img = tf.image.random_saturation(img, 0.7, 1.3)
            return tf.clip_by_value(img, 0.0, 1.0)

        record = self._parse_features(record)

        img = record['image'].values[0]
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32)

        # if self.phase == 'train':
        #     img = _add_random_noise_each(img)
        # img = tf.image.per_image_standardization(img)

        landmarks = record['landmarks']
        landmarks = tf.reshape(landmarks, (-1, 2))
        headpose = record['headpose']

        # if self.phase == 'train':
        # rotation
        theta = tf.random.uniform([], -3.14159265, 3.14159265)
        R_z = rotation_matrix(theta)
        R = tf.reshape(((tf.cos(theta), -tf.sin(theta)),
                        (tf.sin(theta), tf.cos(theta))), (2, 2))

        temp_rot = eul2rot(headpose)
        new_rmat = tf.matmul(temp_rot, R_z)
        headpose = rot2eul(new_rmat)

        rotated = tfa.image.rotate(img, theta, 'BILINEAR')

        centered_landmark = landmarks - tf.cast([0.5, 0.5], tf.float32)
        rotated_landmark = tf.matmul(centered_landmark, R)
        new_landmark = rotated_landmark + tf.cast([0.5, 0.5], tf.float32)

        size_x = tf.random.normal([1], 240, 24)
        size_x = tf.clip_by_value(size_x, 192, 288)
        size_x = tf.cast(size_x, tf.int32)
        size_y = tf.random.normal([1], 240, 24)
        size_y = tf.clip_by_value(size_y, 192, 288)
        size_y = tf.cast(size_y, tf.int32)

        size2d = tf.reshape([size_y, size_x], [2])
        resized = tf.image.resize(rotated, size2d)

        max_offset_x = tf.cast(size_x - 192, tf.float32)
        max_offset_y = tf.cast(size_y - 192, tf.float32)
        center_x = (max_offset_x - 0.0) / 4
        center_y = (max_offset_y - 0.0) / 4

        offset_x = tf.random.normal([1], center_x, center_x / 3)
        offset_y = tf.random.normal([1], center_y, center_y / 3)
        offset_x = tf.clip_by_value(offset_x, 0.0, max_offset_x)
        offset_y = tf.clip_by_value(offset_y, 0.0, max_offset_y)

        offset_x = tf.reshape(offset_x, ())
        offset_y = tf.reshape(offset_y, ())
        target = tf.cast(192, tf.int32)
        img = tf.image.crop_to_bounding_box(resized,
                                            tf.cast(offset_y, tf.int32),
                                            tf.cast(offset_x, tf.int32),
                                            target, target)
        img = tf.image.resize(img, (self.config.img_size, self.config.img_size))
        img = tf.image.per_image_standardization(img)

        # landmarks = new_landmark * tf.cast(size, tf.float32)
        size2d = tf.reshape([size_x, size_y], [2])
        landmarks = new_landmark * tf.cast(size2d, tf.float32)
        landmarks = landmarks - tf.reshape([offset_x, offset_y], [2])

        # else:
        #     # img = tf.image.crop_to_bounding_box(img, 24, 24, 192, 192)
        #     img = tf.image.resize(img, [240, 240])
        #     landmarks = landmarks * tf.cast(240, tf.float32)
        #     # landmarks = landmarks - [24, 24]

        return img, (landmarks, headpose)