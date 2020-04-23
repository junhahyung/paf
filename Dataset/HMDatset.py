from tensorflow.compat.v2.data import Dataset
from tensorflow.python.ops import random_ops
import tensorflow as tf
import tensorflow_addons as tfa
import os

from preprocess_dataset.TFRHelper import get_required_landmarks_2

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
        self.num_landmarks = self.config.num_landmarks
        self.create_heatmap_template()
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

    def create_heatmap_template(self):
        # make a canvas
        dot_size = self.config.dot_size * 2
        template_size = (96 + 96 + 24 + 24) * 2 - 1

        alpha = 24.0
        x_axis = tf.linspace(alpha, -alpha, dot_size)[:, None]
        y_axis = tf.linspace(alpha, -alpha, dot_size)[None, :]

        template = tf.sqrt(x_axis ** 2 + y_axis ** 2)
        template = tf.reduce_max(template) - template
        template = template / tf.reduce_max(template)

        self.kernel = tf.reshape([template] * self.num_landmarks, (self.num_landmarks, dot_size, dot_size, 1))
        self.kernel = tf.transpose(self.kernel, [1, 2, 0, 3])
        # self.kernel = tf.cast(self.kernel, tf.float32)

        self.labeled_template = tf.scatter_nd([[192 + 24, 192 + 24]], [1.0], (template_size, template_size))
        temp = tf.nn.conv2d(self.labeled_template[None, :, :, None], template[:, :, None, None], 1, 'SAME')[0, :, :, 0]
        self.gaussian_distribution_template = tf.expand_dims(temp, -1)
        self.labeled_template = tf.reshape(self.labeled_template, (template_size, template_size, 1))

    def build_dataset(self):
        tfrecord_path = os.path.join(self.config.landmark_path, self.phase)
        self.dataset = self.read_tfrecord(tfrecord_path)

    # Read TFRecord Files in Directory
    def read_tfrecord(self, tfrecord_path):
        no_order_option = tf.data.Options()
        no_order_option.experimental_deterministic = False

        # Make TFRecord File list and shuffle
        tfrecord_data = tf.data.Dataset.list_files(os.path.join(tfrecord_path, '*.tfrecords'), shuffle=False) \
            .with_options(no_order_option) \
            .interleave(tf.data.TFRecordDataset, cycle_length=interleave_cycle, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .apply(tf.data.experimental.ignore_errors()) \
            .shuffle(frame_shuffle_buffer) \
            .batch(self.config.batch_size, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        return tfrecord_data

    def _parse_features(self, example_raw):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_raw, self.feature_description)

    # todo: essentially this is the only part that needs to change
    @tf.function
    def preprocess(self, record):
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

        @tf.function
        def _draw_heatmap(landmarks):
            size = 192
            # create heatmap
            landmarks = tf.cast(landmarks * [size, size], tf.int32)

            heatmap = []
            label = []
            # x, y = tf.transpose(landmarks, [1, 0])
            for i in range(self.num_landmarks):
                x = landmarks[i][0]
                y = landmarks[i][1]
                inv_x = 192 - x + 24
                inv_y = 192 - y + 24
                hm = tf.image.crop_to_bounding_box(self.gaussian_distribution_template, inv_y, inv_x, 192, 192)
                heatmap.append(hm)
                # label.append(tf.image.crop_to_bounding_box(self.labeled_template, inv_y, inv_x, 96, 96))

            heatmap = tf.reshape(heatmap, (self.num_landmarks, 192, 192))
            heatmap = tf.transpose(heatmap, [1, 2, 0])

            # label = tf.reshape(label, (self.num_landmarks, 96, 96))
            # label = tf.transpose(label, [1, 2, 0])

            return heatmap#, label

        def _draw_heatmap_2(landmarks):
            size = self.config.img_size
            landmarks = tf.reshape(landmarks * size, (-1, 2))

            idx = tf.linspace(0.0, 49.0, self.num_landmarks)
            idx = tf.reshape(idx, (-1, 1))
            temp = tf.concat([landmarks, idx], axis=-1)
            temp = tf.cast(temp, tf.int32)

            label = tf.scatter_nd(temp, [1.0] * self.num_landmarks, (size, size, self.num_landmarks))

            input_tensor = tf.reshape(label, (1, size, size, self.num_landmarks))
            heatmap = tf.nn.depthwise_conv2d(input_tensor, self.kernel, strides=[1, 1, 1, 1], padding='SAME')[0]

            heatmap = tf.image.flip_up_down(heatmap)
            heatmap = tf.image.rot90(heatmap, 3)

            return heatmap, label

        def get_random_size():
            size_x = tf.random.normal([1], 240, 24)
            size_x = tf.clip_by_value(size_x, 192, 240)
            size_x = tf.cast(size_x, tf.int32)
            size_y = tf.random.normal([1], 240, 24)
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

        def get_n_landmarks(landmarks, n=68):
            jaw = landmarks[0:17] # 17
            eyebrows = landmarks[17:27] # 10
            nose = landmarks[27:36] # 9
            re = landmarks[36:42] # 6
            le = landmarks[42:48] # 6
            mouth_exterior = landmarks[48:60] # 12
            mouth_interior = landmarks[60:] # 8

            if n == 68:
                return landmarks
            if n == 50:
                # 17 + 9 + 6 + 6 + 12 = 50
                return  tf.concat([jaw, nose, re, le, mouth_exterior], axis=0)
            if n == 36:
                # 3 + 9 + 6 + 6 + 12 = 36
                jaw_part = jaw[7:10]
                return  tf.concat([jaw_part, nose, re, le, mouth_exterior], axis=0)

        record = self._parse_features(record)

        # get image
        img = record['image'].values[0]
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32)

        if self.phase == 'train':
            img = _add_random_noise_each(img)

        img = tf.image.per_image_standardization(img)

        # get landmarks
        landmarks = record['landmarks']
        landmarks = tf.reshape(landmarks, (-1, 2))

        landmarks = get_n_landmarks(landmarks, self.num_landmarks)

        # create heatmap
        heatmap = _draw_heatmap(landmarks)

        # random noises
        theta = tf.random.uniform([], -3.14159265, 3.14159265)
        R = tf.reshape(((tf.cos(theta), -tf.sin(theta)),
                        (tf.sin(theta), tf.cos(theta))), (2, 2))

        size_x, size_y, size2d = get_random_size()
        offset_x, offset_y = get_random_offset(size_x, size_y)
        offset_x = tf.reshape(offset_x, ())
        offset_y = tf.reshape(offset_y, ())

        # landmarks
        centered_landmark = landmarks - tf.cast([0.5, 0.5], tf.float32)
        rotated_landmark = tf.matmul(centered_landmark, R)
        rotated_landmark = rotated_landmark + tf.cast([0.5, 0.5], tf.float32)
        resized_landmark = rotated_landmark * tf.cast(tf.reshape([size_x, size_y], [2]), tf.float32)
        offseted_landmark = resized_landmark - tf.reshape([offset_x, offset_y], [2])
        normed_landmark = offseted_landmark / 192.0
        landmark = normed_landmark * self.config.img_size

        # image
        rotated = tfa.image.rotate(img, theta, 'BILINEAR')
        resized = tf.image.resize(rotated, size2d)
        img = tf.image.crop_to_bounding_box(resized,
                                            tf.cast(offset_y, tf.int32),
                                            tf.cast(offset_x, tf.int32),
                                            192, 192)
        img = tf.image.resize(img, (self.config.img_size, self.config.img_size))

        # heatmap
        rotated_hm = tfa.image.rotate(heatmap, theta, 'BILINEAR')
        resized_hm = tf.image.resize(rotated_hm, size2d)
        heatmap = tf.image.crop_to_bounding_box(resized_hm,
                                                tf.cast(offset_y, tf.int32),
                                                tf.cast(offset_x, tf.int32),
                                                192, 192)
        heatmap = tf.image.resize(heatmap, (self.config.img_size, self.config.img_size))

        # label
        # rotated_lb = tfa.image.rotate(label, theta, 'BILINEAR')
        # resized_lb = tf.image.resize(rotated_lb, size2d)
        # label = tf.image.crop_to_bounding_box(resized_lb,
        #                                       tf.cast(offset_y, tf.int32),
        #                                       tf.cast(offset_x, tf.int32),
        #                                       target, target)
        # label = tf.image.resize(label, (self.config.img_size, self.config.img_size))

        return img, heatmap