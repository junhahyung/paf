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
    def __init__(self, config):
        self.feature_description = {
            'image': tf.io.VarLenFeature(tf.string),
        }

        self.config = config
        self.build_dataset()


    def __call__(self):
        # create dataset copy that ignores error
        temp_dataset = self.dataset.apply(tf.data.experimental.ignore_errors())
        return temp_dataset

    def build_dataset(self):
        tfrecord_path = os.path.join(self.config.unlabeled_path)
        self.dataset = self.read_tfrecord(tfrecord_path)

    # Read TFRecord Files in Directory
    def read_tfrecord(self, tfrecord_path):
        # Make TFRecord File list and shuffle
        tfrecord_data = tf.data.Dataset.list_files(tfrecord_path+'/*.tfrecords') \
            .interleave(tf.data.TFRecordDataset, cycle_length=interleave_cycle, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(self.config.max_num_threads, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        return tfrecord_data

    def _parse_features(self, example_raw):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_raw, self.feature_description)

    # todo: essentially this is the only part that needs to change
    @tf.function
    def preprocess(self, record):
        record = self._parse_features(record)
        byte_img = record['image'].values[0]
        img = tf.image.decode_jpeg(contents=byte_img,
                                   try_recover_truncated=True,
                                   acceptable_fraction=0.5)
        img = tf.image.convert_image_dtype(img, tf.float32)
        cropped = tf.image.crop_to_bounding_box(img, 24, 24, 192, 192)
        img = tf.image.per_image_standardization(cropped)
        img = tf.reshape(img, (192, 192, 3))
        img = tf.image.resize(img, (self.config.img_size, self.config.img_size))

        return byte_img, img
