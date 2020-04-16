from mtcnn import MTCNN
import os, sys
import time
sys.path.append('../')
import tensorflow as tf
import numpy as np
from preprocess_dataset.TFRHelper import *
from utils.image_preprocessor import *
import concurrent.futures
import multiprocessing

def is_img_valid(img):
    h, w, c = img.shape
    # resolution too small
    if max([h, w]) < 192:
        #         print('invalid')
        return False
    if (w == 0) | (h == 0):
        return False
    # image dimension invalid
    if(np.round(w/h) > 1.0) | (np.round(h/w) > 1.0):
        return False
    return True

def get_face_bbox(img, bbox, padding):
    img_h, img_w, c = img.shape

    top, bot, left, right = bbox
    top, bot, left, right = [top * img_h, bot * img_h, left * img_w, right * img_w]
    h = bot - top
    w = right - left
    max_size = max(w, h)
    max_size = max_size * (1 + padding)

    def check_axis_validity(left, right, max_size, img_w):
        additional_h = 0
        w = right - left

        if w < max_size:
            additional_w = (max_size - w) / 2
            right = right + additional_w
            left = left - additional_w

        if right > img_w:
            additional_h = (1 - right) / 2 # negative number
            right = img_w

        if left < 0:
            additional_h = left / 2# negative number
            left = 0

        return left, right, additional_h

    iteration = 0
    cropping = True
    while cropping:
        #         print(f'left: {left} \t right: {right}')
        left, right, additional_h = check_axis_validity(left, right, max_size, img_w)
        #         print(f'new -> left: {left} \t right: {right} \t add_h: {additional_h}')
        top, bot = top - additional_h, bot + additional_h
        #         print(f'top {top} \t bot: {bot}')
        top, bot, additional_w = check_axis_validity(top, bot, max_size, img_h)
        #         print(f'new -> top: {top} \t bot: {bot} \t add_w: {additional_w}')
        left, right = left - additional_w, right + additional_w
        iteration += 1

        if (int(additional_h) == 0) & (int(additional_w) == 0):
            cropping = False
        if iteration > 10:
            cropping = False

    return top, bot, left, right

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_one_image(img, results):

    if len(results) == 0:
        return None

    h, w, c = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    confidence = results[0]['confidence']

    if confidence < 0.8:
        return None

    bbox = results[0]['box']
    left, top, width, height= bbox
    right = left + width
    bottom = top + height

    top, bottom, left, right = top/h, bottom/h, left/w, right/w
    bbox = top, bottom, left, right
    face_bbox = get_face_bbox(img, bbox, 0.35)
    face = crop_image(rgb, face_bbox)

    valid = is_img_valid(face)

    if not valid:
        return None

    face = cv2.resize(face, (240, 240))

    face_byte = tf.io.encode_jpeg(face, quality=100)

    feature = {
        'image': _bytes_feature(face_byte)
    }

    return feature

def parse_one_subject(subject, writer):
    start = time.time()

    fids = os.listdir(os.path.join(data_path, subject))

    for fid in fids:
        img_path = os.path.join(data_path, subject, fid)
        img = cv2.imread(img_path)
        results = face_detector.detect_faces(img)

        feature = parse_one_image(img, results)

        if feature is not None:
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    end = time.time()
    return end - start


data_path = '/home/gabe/data-archive/vggface/train/'

if __name__ == '__main__':
    tf.executing_eagerly()
    tf.debugging.set_log_device_placement(False)
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

    face_detector = MTCNN()

    arg = int(sys.argv[-1])
    max_thread_num = multiprocessing.cpu_count()
    print(f'max_thread_num : {multiprocessing.cpu_count()}')

    subjects = os.listdir(data_path)
    subjects.sort()

    total_count = arg

    total_length = len(subjects)
    each_length = total_length // total_count

    for i in range(total_count):
        if i == total_count - 1:
            subjects = subjects[i * each_length :]
        subjects = subjects[i * each_length : (i + 1) * each_length]

        write_path = '/home/jarvis/data-archive/landmark-tfr/unlabeled'
        tfr_path = os.path.join(write_path, f'vggface_train-{i}.tfrecords')

        print(len(subjects), total_length, each_length)

        with tf.io.TFRecordWriter(tfr_path) as writer:
            for i, subject in enumerate(subjects):
                print(f'Doing {subject} - {i} /{len(subjects)}')
                elapsed_time = parse_one_subject(subject, writer)
                print(f'\tCompleted : {subject}... took {int(elapsed_time)} seconds')
