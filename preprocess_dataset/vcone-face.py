import os, sys, json
import cv2
sys.path.append('../')
from preprocess_dataset.TFRHelper import *
from utils.device_data import device_data2 as device_data
import concurrent.futures
import multiprocessing
import tensorflow as tf
import numpy as np

def parse_data_load(npz_data):
    landmarks = npz_data['landmark']
    face_roi = npz_data['face_roi']
    landmarks = np.array(landmarks) + [face_roi[0], face_roi[1]]
    xy_cam = np.divide(npz_data['gaze_mm'], 10) # in centimeter
    device_name = npz_data['device']
    orientation = npz_data['orientation']

    return landmarks, xy_cam, device_name, orientation

def parse_img_load(image_data):
    byte_img = image_data['bytes']
    nparr = np.frombuffer(byte_img, np.uint8)
    decoded_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return decoded_frame

def get_face(img_data, data):
    temp_lm, xycam, device_name, orientation = parse_data_load(data)
    decoded_frame = parse_img_load(img_data)
    decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)

    x, y = np.transpose(temp_lm)
    face_bbox = get_face_roi(decoded_frame, temp_lm, padding=1)

    face = crop_image(decoded_frame, face_bbox)
    resized = cv2.resize(face, (240, 240))
    return resized

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(subject_path):
    device_json = json.load(open(os.path.join(subject_path, 'device.json')))
    # manufacturer = device_json['manufacturer'].lower()
    device_model = device_json['model'].lower()

    if (device_model not in list(device_data.keys())):
        print(f'Unsupported device: {device_model}')
        return None

    subject = os.path.basename(subject_path)
    sequences = os.listdir(subject_path)
    sequences = [x for x in sequences if '.npz' in x]
    tfrecord_path = os.path.join(write_path, f'{subject}.tfrecords')

    print(f"Converting {subject} to tfrecord - {len(sequences)}")

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for sequence in sequences:
            npz_file = np.load(os.path.join(subject_path, sequence), allow_pickle=True)
            img_load = npz_file['image'][()]
            data_load = npz_file['data'][()]

            frame_ids = img_load.keys()

            for frame_id in frame_ids:
                img_data = img_load[frame_id]
                data = data_load[frame_id]
                face = get_face(img_data, data)

                encoded_face = tf.io.encode_jpeg(face, quality=100).numpy()

                feature = {
                    'image': _bytes_feature(encoded_face),
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


data_path = '/media/gabe/ekko/data-archive/vc-one-preprocessed/'
write_path = os.path.join(data_path, 'vcone-face-tfr')

if __name__ == '__main__':
    crowdworks = os.listdir(os.path.join(data_path, 'crowdworks'))
    crowdwork_paths = [os.path.join(data_path, 'crowdworks', x) for x in crowdworks]

    mturks_folders = os.listdir(data_path + '/mturk')
    mturks_folders.remove('.DS_Store')
    mturks_folders.remove('._.DS_Store')
    mturk_paths = []
    for mturk_folder in mturks_folders:
        uids = os.listdir(os.path.join(data_path, 'mturk', mturk_folder))
        uids = [os.path.join(data_path, 'mturk', mturk_folder, uid) for uid in uids]
        mturk_paths.append(uids)
    mturk_paths = sum(mturk_paths, [])

    subject_paths = mturk_paths + crowdwork_paths

    max_thread_num = multiprocessing.cpu_count()
    print(f'max_thread_num : {multiprocessing.cpu_count()}')

    phases = ['train', 'val']

    if not os.path.exists(write_path):
        os.mkdir(write_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread_num) as executor:
        for subject_path in subject_paths:
            executor.submit(convert_to_tfrecord, subject_path=subject_path)
        convert_to_tfrecord(subject_path)