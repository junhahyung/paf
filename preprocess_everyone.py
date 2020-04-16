import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import cv2

import os, glob, json
import math, time
import copy

from preprocess_dataset.TFRHelper import *

frame_path = '/mnt/SSD1/everyone/val/frames'
json_path = '/mnt/SSD1/everyone/val/data'

generator_path = 'experiments/0406-bafa/checkpoints/158-generator.hdf5'
regressor_path = 'experiments/0403-smaller-kernel-w-everyone/checkpoints/033-regressor.hdf5'

generator = tf.keras.models.load_model(generator_path)
regressor = tf.keras.models.load_model(regressor_path)

IMG_SIZE = 96

compare_date = '2020_04_10'

def resizeInput(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = image.astype(np.float32)
    output = output.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    output = tf.image.per_image_standardization(output)

    return output

def getLandmarks(frame, data):
    bbox = get_face_roi2(frame, data, IMG_SIZE, IMG_SIZE, 1)
    face = crop_image(frame, bbox)
    input_data = resizeInput(face)

    bm_ = generator(input_data)
    landmarks = regressor([input_data, bm_])[0]

    h, w, _ = face.shape
    for i in range(len(landmarks)):
        landmarks[i][0] *= (w / IMG_SIZE)
        landmarks[i][0] += bbox[2]
        landmarks[i][1] *= (h / IMG_SIZE)
        landmarks[i][1] += bbox[0]

    return landmarks


def main():
    print('\n\n==========Start==========\n')
    frame_folders = sorted(glob.glob(os.path.join(frame_path, '*')))
    json_folders = sorted(glob.glob(os.path.join(json_path, '*')))

    frame_all = []
    json_all = []
    for i in frame_folders:
        frame_all += sorted(glob.glob(os.path.join(i, '*')))

    for i in json_folders:
        json_all += sorted(glob.glob(os.path.join(i, '*')))

    print(len(frame_all), len(json_all), '=>', end=' ')
    for i in range(len(json_all)):
        while frame_all[i].split('.')[0].split(os.sep)[-1] != json_all[i].split('.')[0].split(os.sep)[-1]:
            del frame_all[i]
    print(len(frame_all), len(json_all))
    t_begin = time.time()

    for index in range(len(json_all)):
        with open(json_all[index], 'r') as f:
            data = json.load(f)

        if not 'vcft' in data.keys():
            print('key \'vcft\' not found in', json_all[index])
            continue

        if not 'face_landmarks' in data['vcft'].keys():
            print('key \'face_landmarks\' not found in', json_all[index])
            continue

        if 'compare' in data.keys():
            if compare_date in data['compare'].keys():
                print('%d / %d' % (index+1, len(json_all)), json_all[index], 'is already processed')
                continue

#         frame = cv2.imread(frame_all[index])
#         h, w, _ = frame.shape
#         landmarks_old = data['vcft']['face_landmarks']
#         landmarks_old = np.reshape(landmarks_old, (-1, 2))
#         ​
#         # landmarks_new = getLandmarks(frame, landmarks_old)
# ​
#         data['compare'] = {
#             compare_date : {
#                 'landmarks' : (landmarks_new.tolist()),
#                 'MSE': error,
#                 'MSE_REL': error_rel
#             }
#         }
#         with open(json_all[index], 'w') as json_file:
#             json.dump(data, json_file, indent=2)

    print('\n==========END==========\n\n')


if __name__ == '__main__':
    main()