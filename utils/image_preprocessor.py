import cv2
import random
import tensorflow as tf
import numpy as np
from imgaug import augmenters as iaa

from utils.device_data import device_data2 as device_data

def random_augment(img):
    aug = [
        iaa.GaussianBlur(sigma=random.uniform(0, 0.5)),
        iaa.LinearContrast(random.uniform(0.5, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=random.uniform(0.0, 0.015 * 255)),
        iaa.Multiply(random.uniform(0.75, 1.5))
    ]

    random.shuffle(aug)
    seq = iaa.Sequential(aug)

    img = seq.augment_image(img)
    return img


def get_bbox(image, x, y, target_x, target_y, padding):
    max_width = image.shape[1]
    max_height = image.shape[0]
    ratio = target_x / target_y

    left, right = max(int(min(x)), 0), min(int(max(x)), max_width)
    top, bottom = max(int(min(y)), 0), min(int(max(y)), max_height)

    width = right - left
    height = bottom - top

    max_size = int(max(height * ratio, width) * (padding + 1))
    target_width = max_size
    target_height = max_size // ratio

    if target_width > width:
        left_padding = (target_width - width) // 2
        right_padding = (target_width - width) - left_padding

        if left - left_padding < 0:
            right_padding += left_padding - left
            left = 0
        else:
            left -= left_padding

        if right + right_padding > max_width:
            left -= (right + right_padding) - max_width
            right = max_width
        else:
            right = right + right_padding

        if left < 0:
            left = 0

    if target_height > height:
        bottom_padding = (target_height - height) // 2
        top_padding = (target_height - height) - bottom_padding

        if top - top_padding < 0:
            bottom += top_padding - top
            top = 0
        else:
            top -= top_padding

        if bottom + bottom_padding > max_height:
            top -= (bottom + bottom_padding) - max_height
            bottom = max_height
        else:
            bottom = bottom + bottom_padding

        if top < 0:
            top = 0
    return top, bottom, left, right


def cut_eye_image(image, bbox):
    top, bottom, left, right = bbox
    return image[int(top):int(bottom), int(left):int(right)]

def draw_dots(img, landmarks):
    dot_size = max(img.shape) // 64
    # todo: is there faster way // save this
    for i in range(0, 3):
        cv2.circle(img, (int(landmarks[i][0]), int(landmarks[i][1])), dot_size, (0, 0, 255), -1)
    for i in range(3, 12):
        cv2.circle(img, (int(landmarks[i][0]), int(landmarks[i][1])), dot_size, (0, 255, 0), -1)
    for i in range(12, 24):
        cv2.circle(img, (int(landmarks[i][0]), int(landmarks[i][1])), dot_size, (255, 0, 0), -1)

    return img

def create_landmark_frame(decoded_frame, landmarks, device_name, orientation):
    img_with_landmark = draw_dots(decoded_frame, landmarks)
    normalized_frame = normalized_fov(img_with_landmark, device_name, orientation)
    resized_frame = cv2.resize(normalized_frame, (128, 128))
    return resized_frame

def normalized_fov(img, device_name, orientation):
    normalize_focal_length_x = 580
    normalize_focal_length_y = 580
    normalize_roi_size = (640, 640)

    h, w, c = img.shape
    d = max([h, w])

    fx, fy, cx, cy = [float(x) for x in device_data[device_name.lower()]['matrix']]
    rx, ry = device_data[device_name.lower()]['resolution']

    if orientation == 0: # portrait
        ratio = max([rx, ry]) / d
        fx = fx / ratio
        fy = fy / ratio
    else: # landscape
        ratio = max([rx, ry]) / d
        temp = fx
        fx = fy / ratio
        fy = temp / ratio
    cx = w // 2
    cy = h // 2

    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0, 1 ]
    ])

    target_matrix = np.array([
        [normalize_focal_length_x, 0, normalize_roi_size[0] / 2],
        [0, normalize_focal_length_y, normalize_roi_size[1] / 2],
        [0, 0, 1.0],
    ])

    W = np.dot(target_matrix, np.linalg.inv(camera_matrix))
    img_warped = cv2.warpPerspective(img, W, normalize_roi_size)

    return img_warped
