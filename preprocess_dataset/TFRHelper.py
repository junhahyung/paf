import numpy as np
import tensorflow as tf
import cv2
import math
import os

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature2(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def check_paths_exist(path_list):
    for path in path_list:
        if not os.path.exists(path):
            return False
    return True
  
def get_required_landmarks(landmarks):
    idx_to_keep = [7, 8, 9, # jaw
               27, 28, 29, 30, 31, 32, 33, 34, 35, # nose
               36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] # eyes
    
    temp = []
    for idx in idx_to_keep:
        temp.append(landmarks[idx])
    return temp

def get_required_landmarks_2(landmarks):
    jaw = landmarks[0:17]
    eyebrows = landmarks[17:27]
    nose = landmarks[27:36]
    re = landmarks[36:42]
    le = landmarks[42:48]
    mouth_exterior = landmarks[48:60]
    mouth_interior = landmarks[60:68]

    return np.concatenate([jaw, nose, re, le, mouth_exterior])

def is_img_valid(img):
    h, w, c = img.shape
    # resolution too small
    if max([h, w]) < 192:
        #         print('invalid')
        return False
    if (w == 0) | (h == 0):
        return False
    # image dimension invalid
    if( np.round(w/h) > 1.0) | (np.round(h/w) > 1.0):
        return False
    return True


def get_face_roi(img, landmarks, padding=0.25):
    img_h, img_w, _ = img.shape

    x, y = np.transpose(landmarks)

    left, top, right, bot = min(x), min(y), max(x), max(y)

    left, right = max(left, 0), min(right, img_w)
    top, bot = max(top, 0), min(bot, img_h)
    
    width = right - left
    height = bot - top
    
    max_size = int(max(height, width) * (padding + 1))

    def check_axis_validity(left, right, max_size, img_w):
        additional_h = 0
        w = right - left

        if w < max_size:
            additional_w = (max_size - w) / 2
            right = right + additional_w
            left = left - additional_w

        if right > img_w:
            additional_h = additional_h + ((right - img_w) / 2)     # negative number
            right = img_w

        if left < 0:
            additional_h = additional_h + (left / 2)    # negative number
            left = 0

        return left, right, additional_h

    iteration = 0
    cropping = True

    while cropping:

        left, right, additional_h = check_axis_validity(left, right, max_size, img_w)
        top, bot = top - additional_h, bot + additional_h

        top, bot, additional_w = check_axis_validity(top, bot, max_size, img_h)
        left, right = left - additional_w, right + additional_w
        iteration += 1

        if (int(additional_h) == 0) & (int(additional_w) == 0):
            cropping = False
        if iteration > 10:
            cropping = False

    return top, bot, left, right


def get_face_roi2(image, landmarks, target_width, target_height, padding):
    x, y = np.transpose(landmarks)
    max_width = image.shape[1]
    max_height = image.shape[0]
    ratio = target_width/target_height

    min_x = min(x)
    max_x = max(x)
    left = int(max(min_x, 0.0))
    right = int(min(max_x, max_width))
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

def get_face_roi3(image, landmarks):
    # tight face roi
    x, y = np.transpose(landmarks)
    max_width = image.shape[1]
    max_height = image.shape[0]

    min_x = min(x)
    max_x = max(x)
    left = int(max(min_x - 5, 0.0))
    right = int(min(max_x + 5, max_width))
    top = max(int(min(y) - 5), 0)
    bottom = min(int(max(y) + 5), max_height)

    return top, bottom, left, right


def crop_image(image, bbox):
    # top, bot, left, right = bbox
    top, bottom, left, right = [int(np.round(i)) for i in bbox]
    return image[top:bottom, left:right]


def get_new_landmark(landmark, bbox):
    top, bottom, left, right = bbox
    return landmark - np.array([left, top])


def get2DImagePoints(landmarks):
    jaws = landmarks[0:3]
    eye_right = landmarks[18:]
    eye_left = landmarks[12:18]
    nose_bridge = landmarks[3:7]
    nose_bottom = landmarks[9]
    
    image_points = np.vstack((eye_right, eye_left, nose_bridge, nose_bottom, jaws))

    return image_points


def getRVec(img_points_2d, focal_length, center):
    data = {
      "eye_right": [
            [-45.161,  -34.500, 35.797],
            [-39.287,  -39.759, 28.830],
            [-28.392,  -39.432, 27.056],
            [-19.184,  -33.718, 28.925],
            [-28.547,  -30.282, 29.844],
            [-38.151,  -30.684, 30.856]
      ],
      "eye_left": [
            [ 19.184,  -33.718, 28.925],
            [ 28.392,  -39.432, 27.056],
            [ 39.287,  -39.759, 28.830],
            [ 45.161,  -34.500, 35.797],
            [ 38.151,  -30.684, 30.856],
            [ 28.547,  -30.282, 29.844]
      ],
      "nose": [
            [  0.000,  -33.762, 16.068],
            [  0.000,  -22.746, 10.370],
            [  0.000,  -12.328,  3.639],
            [  0.000,  -0.000, 3.077]
      ],
      "nose_bottom": [
            [  0.000, 14.869, 13.731]
      ],
      "jaws": [
            [-14.591, 67.050, 27.722],
            [  0.000, 69.735, 26.787],
            [ 14.591, 67.050, 27.722]
      ]
    }

    eye_right   = np.array( data['eye_right'],   dtype='float32')
    eye_left    = np.array( data['eye_left'],    dtype='float32')
    nose        = np.array( data['nose'],        dtype='float32')
    nose_bottom = np.array( data['nose_bottom'], dtype='float32')
    jaws        = np.array( data['jaws'],        dtype='float32')

    coordinates_3d = np.vstack((eye_right, eye_left, nose, nose_bottom, jaws))
    
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ])

    # distortion_coefficient assumed no
    dist_coef = np.zeros((4, 1))
    ret, rvec, tvec = cv2.solvePnP(coordinates_3d, img_points_2d,
                                   camera_matrix, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
#     ret, rvec, tvec, inliers = cv2.solvePnPRansac(coordinates_3d, img_points_2d,
#                                    camera_matrix, dist_coef, flags=cv2.SOLVEPNP_EPNP)
    ret, rvec, tvec = cv2.solvePnP(coordinates_3d, img_points_2d,
                                   camera_matrix, dist_coef, rvec, tvec, True)

    return rvec, tvec


def isRotationMatrix(R):
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])