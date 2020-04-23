import os
import sys
sys.path.append('../')
import json
from tqdm import tqdm
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
import concurrent.futures
import multiprocessing

from utils.image_preprocessor import *
from TFRHelper import *

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


def get_everyone_data(fid):
    json_path = os.path.join(data_path, 'data', fid + '.json')
    frame_path = os.path.join(data_path, 'frames', fid + '.jpg')


    img = cv2.imread(frame_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = json.load(open(json_path, 'r'))

    if not 'vcft' in data.keys():
        return None
    if not 'face_landmarks' in data['vcft'].keys():
        return None

    all_landmarks = data['vcft']['face_landmarks']

    all_landmarks = np.reshape(all_landmarks, (-1, 2))

    return img, all_landmarks, data


def process_frame_to_face(img, landmarks):
    face_bbox = get_face_roi3(img, landmarks)
    face = crop_image(img, face_bbox)

    landmarks = get_new_landmark(landmarks, face_bbox)

    h, w, c = face.shape

    landmarks = landmarks / [w, h]

    return face, landmarks


def get_boundary_map(pts2d, img):
    landmarks = pts2d

    jaw = landmarks[0:17]
    right_eyebrow = landmarks[17:22]
    left_eyebrow = landmarks[22:27]
    nose_vert = landmarks[27:31]
    nose_hori = landmarks[31:36]
    re_upper = landmarks[36:40]
    re_lower = landmarks[39:42] + landmarks[36:37]
    le_upper = landmarks[42:46]
    le_lower = landmarks[45:48] + landmarks[42:43]
    mouth_upper = landmarks[48:54] + landmarks[54:55]
    mouth_lower = landmarks[54:60] + landmarks[48:49]

    boundaries = [jaw,
                  right_eyebrow, left_eyebrow,
                  nose_vert, nose_hori,
                  re_upper, re_lower,
                  le_upper, le_lower,
                  mouth_upper, mouth_lower]
    h,w,c, = img.shape
    heatmap = []

    pad = 0
    for landmark in boundaries:
        pts = np.pad(landmark, [(pad,pad), (0,0)], mode='wrap')
        x, y = np.transpose(landmark)
        i = np.arange(0, len(landmark))

        interp_i = np.linspace(0, i.max(), 5 * i.size)

        xi = interp1d(i, x, kind='cubic')(interp_i)
        yi = interp1d(i, y, kind='cubic')(interp_i)

        pts = np.reshape(np.stack([xi, yi], 1), (-1, 2))
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))

        temp = np.zeros((h, w))
        temp = cv2.polylines(temp, [pts], False, (200), 1)

        heatmap.append(temp)

    heatmap = np.reshape(heatmap, (len(boundaries), h, w))
    heatmap = np.transpose(heatmap, [1, 2, 0])

    return heatmap

def get_paf(lms, img):
    landmarks = lms

    jaw = landmarks[0:17]
    right_eyebrow = landmarks[17:22]
    left_eyebrow = landmarks[22:27]
    nose_vert = landmarks[27:31]
    nose_hori = landmarks[31:36]
    re_upper = landmarks[36:40]
    re_lower = landmarks[39:42] + landmarks[36:37]
    le_upper = landmarks[42:46]
    le_lower = landmarks[45:48] + landmarks[42:43]
    mouth_upper = landmarks[48:54] + landmarks[54:55]
    mouth_lower = landmarks[54:60] + landmarks[48:49]

    boundaries = [jaw,
                  right_eyebrow, left_eyebrow,
                  nose_vert, nose_hori,
                  re_upper, re_lower,
                  le_upper, le_lower,
                  mouth_upper, mouth_lower]
    
    eyebrow_jaw_r = landmarks[0:1] + landmarks[17:18]
    eyebrow_jaw_l = landmarks[16:17] + landmarks[26:27]
    nose_eye_r = landmarks[27:28] + landmarks[39:40]
    nose_eye_l = landmarks[27:28] + landmarks[42:43]
    jaw_mouth_r = landmarks[0:1] + landmarks[48:49]
    jaw_mouth_l = landmarks[16:17] + landmarks[54:55]
    horinose_eye_r = landmarks[31:32] + landmarks[39:40]
    horinose_eye_l = landmarks[35:36] + landmarks[42:43]
    eye_jaw_r = landmarks[0:1] + landmarks[36:37]
    eye_jaw_l = landmarks[16:17] + landmarks[45:46]
    eyebrow_connect = landmarks[21:23]
    nose_connect_r = landmarks[30:31] + landmarks[31:32]
    nose_connect_l = landmarks[30:31] + landmarks[35:36]
    eye_eyebrow_r = landmarks[21:22] + landmarks[39:40]
    eye_eyebrow_l = landmarks[22:23] + landmarks[42:43]
    nose_mouth_r = landmarks[31:32] + landmarks[48:49]
    nose_mouth_l = landmarks[35:36] + landmarks[54:55]
    
    # connecting two boundaries
    connects = [eyebrow_jaw_l, eyebrow_jaw_r, 
                nose_eye_r, nose_eye_l,
                jaw_mouth_l, jaw_mouth_r,
               horinose_eye_l, horinose_eye_r,
               eye_jaw_r, eye_jaw_l,
               nose_connect_r, nose_connect_l, eyebrow_connect,
               eye_eyebrow_r, eye_eyebrow_l,
               nose_mouth_r, nose_mouth_l]
    
    boundaries = [np.array(p) for p in boundaries]
    connects = [np.array(p) for p in connects]
    h,w,c, = img.shape
    paf_x = []
    paf_y = []
    
    for landmark in boundaries:
        llandmark = landmark[1:]
        rlandmark = landmark[:-1]
        
        pvec = llandmark - rlandmark
        pvec = np.pad(pvec, [(0,1),(0,0)], mode='edge')
        px, py = np.transpose(pvec)
        x, y = np.transpose(landmark)
        
        i = np.arange(0, len(pvec))
        interp_i = np.linspace(0, i.max(), 30 * i.size)
        
        vectormap_x = np.zeros((h,w))
        vectormap_y = np.zeros((h,w))

        px = interp1d(i, px)(interp_i)
        py = interp1d(i, py)(interp_i)
        
        xi = np.array(interp1d(i, x)(interp_i), int)
        yi = np.array(interp1d(i, y)(interp_i), int)
        
        xi = np.clip(xi, None, w-1)
        yi = np.clip(yi, None, h-1)
        
        pvec = np.reshape(np.stack([px, py], 1), (-1, 2))
        unitvec = pvec / np.linalg.norm(pvec, axis=1)[:, None]
        px, py = np.transpose(unitvec)
        
        vectormap_x[xi, yi] = px
        vectormap_y[xi, yi] = py
        paf_x.append(vectormap_x)
        paf_y.append(vectormap_y)
        
    paf = np.array([paf_x, paf_y])
    
    paf_x = []
    paf_y = []
    for landmark in connects:
        llandmark = landmark[1:]
        rlandmark = landmark[:-1]
        
        pvec = llandmark - rlandmark
        px, py = np.transpose(pvec)
        x, y = np.transpose(landmark)
        
        i = np.arange(0, len(pvec)+1)
        interp_i = np.linspace(0, i.max(), 30 * i.size)
        
        vectormap_x = np.zeros((h,w))
        vectormap_y = np.zeros((h,w))

        px = interp1d(i+1, np.concatenate((px, px)), fill_value="extrapolate")(interp_i)
        py = interp1d(i+1, np.concatenate((py, py)), fill_value="extrapolate")(interp_i)
        
        xi = np.array(interp1d(i, x)(interp_i), int)
        yi = np.array(interp1d(i, y)(interp_i), int)
        
        xi = np.clip(xi, None, w-1)
        yi = np.clip(yi, None, h-1)

        pvec = np.reshape(np.stack([px, py], 1), (-1, 2))
        unitvec = pvec / np.linalg.norm(pvec, axis=1)[:, None]
        px, py = np.transpose(unitvec)
        
        vectormap_x[xi, yi] = px
        vectormap_y[xi, yi] = py
        paf_x.append(vectormap_x)
        paf_y.append(vectormap_y)
    
    paf_connect = np.array([paf_x, paf_y])
    paf_total = np.concatenate((paf,paf_connect),1) #(2,28,240,240)
    total_x = paf_total[0]
    total_y = paf_total[1]
    total_x = np.transpose(total_x, (1,2,0))
    total_y = np.transpose(total_y, (1,2,0)) #(240,240,28)
    total_x = tf.math.reduce_sum(total_x, axis=-1)
    total_y = tf.math.reduce_sum(total_y, axis=-1) #(240,240)
    
    w, h = total_x.shape
    total_x = tf.reshape(total_x, (w, h, 1))
    total_y = tf.reshape(total_y, (w, h, 1))

    return total_x, total_y


def create_gaussian_kernel(dot_size, num_channels):
    # make a canvas
    dot_size = dot_size * 2

    alpha = 20.0
    x_axis = tf.linspace(alpha, -alpha, dot_size)[:, None]
    y_axis = tf.linspace(alpha, -alpha, dot_size)[None, :]

    template = tf.sqrt(x_axis ** 2 + y_axis ** 2)
    template = tf.reduce_max(template) - template
    template = template / tf.reduce_max(template)

    kernel = tf.reshape([template] * num_channels, (num_channels, dot_size, dot_size, 1))
    kernel = tf.transpose(kernel, [1, 2, 0, 3])
    return kernel


def get_features(fid):
    dt = get_everyone_data(fid)

    if dt == None:
        return None

    img, landmarks, data = dt

    if not 'vcft' in data.keys():
        return None

    if not 'face_landmarks' in data['vcft'].keys():
        return None

    if data['vcft']['confidence'] < 0.95:
        return None

    face, normed_landmarks = process_frame_to_face(img, landmarks)
    face = cv2.resize(face, (240, 240))
    face_landmarks = normed_landmarks * 240

    boundary_map = get_boundary_map(face_landmarks.tolist(), face)

    w, h, c = boundary_map.shape
    input_tensor = np.reshape(boundary_map, (1, w, h, c))
    input_tensor = tf.cast(input_tensor, np.float32)
    gaussian_filtered_map = tf.nn.depthwise_conv2d(input_tensor, kernel_heat, strides=[1, 1, 1, 1], padding='SAME')[0]

    total_x, total_y = get_paf(face_landmarks.tolist(), face)
    #total_x, total_y : (240,240, 1)

    w, h, c = total_x.shape
    input_tensor = np.reshape(total_x, (1, w, h, c))
    input_tensor = tf.cast(input_tensor, np.float32)
    total_x = tf.nn.depthwise_conv2d(input_tensor, kernel_paf, strides=[1, 1, 1, 1], padding='SAME')[0]
    
    input_tensor = np.reshape(total_y, (1, w, h, c))
    input_tensor = tf.cast(input_tensor, np.float32)
    total_y = tf.nn.depthwise_conv2d(input_tensor, kernel_paf, strides=[1, 1, 1, 1], padding='SAME')[0]
    
    gaussian_filtered_paf = tf.stack([total_x, total_y])

    # Headpose Estimation
    # f = img_w
    # c = (img_w / 2, img_h / 2)
    #
    # required_landmarks = get_required_landmarks(landmarks)
    # image_points_2d = get2DImagePoints(required_landmarks)
    # rvec, tvec = getRVec(image_points_2d, f, c)
    #
    # rmat = cv2.Rodrigues(rvec)[0]
    #
    # if tvec[-1] < 0:
    #     rmat[:,1:] = rmat[:,1:] * -1 # note: for some reason, solvePnP fails to find solution on positive z-axis.
    # hp = rotationMatrixToEulerAngles(rmat)

    return face, normed_landmarks, gaussian_filtered_map, gaussian_filtered_paf

def create_a_heatmap(landmark, img_size=96):
    landmark = landmark * [img_size, img_size]
    pos = np.dstack(np.mgrid[0:img_size:1, 0:img_size:1])
    rv = multivariate_normal(mean=[landmark[0], landmark[1]], cov=4)
    hm = rv.pdf(pos)

    # scaling for int
    # hm = np.array(hm * 1000, dtype=np.int)

    hm = cv2.flip(hm, 0)
    hm = cv2.rotate(hm, cv2.ROTATE_90_CLOCKWISE)
    return hm

def encode_boundary_map(boundary_map):
    w, h, c = boundary_map.shape
    encoded_boundaries = []
    for i in range(c):
        temp = boundary_map[:, :, i]
        temp2 = cv2.normalize(temp.numpy(), None, 0, 255, cv2.NORM_MINMAX)
        temp3 = tf.cast(temp2, tf.uint8)
        temp4 = tf.reshape(temp3, (240, 240, 1))
        encoded = tf.io.encode_jpeg(temp4, quality=100).numpy()
        encoded_boundaries.append(encoded)
    return encoded_boundaries

def encode_features(feature):
    face, normed_landmarks, boundary_map, paf = feature

    encoded_face = tf.io.encode_jpeg(face, quality=100).numpy()
    encoded_boundaries = encode_boundary_map(boundary_map)

    feature = {
        'image': _bytes_feature(encoded_face),
        'landmarks': _float_feature(np.reshape(normed_landmarks, (-1)).tolist()),
        'headpose': _float_feature(np.reshape([0, 0, 0], (-1)).tolist()),
        'paf': _float_feature(np.reshape(paf, (-1)).tolist()),
        'jaw': _bytes_feature(encoded_boundaries[0]),
        'right_eyebrow': _bytes_feature(encoded_boundaries[1]),
        'left_eyebrow': _bytes_feature(encoded_boundaries[2]),
        'nose_vert': _bytes_feature(encoded_boundaries[3]),
        'nose_hori': _bytes_feature(encoded_boundaries[4]),
        're_upper': _bytes_feature(encoded_boundaries[5]),
        're_lower': _bytes_feature(encoded_boundaries[6]),
        'le_upper': _bytes_feature(encoded_boundaries[7]),
        'le_lower': _bytes_feature(encoded_boundaries[8]),
        'mouth_upper': _bytes_feature(encoded_boundaries[9]),
        'mouth_lower': _bytes_feature(encoded_boundaries[10])
    }

    return feature

def convert_to_tfrecord(json_path, writer):
    fid = os.path.splitext(json_path)[0]
    feature = get_features(fid)
    if feature == None:
        return None
    feature = encode_features(feature)
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

    return 1


data_path = "/home/banner/data-archive/everyone/train"
all_json_paths = os.listdir(os.path.join(data_path, 'data'))

temp = [x.split('_')[0] for x in all_json_paths]
all_uids = np.unique(temp)
# counts_per_uid = np.stack(counts_per_uid, axis=-1)

np.random.seed(2000)
val_uids = random.choices(all_uids.tolist(), k=20)
train_uids = [x for x in all_uids if x not in val_uids]
print(len(train_uids), len(val_uids))

train_json_paths = [x for x in all_json_paths if x.split('_')[0] in train_uids]
val_json_paths = [x for x in all_json_paths if x.split('_')[0] in val_uids]

print(len(train_json_paths), len(val_json_paths))

max_thread_num = multiprocessing.cpu_count()

kernel_heat = create_gaussian_kernel(5, 11)
kernel_paf = create_gaussian_kernel(3, 1)

write_path = '/mnt/sata3/paf/everyone-paf-train.tfrecords'
counter = 0
with tf.io.TFRecordWriter(write_path) as writer:
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread_num) as executor:
    for json_path in tqdm(train_json_paths):
        # print(f"Train counter: {i}/{len(train_json_paths)}")
        convert_to_tfrecord(json_path=json_path, writer=writer)

write_path = '/mnt/sata3/paf/everyone-paf-val.tfrecords'
counter = 0
with tf.io.TFRecordWriter(write_path) as writer:
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread_num) as executor:
    for json_path in tqdm(val_json_paths):
        # print(f"Val counter: {i}/{len(val_json_paths)}")
        convert_to_tfrecord(json_path=json_path, writer=writer)
        # executor.submit(convert_to_tfrecord, json_path=json_path, writer=writer)
