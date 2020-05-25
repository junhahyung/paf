import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
import os

from scipy.interpolate import interp1d

"""
PAF Dataset (loading from BAFA dataset)
"""

file_shuffle_buffer = 100
frame_shuffle_buffer = 100
batch_shuffle_buffer = 80
interleave_cycle = 20

class TFDataset():
    def __init__(self, config, phase):
        self.feature_description = {
            'image': tf.io.VarLenFeature(tf.string),
            'landmarks': tf.io.FixedLenFeature([68*2], tf.float32),
            'headpose': tf.io.FixedLenFeature([3], tf.float32),
            'paf_x': tf.io.VarLenFeature(tf.string),
            'paf_y': tf.io.VarLenFeature(tf.string)
        }

        self.config = config
        self.phase = phase
        self.rotation = False

        self.build_dataset()

    def __call__(self):
        temp = self.dataset.repeat(self.config.num_epochs)
        return temp

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
            .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .apply(tf.data.experimental.ignore_errors()) \
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

        """
        def get_paf(lms, img):


            def _get_paf_py(img, landmarks):

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

                kernel = create_gaussian_kernel(3,2)

                jaw = landmarks[0:17]
                right_eyebrow = landmarks[17:22]
                left_eyebrow = landmarks[22:27]
                nose_vert = landmarks[27:31]
                nose_hori = landmarks[31:36]
                re_upper = landmarks[36:40]
                re_lower = tf.concat((landmarks[39:42], landmarks[36:37]), axis=0)
                le_upper = landmarks[42:46]
                le_lower = tf.concat((landmarks[45:48], landmarks[42:43]), axis=0)
                mouth_upper = tf.concat((landmarks[48:54], landmarks[54:55]), axis=0)
                mouth_lower = tf.concat((landmarks[54:60], landmarks[48:49]), axis=0)

                boundaries = [jaw,
                              right_eyebrow, left_eyebrow,
                              nose_vert, nose_hori,
                              re_upper, re_lower,
                              le_upper, le_lower,
                              mouth_upper, mouth_lower]
                h,w,c, = img.shape
                paf_x = []
                paf_y = []

                for landmark in boundaries:
                    llandmark = landmark[1:]
                    rlandmark = landmark[:-1]
                    
                    pvec = llandmark - rlandmark
                    pvec = tf.pad(pvec, [(0,1),(0,0)], mode='SYMMETRIC')
                    px, py = tf.transpose(pvec)
                    x, y = tf.transpose(landmark)
                    
                    i = tf.range(0, len(pvec))
                    interp_i = tf.linspace(0., len(pvec)-1, 30 * tf.size(i))
                    
                    vectormap_x = np.zeros((h,w))
                    vectormap_y = np.zeros((h,w))
                    
                    px = tfp.math.interp_regular_1d_grid(interp_i, 0, len(pvec), px)
                    py = tfp.math.interp_regular_1d_grid(interp_i, 0, len(pvec), py)
                    #px = interp1d(i, px)(interp_i)
                    #py = interp1d(i, py)(interp_i)
                    
                    xi = tfp.math.interp_regular_1d_grid(interp_i, 0, len(pvec), x)
                    yi = tfp.math.interp_regular_1d_grid(interp_i, 0, len(pvec), y)
                    #xi = interp1d(i, x)(interp_i), int
                    #yi = interp1d(i, y)(interp_i), int
                    
                    xi = tf.cast(tf.clip_by_value(xi, 0, w-1), tf.int32)
                    yi = tf.cast(tf.clip_by_value(yi, 0, h-1), tf.int32)
                    
                    pvec = tf.stack([px, py], 1)
                    unitvec = pvec / tf.linalg.norm(pvec, axis=1)[:, None]
                    px, py = tf.transpose(unitvec)
                    
                    vectormap_x[xi, yi] = px
                    vectormap_y[xi, yi] = py
                    paf_x.append(vectormap_x)
                    paf_y.append(vectormap_y)
                paf_boundary = tf.stack([paf_x, paf_y]) # (2, 11, 240, 240)
            
                # connections
                eyebrow_jaw_r = tf.concat((landmarks[0:1], landmarks[17:18]), axis=0)
                eyebrow_jaw_l = tf.concat((landmarks[16:17], landmarks[26:27]), axis=0)
                nose_eye_r = tf.concat((landmarks[27:28], landmarks[39:40]), axis=0)
                nose_eye_l = tf.concat((landmarks[27:28], landmarks[42:43]), axis=0)
                jaw_mouth_r = tf.concat((landmarks[0:1], landmarks[48:49]), axis=0)
                jaw_mouth_l = tf.concat((landmarks[16:17], landmarks[54:55]), axis=0)
                horinose_eye_r = tf.concat((landmarks[31:32], landmarks[39:40]), axis=0)
                horinose_eye_l = tf.concat((landmarks[35:36], landmarks[42:43]), axis=0)
                eye_jaw_r = tf.concat((landmarks[0:1], landmarks[36:37]), axis=0)
                eye_jaw_l = tf.concat((landmarks[16:17], landmarks[45:46]), axis=0)
                nose_connect_r = tf.concat((landmarks[30:31], landmarks[31:32]), axis=0)
                nose_connect_l = tf.concat((landmarks[30:31], landmarks[35:36]), axis=0)
                eye_eyebrow_r = tf.concat((landmarks[21:22], landmarks[39:40]), axis=0)
                eye_eyebrow_l = tf.concat((landmarks[22:23], landmarks[42:43]), axis=0)
                nose_mouth_r = tf.concat((landmarks[31:32], landmarks[48:49]), axis=0)
                nose_mouth_l = tf.concat((landmarks[35:36], landmarks[54:55]), axis=0)
                eyebrow_connect = landmarks[21:23]

                # connecting two boundaries
                connects = [eyebrow_jaw_l, eyebrow_jaw_r, 
                            nose_eye_r, nose_eye_l,
                            jaw_mouth_l, jaw_mouth_r,
                           horinose_eye_l, horinose_eye_r,
                           eye_jaw_r, eye_jaw_l,
                           nose_connect_r, nose_connect_l, eyebrow_connect,
                           eye_eyebrow_r, eye_eyebrow_l,
                           nose_mouth_r, nose_mouth_l]
                h,w,c, = img.shape
                paf_x = []
                paf_y = []
                for landmark in connects:
                    llandmark = landmark[1:]
                    rlandmark = landmark[:-1]
                    
                    pvec = llandmark - rlandmark
                    px, py = tf.transpose(pvec)
                    x, y = tf.transpose(landmark)
                    
                    i = tf.range(0, len(pvec)+1)
                    interp_i = tf.linspace(0., len(pvec), 30 * (tf.size(i)-1))
                    
                    vectormap_x = np.zeros((h,w))
                    vectormap_y = np.zeros((h,w))

                    px = tfp.math.interp_regular_1d_grid(interp_i, 0, len(pvec), px)
                    py = tfp.math.interp_regular_1d_grid(interp_i, 0, len(pvec), py)
                    #px = interp1d(i+1, np.concatenate((px, px)), fill_value="extrapolate")(interp_i)
                    #py = interp1d(i+1, np.concatenate((py, py)), fill_value="extrapolate")(interp_i)
                    
                    xi = tfp.math.interp_regular_1d_grid(interp_i, 0, len(pvec), x)
                    yi = tfp.math.interp_regular_1d_grid(interp_i, 0, len(pvec), y)
                    #xi = np.array(interp1d(i, x)(interp_i), int)
                    #yi = np.array(interp1d(i, y)(interp_i), int)
                    
                    xi = tf.cast(tf.clip_by_value(xi, 0, w-1), tf.int32)
                    yi = tf.cast(tf.clip_by_value(yi, 0, w-1), tf.int32)

                    pvec = tf.stack([px, py],1)
                    unitvec = pvec / tf.linalg.norm(pvec, axis=1)[:, None]
                    px, py = tf.transpose(unitvec)
                    
                    vectormap_x[xi, yi] = px
                    vectormap_y[xi, yi] = py
                    paf_x.append(vectormap_x)
                    paf_y.append(vectormap_y)
                
                paf_connect = tf.stack([paf_x, paf_y])

                #paf_connect = tf.zeros((2,17,240,240))

                paf_boundary = tf.transpose(paf_boundary, [2,3,1,0])
                paf_connect = tf.transpose(paf_connect, [2,3,1,0])
                paf = tf.concat([paf_boundary, paf_connect], -2) # (240,240,28,2)
                paf = tf.math.reduce_sum(paf, axis=-2) # (240, 240, 2)
                
                # too slow?
                '''
                w, h, c = paf.shape
                input_tensor = tf.reshape(paf, (1,w,h,c))
                input_tensor = tf.cast(input_tensor, tf.float32)
                paf = tf.nn.depthwise_conv2d(input_tensor, kernel, strides=[1,1,1,1], padding='SAME')
                paf = tf.squeeze(paf)
                '''
                paf = tf.math.divide_no_nan(paf, tf.expand_dims(tf.linalg.norm(paf, axis=-1), -1))
                paf = tf.cast(paf, tf.float32)

                return paf


            paf = tf.py_function(func=_get_paf_py, inp=[img,lms], Tout=tf.float32)

            return paf
            """

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

        record = self._parse_features(record)

        # get image
        img = record['image'].values[0]
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32)

        if self.phase == 'train':
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

        if self.rotation:
            landmarks = landmarks - tf.cast([0.5, 0.5], tf.float32)
            rotated_landmark = tf.matmul(landmarks, R)
            landmarks = rotated_landmark + tf.cast([0.5, 0.5], tf.float32)
            img = tfa.image.rotate(img, theta, 'BILINEAR')

        resized_landmark = landmarks * tf.cast(tf.reshape([size_x, size_y], [2]), tf.float32)
        offseted_landmark = resized_landmark - tf.reshape([offset_x, offset_y], [2])
        normed_landmark = offseted_landmark / 192.0
        landmark = normed_landmark * self.config.img_size

        # image
        resized = tf.image.resize(img, size2d)
        img = tf.image.crop_to_bounding_box(resized,
                                            tf.cast(offset_y, tf.int32),
                                            tf.cast(offset_x, tf.int32),
                                            192, 192)
        img = tf.image.resize(img, (self.config.img_size, self.config.img_size))

        img = tf.image.per_image_standardization(img)
        paf_x = record['paf_x'].values[0]
        paf_y = record['paf_y'].values[0]
        paf_x = tf.image.decode_jpeg(paf_x)
        paf_x = tf.cast(paf_x, tf.float32)
        paf_y = tf.image.decode_jpeg(paf_y)
        paf_y = tf.cast(paf_y, tf.float32)
        paf = tf.stack([paf_x, paf_y], axis=-1)
        paf = tf.linalg.normalize(paf, axis=-1)[0]
        paf = tf.squeeze(paf)
        paf = tf.where(tf.math.is_nan(paf), tf.zeros_like(paf), paf)

        return img, (landmark, paf)
