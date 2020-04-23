import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler
from preprocess_dataset.TFRHelper import _bytes_feature, _float_feature
from tqdm import tqdm, trange
from utils.losses import *
import concurrent.futures
import multiprocessing

class Trainer():
    def __init__(self, model, train_data, val_data, unlabeled_data, config):
        self.data = train_data
        self.val_data = val_data
        self.unlabeled_data = unlabeled_data
        self.iterations = 0

        self.model = model
        self.config = config
        self.callbacks = list()

        self.first_iteration = True

    def init_callbacks(self):
        self.callbacks = list()
        monitor = 'val_loss'
        modelcheckpoint = ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, str(self.iterations) + '-{epoch:02d}-{val_loss:04f}.hdf5'),
            monitor=monitor,
            mode='min',
            save_best_only=True,
            verbose=1,
        )
        tensorboard = TensorBoard(
            log_dir=self.config.tensorboard_log_dir,
            write_graph=True,
        )
        reduceLR = ReduceLROnPlateau(
            monitor=monitor,
            factor=0.9,
            patience=3,
            verbose=1,
        )
        earlyStopping = EarlyStopping(
            monitor=monitor,
            patience=15,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        csvLogger = CSVLogger(
            filename=os.path.join(self.config.checkpoint_dir, f'training-{self.iterations}.log'),
            append=True
        )

        def schedule(epoch):
            return self.config.learning_rate * (0.98 ** epoch)

        scheduler = LearningRateScheduler(
            schedule=schedule,
            verbose=1
        )

        self.callbacks.append(modelcheckpoint)
        self.callbacks.append(tensorboard)
        self.callbacks.append(reduceLR)
        self.callbacks.append(earlyStopping)
        self.callbacks.append(csvLogger)
        # self.callbacks.append(scheduler)

    def train(self):
        while (self.iterations < 10):
            self.init_callbacks()

            self.data.build_dataset()
            self.val_data.build_dataset()
            self.model.build_model()

            model = self.model.model

            if self.first_iteration:
                steps = self.config.training_steps
            else :
                steps = 30000

            model.fit(
                x=self.data(),
                steps_per_epoch=steps,
                epochs=self.config.num_epochs,
                callbacks=self.callbacks,
                validation_data=self.val_data(),
                validation_steps=self.config.validation_steps,
                verbose=1
            )

            self.first_iteration = False

            self.iterations = self.iterations + 1

            self.unlabeled_data.build_dataset()
            iterator = self.unlabeled_data().as_numpy_iterator()
            write_path = os.path.join(self.config.landmark_path, 'train', 'unlabeled.tfrecords')

            with tf.io.TFRecordWriter(write_path) as writer:
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    num_steps = 996310 // self.config.max_num_threads #996310
                    for i in trange(num_steps):
                        original, cropped = iterator.next()
                        results = model.predict(cropped)
                        landmarks, headposes = results

                        for n in range(self.config.max_num_threads):
                            executor.submit(self.write_to_tfrecords, landmarks=landmarks[n], headpose=headposes[n],
                                            original=original[n], writer=writer)

    def write_to_tfrecords(self, landmarks, headpose, original, writer):
        landmarks = landmarks.flatten() + 24 # git because all images has [24, 24] padding
        normalized_landmarks = tf.math.divide(landmarks, 240.0)

        feature = {
            'source': _bytes_feature(bytes('unlabeled', encoding='ascii')),
            'image': _bytes_feature(original),
            'landmarks': _float_feature(normalized_landmarks.tolist()),
            'headpose': _float_feature(headpose.tolist())
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())