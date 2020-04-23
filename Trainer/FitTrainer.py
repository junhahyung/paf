import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler

class Trainer():
    def __init__(self, model, train_data, val_data, config):
        self.data = train_data()
        self.val_data = val_data()

        self.model = model
        self.config = config
        self.callbacks = list()
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks = list()
        monitor = 'val_loss'
        modelcheckpoint = ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '{epoch:02d}-{val_loss:04f}.hdf5'),
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
            patience=6,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        csvLogger = CSVLogger(
            filename=os.path.join(self.config.tensorboard_log_dir, 'training.log'),
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
        # self.callbacks.append(earlyStopping)
        self.callbacks.append(csvLogger)
        # self.callbacks.append(scheduler)
    def train(self):
        # todo: edit to use GradientTape functionality
        # todo: implement validation dataset
        self.model.model.fit(
            x=self.data,
            epochs=self.config.num_epochs,
            steps_per_epoch=self.config.training_steps,
            callbacks=self.callbacks,
            validation_data=self.val_data,
            validation_steps=self.config.validation_steps,
            verbose=1
        )