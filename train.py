from Trainer.PAFTrainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from Models.PAF import Model
import tensorflow as tf
from Dataset.PAFDataset import TFDataset as DataGenerator

import argparse


def main():
    print (tf.__version__)
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

    parser = argparse.ArgumentParser(description='Train TAM')
    parser.add_argument('--config', type=str, help='Configuration json file. e.g) config/configuration.json', default='config/paf_configuration.json')
    args = parser.parse_args()

    config = process_config(args.config)

    # create the experiments dirs
    create_dirs([config.tensorboard_log_dir, config.checkpoint_dir])

    print('[*] Create the model.')
    model = Model(config)

    print('[*] Create the data generator.')
    train_data_generator = DataGenerator(config, 'train')
    val_data_generator = DataGenerator(config, 'val')

    print('[*] Create the trainer')
    trainer = Trainer(model, train_data_generator, val_data_generator, config)

    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # with strategy.scope():
    print('[*] Start training the model.')
    trainer.train()

if __name__ == '__main__':
    main()
