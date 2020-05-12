from Dataset.BAFADataset import TFDataset as DataGenerator 
from utils.config import process_config

import tensorflow as tf

import argparse
import numpy as np

def main():
    print(tf.__version__)
    print(tf.executing_eagerly())
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

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--config', type=str, help='Configuration json file', default='config/paf_configuration.json')
    args = parser.parse_args()

    config = process_config(args.config)

    train_data_generator = DataGenerator(config, 'train')

    dataset = iter(train_data_generator())
    x,y = next(dataset)
    """
    print(y[0])
    print(y[1])
    print(y[1].shape)
    """



if __name__=='__main__':
    main()
