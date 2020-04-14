import os
import json
import time

from bunch import Bunch


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
        config = Bunch(config_dict)
        return config

def process_config(json_file):
    config = get_config_from_json(json_file)
    config.tensorboard_log_dir = os.path.join("experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    config.figures_dir = os.path.join("figures", config.exp_name)
    return config
