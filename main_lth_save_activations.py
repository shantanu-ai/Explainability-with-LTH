import argparse
import os
import sys

import yaml

import utils
from lth_pruning import lth_save_activations

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))


def run_mnist():
    print("Save act of BB using pruning by LTH")
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config/BB_mnist.yaml")
    parser.add_argument(
        "--main_dir", "-m", default="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/")
    args = parser.parse_args()
    main_dir = args.main_dir

    # run config
    with open(os.path.join(main_dir, args.config)) as config_file:
        config = yaml.safe_load(config_file)
        _bb_layers = config["bb_layers_for_concepts"]
        _img_size = config["img_size"]
        _seed = config["seed"]
        _dataset_name = config["dataset_name"]
        _data_root = config["data_root"]
        _json_root = config["json_root"]
        _model_arch = config["model_arch"]
        _pretrained = config["pretrained"]
        _transfer_learning = config["transfer_learning"]

        _lr = config["lr"]
        _logs = config["logs"]
        _num_classes = config["num_classes"]
        _epochs = config["epochs"]
        _device = utils.get_device()

        _prune_type = config["prune_type"]
        _prune_iterations = config["prune_iterations"]
        _prune_percent = config["prune_percent"]
        _start_iter = config["start_iter"]
        _end_iter = config["end_iter"]
        _resample = config["resample"]
        _epsilon = config["epsilon"]
        _attribute_file_name = config["attribute_file_name"]
        _batch_size = 1

        lth_save_activations.save_activations_with_Pruning(
            _seed,
            _num_classes,
            _pretrained,
            _transfer_learning,
            _prune_type,
            _dataset_name,
            _data_root,
            _json_root,
            _batch_size,
            _img_size,
            _start_iter,
            _prune_iterations,
            _logs,
            _model_arch,
            _bb_layers,
            _attribute_file_name,
            _device
        )


def run_cubs():
    print("Save act CUBS-200, of BB using pruning by LTH")
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config/BB_cub.yaml")
    parser.add_argument(
        "--main_dir", "-m", default="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/")
    args = parser.parse_args()
    main_dir = args.main_dir

    # run config
    with open(os.path.join(main_dir, args.config)) as config_file:
        config = yaml.safe_load(config_file)
        _bb_layers = config["bb_layers_for_concepts"]
        _img_size = config["img_size"]
        _seed = config["seed"]
        _dataset_name = config["dataset_name"]
        _data_root = config["data_root"]
        _json_root = config["json_root"]
        _model_arch = config["model_arch"]
        _pretrained = config["pretrained"]
        _transfer_learning = config["transfer_learning"]

        _logs = config["logs"]
        _num_classes = config["num_classes"]
        _device = utils.get_device()

        _prune_type = config["prune_type"]
        _prune_iterations = config["prune_iterations"]
        _start_iter = config["start_iter"]
        _attribute_file_name = config["attribute_file_name"]
        _batch_size = 1

        lth_save_activations.save_activations_with_Pruning(
            _seed,
            _num_classes,
            _pretrained,
            _transfer_learning,
            _prune_type,
            _dataset_name,
            _data_root,
            _json_root,
            _batch_size,
            _img_size,
            _start_iter,
            _prune_iterations,
            _logs,
            _model_arch,
            _bb_layers,
            _attribute_file_name,
            _device
        )


if __name__ == '__main__':
    # run_mnist()
    run_cubs()
