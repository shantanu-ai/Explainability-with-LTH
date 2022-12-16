import argparse
import os
import sys

import yaml

import utils
from lth_pruning import lth

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))


def run_mnist(main_dir, args):
    print("Run MNIST")
    # run config
    with open(os.path.join(main_dir, args.config)) as config_file:
        config = yaml.safe_load(config_file)
        _img_size = config["img_size"]
        _seed = config["seed"]
        _dataset_name = config["dataset_name"]
        _data_root = config["data_root"]
        _json_root = config["json_root"]
        _model_arch = config["model_arch"]
        _pretrained = config["pretrained"]
        _transfer_learning = config["transfer_learning"]
        _batch_size = config["batch_size"]
        _lr = config["lr"]
        _logs = config["logs"]
        _num_classes = config["num_classes"]
        # _epochs = config["epochs"]
        _device = utils.get_device()

        _prune_type = config["prune_type"]
        _prune_iterations = config["prune_iterations"]
        _prune_percent = config["prune_percent"]
        _start_iter = config["start_iter"]
        _end_iter = config["end_iter"]
        _resample = config["resample"]
        _epsilon = config["epsilon"]
        _initialized_BB_weights = config["initialized_BB_weights"]
        _continue_pruning = config["continue_pruning"]
        _last_model_chk_pt_file = config["last_model_chk_pt_file"]
        _last_model_mask_file = config["last_model_mask_file"]
        lth_train = lth.LTH(
            _model_arch,
            _num_classes,
            _dataset_name,
            _pretrained,
            _transfer_learning,
            _logs,
            _prune_type,
            _device,
            _initialized_BB_weights,
            _continue_pruning,
            _last_model_chk_pt_file,
            _last_model_mask_file
        )
        lth_train.prune_and_train_BB(
            _seed,
            _epsilon,
            _data_root,
            _json_root,
            _dataset_name,
            _lr,
            _img_size,
            _batch_size,
            _resample,
            _prune_percent,
            _prune_type,
            _prune_iterations,
            _start_iter,
            _end_iter
        )


def run_cub(main_dir, args):
    # run config
    with open(os.path.join(main_dir, args.config)) as config_file:
        config = yaml.safe_load(config_file)
        _img_size = config["img_size"]
        _seed = config["seed"]
        _dataset_name = config["dataset_name"]
        _data_root = config["data_root"]
        _json_root = config["json_root"]
        _model_arch = config["model_arch"]
        _pretrained = config["pretrained"]
        _transfer_learning = config["transfer_learning"]
        _batch_size = config["batch_size"]
        _lr = config["lr"]
        _logs = config["logs"]
        _num_classes = config["num_classes"]
        _device = utils.get_device()

        _prune_type = config["prune_type"]
        _prune_iterations = config["prune_iterations"]
        _prune_percent = config["prune_percent"]
        _start_iter = config["start_iter"]
        _end_iter = config["end_iter"]
        _resample = config["resample"]
        _epsilon = config["epsilon"]

        _initialized_BB_weights = config["initialized_BB_weights"]
        _continue_pruning = config["continue_pruning"]
        _last_model_chk_pt_file = config["last_model_chk_pt_file"]
        _last_model_mask_file = config["last_model_mask_file"]

        lth_train = lth.LTH(
            _model_arch,
            _num_classes,
            _dataset_name,
            _pretrained,
            _transfer_learning,
            _logs,
            _prune_type,
            _device,
            _initialized_BB_weights,
            _continue_pruning,
            last_check_pt=_last_model_chk_pt_file,
            last_mask=_last_model_mask_file
        )
        lth_train.prune_and_train_BB(
            _seed,
            _epsilon,
            _data_root,
            _json_root,
            _dataset_name,
            _lr,
            _img_size,
            _batch_size,
            _resample,
            _prune_percent,
            _prune_type,
            _prune_iterations,
            _start_iter,
            _end_iter
        )


if __name__ == '__main__':
    print("Train BB using pruning by LTH")
    # parse arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--config", "-c", default="config/BB_mnist.yaml")
    parser.add_argument(
        "--config", "-c", default="config/BB_cub.yaml")
    parser.add_argument(
        "--main_dir", "-m", default="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/")
    args = parser.parse_args()
    main_dir = args.main_dir
    # run_mnist(main_dir, args)
    run_cub(main_dir, args)
