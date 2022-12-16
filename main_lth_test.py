import argparse
import os
import sys

import lth_pruning.lth_test_multiclass as lth

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))
import utils
import yaml


def run_CUB():
    print("Test pruned models for CUB")
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
    device = utils.get_device()
    print(f"Device: {device}")
    _img_size = config["img_size"]
    _seed = config["seed"]
    _data_root = config["data_root"]

    _root = config["data_root"]
    _json_root = config["json_root"]
    _model_arch = config["model_arch"]
    _dataset_name = config["dataset_name"]
    _pretrained = config["pretrained"]
    _transfer_learning = config["transfer_learning"]
    _num_classes = config["num_classes"]
    _logs = config["logs"]
    _bb_layers = config["bb_layers_for_concepts"]
    _size = config["img_size"]
    _batch_size = 1
    _num_workers = 1

    _labels_for_tcav = config["labels_for_tcav"]
    _class_labels = config["labels"]
    _concepts_for_tcav = config["concepts_for_tcav"]
    _concept_names = config["concept_names"]


    _cav_flattening_type = config["cav_flattening_type"]

    _prune_iterations = config["prune_iterations"]
    _prune_percent = config["prune_percent"]
    _start_iter = config["start_iter"]
    _end_iter = config["end_iter"]

    _bb_layer = config["bb_layers_for_concepts"][0]
    _prune_type = config["prune_type"]
    _attribute_file_name = config["attribute_file_name"]
    # print("cav vector sign (-ve) || dot product sign (-ve)")
    # print("-np.array(concept_model.coef_)")
    # print("np.dot(flatten_gradients, cav) < 0")
    lth.test_pruned_models(
        _seed,
        _data_root,
        _json_root,
        _model_arch,
        _num_classes,
        _pretrained,
        _transfer_learning,
        _logs,
        _cav_flattening_type,
        _dataset_name,
        _img_size,
        _start_iter,
        _prune_iterations,
        _prune_type,
        _batch_size,
        device
    )


if __name__ == '__main__':
    # run_mnist()
    run_CUB()
