import argparse
import os
import sys

import yaml

import concept_activations.cav_generation as cav_for_mnist

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))

if __name__ == '__main__':
    print("CAV generation")
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
        _dataset_name = config["dataset_name"]
        _concept_names = config["concept_names"]
        _model_arch = config["model_arch"]
        _logs = config["logs"]
        _cav_flattening_type = config["cav_flattening_type"]
        cav_for_mnist.generate_cavs(
            _logs,
            _model_arch,
            _dataset_name,
            _concept_names,
            _cav_flattening_type
        )
