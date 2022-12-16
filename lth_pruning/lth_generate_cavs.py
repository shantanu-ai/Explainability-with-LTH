import os
import pickle
import random

import numpy as np
import torch

import concept_activations.cav_generation as cavs
import utils


def generate_cavs_with_Pruning(
        seed,
        prune_type,
        dataset_name,
        start_iter,
        prune_iterations,
        logs,
        model_arch,
        bb_layers,
        concept_names,
        cav_flattening_type
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    activations_path = os.path.join(
        logs,
        "activations",
        "Pruning",
        model_arch,
        dataset_name,
        "BB_act",
        f"Prune_type_{prune_type}"
    )

    cav_path = os.path.join(
        logs,
        "activations",
        "Pruning",
        model_arch,
        dataset_name,
        "cavs",
        f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}"
    )

    utils.create_dir(
        path_dict={
            "path_name": cav_path,
            "path_type": "cavs-of-BB"
        })

    for _ite in range(start_iter, prune_iterations):
        print(f"Prune iteration: {_ite} =======================================>")
        train_activations_file = f"train_activations_prune_iteration_{_ite}.h5"
        val_activation_file = f"val_activations_prune_iteration_{_ite}.h5"
        test_activation_file = f"test_activations_prune_iteration_{_ite}.h5"

        train_GT_file = f"train_np_attr_GT_prune_iteration_{_ite}.npy"
        val_GT_file = f"val_np_attr_GT_prune_iteration_{_ite}.npy"
        test_GT_file = f"test_np_attr_GT_prune_iteration_{_ite}.npy"

        if dataset_name == "mnist":
            train_cavs, train_cav_cls_report = cavs.generate_cavs_using_pruning(
                concept_names,
                cav_flattening_type,
                bb_layers,
                activations_path,
                train_activations_file,
                val_activation_file,
                # test_activation_file,
                train_GT_file,
                val_GT_file,
                # test_GT_file
            )
        elif dataset_name == "cub":
            train_cavs, train_cav_cls_report = cavs.generate_cavs_using_pruning(
                concept_names,
                cav_flattening_type,
                bb_layers,
                activations_path,
                train_activations_file,
                val_activation_file,
                # test_activation_file,
                train_GT_file,
                val_GT_file,
                # test_GT_file,
                multi_label=True
            )

        if cav_flattening_type == "max_pooled":
            cav_file_name = f"max_pooled_train_cavs_prune_iteration_{_ite}.pkl"
            cls_report_file = f"max_pooled_train_cls_report_prune_iteration_{_ite}.pkl"
        elif cav_flattening_type == f"flattened":
            cav_file_name = f"flattened_train_cavs_prune_iteration_{_ite}.pkl"
            cls_report_file = f"flattened_train_cls_report_prune_iteration_{_ite}.pkl"
        else:
            cav_file_name = f"avg_pooled_train_cavs_prune_iteration_{_ite}.pkl"
            cls_report_file = f"avg_pooled_train_cls_report_prune_iteration_{_ite}.pkl"

        cav_file = open(os.path.join(cav_path, cav_file_name), "wb")
        pickle.dump(train_cavs, cav_file)
        cav_file.close()

        concept_classifier_report_file = open(
            os.path.join(cav_path, cls_report_file),
            "wb"
        )
        pickle.dump(train_cav_cls_report, concept_classifier_report_file)
        print(f"Activation dictionary is saved in the location: {cav_path}")
