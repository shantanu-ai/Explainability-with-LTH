import os
import pickle
import random
import time

import numpy as np
import torch

import concept_activations.TCAV as TCAV
import concept_activations.concept_activations_utils as ca_utils
import lth_pruning.pruning_utils as pruning_utils
import utils
from model_factory.model_meta import Model_Meta


def cal_TCAV_w_pruning(
        seed,
        data_root,
        json_root,
        model_arch,
        num_classes,
        pretrained,
        transfer_learning,
        logs,
        cav_flattening_type,
        dataset_name,
        img_size,
        start_iter,
        prune_iterations,
        prune_type,
        bb_layer,
        batch_size,
        concept_names,
        class_list,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform_params = {
        "img_size": img_size
    }
    start = time.time()
    test_loader = pruning_utils.get_test_dataloader(
        dataset_name,
        data_root,
        json_root,
        batch_size,
        transform_params
    )
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")
    bb_checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
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

    activations_path = os.path.join(
        logs,
        "activations",
        "Pruning",
        model_arch,
        dataset_name,
        "BB_act",
        f"Prune_type_{prune_type}"
    )

    prune_stat_path = os.path.join(
        logs,
        "predictions",
        "prune-statistics",
        model_arch,
        dataset_name,
        f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}"
        "TCAV-scores"
    )
    utils.create_dir(
        path_dict={
            "path_name": prune_stat_path,
            "path_type": "prune_stat_path-for-each-prune-iteration"
        })

    for _ite in range(start_iter, prune_iterations):
        print(f"Prune iteration: {_ite} =======================================>")
        test_activations_file = f"test_activations_prune_iteration_{_ite}.h5"
        concept_vectors = ca_utils.get_concept_vectors_for_pruning(
            _ite,
            cav_path,
            bb_layer,
            cav_flattening_type
        )

        bb_model = pruning_utils.load_BB_model_w_pruning(
            model_arch,
            num_classes,
            pretrained,
            transfer_learning,
            dataset_name,
            device,
            _ite,
            bb_checkpoint_path
        )
        bb_model.eval()
        if type(bb_layer) == str:
            bb_model_meta = Model_Meta(bb_model, [bb_layer])
        else:
            bb_model_meta = Model_Meta(bb_model, bb_layer)

        start = time.time()

        if len(class_list) == 2:
            stat_dict = TCAV.calculate_cavs_binary_classification(
                test_loader,
                concept_vectors,
                bb_model,
                bb_model_meta,
                cav_flattening_type,
                bb_layer,
                concept_names,
                class_list,
                model_arch
            )
            done = time.time()
            elapsed = done - start
            print("Time to complete this iteration: " + str(elapsed) + " secs")
            metric_file_per_iter = open(os.path.join(
                prune_stat_path,
                f"TCAV_scores_file_pruning_iter_{_ite}.pkl"
            ), "wb")
            pickle.dump(stat_dict, metric_file_per_iter)
            metric_file_per_iter.close()


def cal_TCAV_w_pruning_multiclass(
        seed,
        data_root,
        json_root,
        model_arch,
        num_classes,
        pretrained,
        transfer_learning,
        logs,
        cav_flattening_type,
        dataset_name,
        img_size,
        start_iter,
        prune_iterations,
        prune_type,
        bb_layer,
        batch_size,
        concept_names,
        concepts_for_tcav,
        class_labels,
        labels_for_tcav,
        attribute_file_name,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start = time.time()
    test_loader = None
    if dataset_name == "cub":
        test_loader = pruning_utils.get_test_dataloader_cub(
            dataset_name,
            data_root,
            json_root,
            batch_size,
            img_size,
            attribute_file_name
        )
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")
    labels_index_list_TCAV = pruning_utils.get_class_index_for_TCAV(labels_for_tcav, class_labels)
    concept_index_list_TCAV = pruning_utils.get_class_index_for_TCAV(concepts_for_tcav, concept_names)

    bb_checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
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

    activations_path = os.path.join(
        logs,
        "activations",
        "Pruning",
        model_arch,
        dataset_name,
        "BB_act",
        f"Prune_type_{prune_type}"
    )

    prune_stat_path = os.path.join(
        logs,
        "predictions",
        "prune-statistics",
        model_arch,
        dataset_name,
        f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}",
        labels_for_tcav[0]
    )
    utils.create_dir(
        path_dict={
            "path_name": prune_stat_path,
            "path_type": "prune_stat_path-for-each-prune-iteration"
        })

    for _ite in range(start_iter, prune_iterations):
        print(f"Prune iteration: {_ite} =======================================>")

        concept_vectors = ca_utils.get_concept_vectors_for_pruning(
            _ite,
            cav_path,
            bb_layer,
            cav_flattening_type
        )

        bb_model = pruning_utils.load_BB_model_w_pruning(
            model_arch,
            num_classes,
            pretrained,
            transfer_learning,
            dataset_name,
            device,
            _ite,
            bb_checkpoint_path
        )
        bb_model.eval()
        if type(bb_layer) == str:
            bb_model_meta = Model_Meta(bb_model, [bb_layer])
        else:
            bb_model_meta = Model_Meta(bb_model, bb_layer)

        start = time.time()

        stat_dict = TCAV.calculate_cavs_multiclass(
            test_loader,
            concept_vectors,
            bb_model,
            bb_model_meta,
            cav_flattening_type,
            bb_layer,
            model_arch,
            class_labels,
            labels_index_list_TCAV,
            concept_names,
            concept_index_list_TCAV
        )
        done = time.time()
        elapsed = done - start
        print("Time to complete this iteration: " + str(elapsed) + " secs")
        print(f"TCAV Stats for this iteration: {_ite} ")
        print(stat_dict)
        metric_file_per_iter = open(os.path.join(
            prune_stat_path,
            f"TCAV_scores_file_pruning_iter_{_ite}.pkl"
        ), "wb")
        pickle.dump(stat_dict, metric_file_per_iter)
        metric_file_per_iter.close()
