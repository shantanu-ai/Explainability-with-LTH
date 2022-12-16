import os
import random
import time

import numpy as np
import torch

import concept_activations.concept_activations_utils as ca_utils
import concept_activations.concept_completeness_train as cav_train
import lth_pruning.pruning_utils as prun_utils
import utils
from model_factory.model_meta import Model_Meta


def train_G_completeness_w_pruning(
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
        num_labels,
        g_lr,
        batch_size,
        epochs,
        hidden_features,
        th,
        val_after_th,
        attribute_file_name,
        device
):
    train_loader = None
    val_loader = None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start = time.time()
    if dataset_name == "mnist":
        train_loader, val_loader = ca_utils.get_dataloader_mnist(
            data_root,
            json_root,
            dataset_name,
            img_size,
            batch_size
        )
    elif dataset_name == "cub":
        train_loader, val_loader = ca_utils.get_dataloader_cub(
            data_root,
            json_root,
            dataset_name,
            img_size,
            batch_size,
            attribute_file_name=attribute_file_name
        )
    done = time.time()
    elapsed = done - start
    print("Time to load the desired dataset from disk: " + str(elapsed) + " secs")

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

    g_model_checkpoint_path = os.path.join(
        logs,
        "chk_pt",
        "Pruning",
        model_arch,
        bb_layer,
        dataset_name,
        "G",
        f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}"
    )

    tb_path = os.path.join(
        logs,
        "tensorboard_logs",
        "G",
        "Pruning",
        f"Prune_type_{prune_type}",
        model_arch,
        dataset_name,
        f"cav_flattening_type_{cav_flattening_type}"
    )

    utils.create_dir(
        path_dict={
            "path_name": g_model_checkpoint_path,
            "path_type": "checkpoint-for-G"
        })
    utils.create_dir(
        path_dict={
            "path_name": tb_path,
            "path_type": "tensorboard-for-G"
        })

    for _ite in range(start_iter, prune_iterations):
        print(f"Prune iteration: {_ite} =======================================>")
        concept_vectors = ca_utils.get_concept_vectors_for_pruning(
            _ite,
            cav_path,
            bb_layer,
            cav_flattening_type
        )

        torch_concept_vector = torch.from_numpy(concept_vectors).to(device, dtype=torch.float32)
        bb_model = prun_utils.load_BB_model_w_pruning(
            model_arch,
            num_classes,
            pretrained,
            transfer_learning,
            dataset_name,
            device,
            _ite,
            bb_checkpoint_path
        )
        for param in bb_model.parameters():
            param.requires_grad = False

        if type(bb_layer) == str:
            bb_model_meta = Model_Meta(bb_model, [bb_layer])
        else:
            bb_model_meta = Model_Meta(bb_model, bb_layer)

        start = time.time()

        cav_train.train_concept_to_activation_model(
            _ite,
            train_loader,
            val_loader,
            torch_concept_vector,
            bb_model,
            bb_model_meta,
            model_arch,
            cav_flattening_type,
            dataset_name,
            bb_layer,
            g_lr,
            g_model_checkpoint_path,
            tb_path,
            epochs,
            hidden_features,
            th,
            val_after_th,
            num_classes,
            num_labels,
            device
        )

        done = time.time()
        elapsed = done - start
        print("Time to train for the iteration: " + str(elapsed) + " secs")
