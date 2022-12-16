import os
import pickle
import random
import time

import numpy as np
import torch

import concept_activations.concept_activations_utils as ca_utils
import concept_activations.concept_completeness_mnist_test as cav_test
import lth_pruning.pruning_utils as pruning_utils
import utils
from model_factory.model_meta import Model_Meta


def test_pruned_model_with_completeness_score(
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
        batch_size,
        hidden_features,
        th,
        val_after_th,
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

    prune_stat_path = os.path.join(
        logs,
        "predictions",
        "prune-statistics",
        model_arch,
        dataset_name,
        f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}"
    )
    utils.create_dir(
        path_dict={
            "path_name": prune_stat_path,
            "path_type": "prune_stat_path-for-each-prune-iteration"
        })

    percent_weight_remaining = [
        100.0, 90.0, 81.0, 72.9, 65.6, 59.1, 53.2, 47.8, 43.1, 38.8, 34.9, 31.4,
        28.3, 25.4, 22.9, 20.6, 18.6, 16.7, 15.0, 13.5, 12.2, 11.0, 9.9, 8.9,
        8.0, 7.2, 6.5, 5.9, 5.3, 4.7, 4.3, 3.9, 3.5, 3.1, 2.8
    ]

    metric_arr = []
    for _ite in range(start_iter, prune_iterations):
        print(f"Prune iteration: {_ite} =======================================>")
        concept_vectors = ca_utils.get_concept_vectors_for_pruning(
            _ite,
            cav_path,
            bb_layer,
            cav_flattening_type
        )

        torch_concept_vector = torch.from_numpy(concept_vectors).to(device, dtype=torch.float32)
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
        g_model_checkpoint = os.path.join(g_model_checkpoint_path, f"best_prune_iteration_{_ite}.pth.tar")
        print("Checkpoint for G loaded from here:\n")
        print(g_model_checkpoint)
        stats = cav_test.calculate_concept_completeness_score(
            test_loader,
            torch_concept_vector,
            bb_model,
            bb_model_meta,
            model_arch,
            cav_flattening_type,
            bb_layer,
            g_model_checkpoint,
            hidden_features,
            th,
            val_after_th,
            num_labels,
            percent_weight_remaining[_ite],
            dataset_name,
            device
        )

        metric = stats["metric"]
        metric_arr.append(metric)

        print(f"Percent weight remaining: {metric['percent_weight_remaining']}")
        print("Accuracy using BB: ")
        print(f"Accuracy: {metric['BB']['Accuracy']}")
        print(f"Precision: {metric['BB']['Precision']}")
        print(f"Recall: {metric['BB']['Recall']}")
        print(f"RocAUC: {metric['BB']['RocAUC']}")
        print(f"F1 score: {metric['BB']['F1_score']}")

        print("Accuracy using G: ")
        print(f"Accuracy: {metric['G']['Accuracy']}")
        print(f"Precision: {metric['G']['Precision']}")
        print(f"Recall: {metric['G']['Recall']}")
        print(f"RocAUC: {metric['G']['RocAUC']}")
        print(f"F1 score: {metric['G']['F1_score']}")

        print(f"Completeness score for dataset [{dataset_name}] using [{model_arch}]: "
              f"{metric['Completeness_score']}")

        np.save(
            os.path.join(prune_stat_path, f"out_put_GT_prune_ite_{_ite}.npy"),
            stats["out_put_GT_np"]
        )
        np.save(
            os.path.join(prune_stat_path, f"out_put_predict_bb_prune_ite_{_ite}.npy"),
            stats["out_put_predict_bb_np"]
        )
        np.save(
            os.path.join(prune_stat_path, f"out_put_predict_g_prune_ite_{_ite}.npy"),
            stats["out_put_predict_g"]
        )

        done = time.time()
        elapsed = done - start
        print("Time to execute for this iteration: " + str(elapsed) + " secs")

    metric_file = open(os.path.join(
        prune_stat_path,
        f"metric_{cav_flattening_type}_completeness.pkl"
    ), "wb")
    pickle.dump(metric_arr, metric_file)
    metric_file.close()
    print(f"Activation dictionary is saved in the location: {prune_stat_path}")
