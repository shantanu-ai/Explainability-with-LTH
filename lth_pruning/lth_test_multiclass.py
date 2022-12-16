import os
import pickle
import random
import time

import numpy as np
import torch
from tqdm import tqdm

import lth_pruning.pruning_utils as pruning_utils
import utils
from model_factory.model_meta import Model_Meta


def test_pruned_models(
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
        batch_size,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    test_loader = None

    start = time.time()

    if dataset_name == "cub":
        test_loader = pruning_utils.get_test_dataloader_cub(
            dataset_name,
            data_root,
            json_root,
            batch_size,
            img_size,
            "attributes.npy"
        )
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")
    bb_checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
    output_stat_path = os.path.join(
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
            "path_name": output_stat_path,
            "path_type": "prune_stat_path-for-each-prune-iteration"
        })

    percent_weight_remaining = pruning_utils.get_percent_weight_remains(dataset_name, model_arch)

    metric_arr = []
    for _ite in range(start_iter, prune_iterations):
        print(f"Prune iteration: {_ite} =======================================>")
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
        start = time.time()
        stats = validate(
            test_loader,
            _ite,
            bb_model,
            dataset_name,
            device,
            percent_weight_remaining[_ite],
            num_classes
        )
        metric = stats["metric"]
        metric_arr.append(metric)

        print(f"Percent weight remaining: {metric['percent_weight_remaining']}")
        print("Accuracy using BB: ")
        print(f"Accuracy: {metric['BB']['Accuracy']}")
        np.save(
            os.path.join(output_stat_path, f"out_put_GT_prune_ite_{_ite}.npy"),
            stats["out_put_GT_np"]
        )
        np.save(
            os.path.join(output_stat_path, f"out_put_predict_bb_prune_ite_{_ite}.npy"),
            stats["out_put_predict_bb_np"]
        )

        done = time.time()
        elapsed = done - start
        print("Time to execute for this iteration: " + str(elapsed) + " secs")

    metric_file = open(os.path.join(
        output_stat_path,
        f"metric_{cav_flattening_type}_test.pkl"
    ), "wb")
    pickle.dump(metric_arr, metric_file)
    metric_file.close()
    print(f"Metric dictionary is saved in the location: {output_stat_path}")


def validate(
        test_loader,
        iter_,
        model,
        dataset_name,
        device,
        percent_weight_remaining,
        num_classes
):
    out_prob_arr_bb = []
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict_bb = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_idx, data_tuple in enumerate(test_loader):
                data, target = pruning_utils.get_image_target(data_tuple, dataset_name, device)
                y_hat_bb = model(data)

                out_prob_arr_bb.append(y_hat_bb)
                out_put_predict_bb = torch.cat((out_put_predict_bb, y_hat_bb), dim=0)
                out_put_GT = torch.cat((out_put_GT, target), dim=0)

                t.set_postfix(iteration=f"{iter_}")
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_bb_np = out_put_predict_bb.cpu().numpy()
    y_hat_bb = out_put_predict_bb.cpu().argmax(dim=1)
    acc_bb = utils.cal_accuracy(out_put_GT_np, y_hat_bb)
    metric = {
        "BB": {
            "Accuracy": acc_bb,
        },
        "percent_weight_remaining": percent_weight_remaining
    }

    return {
        "out_put_GT_np": out_put_GT_np,
        "out_put_predict_bb_np": out_put_predict_bb_np,
        "metric": metric
    }
