import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

import concept_activations.concept_activations_utils as ca_utils
from concept_activations.g import G

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))
import utils
import yaml
import torch


def calculate_concept_completeness_score(
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
        percent_weight_remaining,
        dataset_name,
        device
):
    g_model_ip_size, g_model_op_size = ca_utils.get_g_model_ip_op_size(
        test_loader,
        device,
        bb_model,
        bb_model_meta,
        torch_concept_vector,
        bb_layer,
        cav_flattening_type,
        dataset_name
    )

    g = G(g_model_ip_size, g_model_op_size, hidden_features).to(device)
    g.load_state_dict(torch.load(g_model_checkpoint))
    bb_model_mid, bb_model_tail = ca_utils.dissect_bb_model(model_arch, bb_model)

    out_prob_arr_bb = []
    out_prob_arr_g = []
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict_bb = torch.FloatTensor().cuda()
    out_put_predict_g = torch.FloatTensor().cuda()

    g.eval()
    bb_model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, (images, labels) in enumerate(test_loader):
                bs = images.size(0)
                images = images.to(device)
                labels = labels.to(torch.float32)
                labels = labels.to(device)
                labels = labels.reshape((labels.shape[0], 1))
                y_hat_bb = bb_model(images)

                activations = bb_model_meta.model_activations_store[bb_layer]
                norm_vc = ca_utils.get_normalized_vc(
                    activations,
                    torch_concept_vector,
                    th,
                    val_after_th,
                    cav_flattening_type
                )
                concept_to_act = g(norm_vc)
                y_hat_g = ca_utils.get_concept_to_pred(
                    concept_to_act,
                    bs,
                    activations,
                    bb_model_mid,
                    bb_model_tail
                )

                out_prob_arr_bb.append(y_hat_bb)
                out_prob_arr_g.append(y_hat_g)

                out_put_predict_bb = torch.cat((out_put_predict_bb, y_hat_bb), dim=0)
                out_put_predict_g = torch.cat((out_put_predict_g, y_hat_g), dim=0)
                out_put_GT = torch.cat((out_put_GT, labels), dim=0)

                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    test_accuracy = utils.get_correct(out_put_predict_bb, out_put_GT, 1) / out_put_GT.size(0)
    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_bb_np = out_put_predict_bb.cpu().numpy()
    out_put_predict_g_np = out_put_predict_g.cpu().numpy()
    y_hat_bb = np.where(out_put_predict_bb_np > 0.5, 1, 0)
    y_hat_g = np.where(out_put_predict_g_np > 0.5, 1, 0)
    acc_bb = utils.cal_accuracy(out_put_GT_np, y_hat_bb)
    acc_g = utils.cal_accuracy(out_put_GT_np, y_hat_g)
    metric = {
        "BB": {
            "Accuracy": acc_bb,
            "Precision": utils.cal_precision(out_put_GT_np, y_hat_bb),
            "Recall": utils.cal_recall(out_put_GT_np, y_hat_bb),
            "RocAUC": utils.cal_roc_auc(out_put_GT_np, y_hat_bb),
            "F1_score": utils.cal_f1_score(out_put_GT_np, y_hat_bb)
        },
        "G": {
            "Accuracy": acc_g,
            "Precision": utils.cal_precision(out_put_GT_np, y_hat_g),
            "Recall": utils.cal_recall(out_put_GT_np, y_hat_g),
            "RocAUC": utils.cal_roc_auc(out_put_GT_np, y_hat_g),
            "F1_score": utils.cal_f1_score(out_put_GT_np, y_hat_g)
        },
        "Completeness_score": utils.cal_completeness_score(num_labels, acc_g, acc_bb),
        "percent_weight_remaining": percent_weight_remaining
    }

    return {
        "out_put_GT_np": out_put_GT_np,
        "out_put_predict_bb_np": out_put_predict_bb_np,
        "out_put_predict_g": out_put_predict_g_np,
        "metric": metric
    }
