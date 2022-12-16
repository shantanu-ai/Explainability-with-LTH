import os
from collections import namedtuple
from itertools import product

import h5py
import numpy as np
import sklearn.metrics as metrics
import torch


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_correct(y_hat, y, num_classes):
    if num_classes == 1:
        y_hat = [1 if y_hat[i] >= 0.5 else 0 for i in range(len(y_hat))]
        correct = [1 if y_hat[i] == y[i] else 0 for i in range(len(y_hat))]
        return np.sum(correct)
    else:
        return y_hat.argmax(dim=1).eq(y).sum().item()


def flatten_cnn_activations_using_max_pooled(activations, kernel_size, stride=1):
    max_pool = torch.nn.MaxPool2d(kernel_size, stride=1)
    torch_activation = torch.from_numpy(activations)
    max_pool_activation = max_pool(torch_activation)
    flatten_activations = max_pool_activation.view(
        max_pool_activation.size()[0], -1
    ).numpy()
    return flatten_activations


def flatten_cnn_activations_using_avg_pooled(activations, kernel_size, stride=1):
    avg_pool = torch.nn.AvgPool2d(kernel_size, stride=1)
    torch_activation = torch.from_numpy(activations)
    avg_pool_activation = avg_pool(torch_activation)
    flatten_activations = avg_pool_activation.view(
        avg_pool_activation.size()[0], -1
    ).numpy()
    return flatten_activations


def flatten_cnn_activations_using_activations(activations):
    return activations.reshape(activations.shape[0], -1)


def cal_accuracy(label, out):
    return metrics.accuracy_score(label, out)


def cal_precision(label, out):
    return metrics.precision_score(label, out)


def cal_recall(label, out):
    return metrics.recall_score(label, out)


def cal_roc_auc(label, out_prob):
    return metrics.roc_auc_score(label, out_prob)


def cal_f1_score(label, out):
    return metrics.f1_score(label, out)


def cal_completeness_score(num_labels, acc_g, acc_bb):
    random_pred_acc = 1 / num_labels
    completeness_score = (acc_g - random_pred_acc) / (acc_bb - random_pred_acc)
    return completeness_score


def create_dir(path_dict):
    try:
        os.makedirs(path_dict["path_name"], exist_ok=True)
        print(f"{path_dict['path_type']} directory is created successfully at:")
        print(path_dict["path_name"])
    except OSError as error:
        print(f"{path_dict['path_type']} directory at {path_dict['path_name']} can not be created")


def get_runs(params):
    """
    Gets the run parameters using cartesian products of the different parameters.
    :param params: different parameters like batch size, learning rates
    :return: iterable run set
    """
    Run = namedtuple("Run", params.keys())
    runs = []
    for v in product(*params.values()):
        runs.append(Run(*v))

    return runs


def get_num_correct(y_hat, y):
    return y_hat.argmax(dim=1).eq(y).sum().item()


def save_activations(activations_path, activation_file, bb_layers, train_activations):
    with h5py.File(os.path.join(activations_path, activation_file), 'w') as f:
        for l in bb_layers:
            f.create_dataset(l, data=train_activations[l])
