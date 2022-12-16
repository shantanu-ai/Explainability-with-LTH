import os
import time

import h5py
import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from concept_activations.flatten_LR import Flatten_LR
from utils import flatten_cnn_activations_using_max_pooled, flatten_cnn_activations_using_activations, \
    flatten_cnn_activations_using_avg_pooled, get_device


def load_activations(path):
    activations = {}
    with h5py.File(path, 'r') as f:
        for k, v in f.items():
            activations[k] = np.array(v)
    return activations


def flatten_activations(
        bb_layers,
        train_activations,
        val_activations,
        # test_activations,
        cav_flattening_type
):
    flatten_train_activations = {}
    flatten_val_activations = {}
    # flatten_test_activations = {}

    kernel_size = {}

    for l in bb_layers:
        kernel_size[l] = train_activations[l].shape[-1]
        flatten_train_activations[l] = []
        flatten_val_activations[l] = []
        # flatten_test_activations[l] = []

    for layer in bb_layers:
        print(f"layer: {layer}")
        if cav_flattening_type == "max_pooled":
            flatten_train_activations[layer] = flatten_cnn_activations_using_max_pooled(
                train_activations[layer],
                kernel_size=kernel_size[layer],
                stride=1
            )
            flatten_val_activations[layer] = flatten_cnn_activations_using_max_pooled(
                val_activations[layer],
                kernel_size=kernel_size[layer],
                stride=1
            )
            # flatten_test_activations[layer] = flatten_cnn_activations_using_max_pooled(
            #     test_activations[layer],
            #     kernel_size=kernel_size[layer],
            #     stride=1
            # )
        elif cav_flattening_type == "flattened":
            flatten_train_activations[layer] = flatten_cnn_activations_using_activations(
                train_activations[layer]
            )
            flatten_val_activations[layer] = flatten_cnn_activations_using_activations(
                val_activations[layer]
            )
            # flatten_test_activations[layer] = flatten_cnn_activations_using_activations(
            #     test_activations[layer]
            # )
        elif cav_flattening_type == "avg_pooled":
            flatten_train_activations[layer] = flatten_cnn_activations_using_avg_pooled(
                train_activations[layer],
                kernel_size=kernel_size[layer],
                stride=1
            )
            flatten_val_activations[layer] = flatten_cnn_activations_using_avg_pooled(
                val_activations[layer],
                kernel_size=kernel_size[layer],
                stride=1
            )
            # flatten_test_activations[layer] = flatten_cnn_activations_using_avg_pooled(
            #     test_activations[layer],
            #     kernel_size=kernel_size[layer],
            #     stride=1
            # )

        print(f"Train: {flatten_train_activations[layer].shape}")
        print(f"Val: {flatten_val_activations[layer].shape}")
        # print(f"Test: {flatten_test_activations[layer].shape}")
        print("-----")

    return flatten_train_activations, flatten_val_activations


def generate_CAV_vectors(
        x_train,
        y_train,
        x_val,
        y_val,
        concept_names,
        multi_label,
        cav_flattening_type
):
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    print(f"Train set size, x_train: {x_train.shape} y_train: {y_train.shape}")
    print(f"Val set size, x_val: {x_val.shape} y_val: {y_val.shape}")

    if multi_label and (cav_flattening_type == "max_pooled" or cav_flattening_type == "avg_pooled"):
        # For datasets like cub and corresponding flattening type
        return fit_logistic_regression_multilabel(x_train, y_train, x_val, y_val, concept_names)
    elif multi_label and cav_flattening_type == "flattened":
        # For datasets like cub nd corresponding flattening type
        return fit_logistic_regression_using_NN(x_train, y_train, x_val, y_val, concept_names)
    else:
        # For datasets like mnist
        return fit_logistic_regression_multiclass(x_train, y_train, x_val, y_val, concept_names)


def fit_logistic_regression_using_NN(x_train, y_train, x_val, y_val, concept_names):
    device = get_device()
    model = Flatten_LR(ip_size=x_train.shape[1], op_size=len(concept_names)).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)),
        batch_size=128,
        num_workers=4,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val)),
        batch_size=10,
        shuffle=False
    )

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    model.train()
    print("#######################################")
    for epoch_id in range(0, 12):
        running_loss = 0
        samples = 0
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (x, labels) in enumerate(train_loader):
                x = x.to(device, dtype=torch.float)
                labels = labels.to(device)
                pred_labels = model(x)
                loss = criterion(pred_labels, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print("batch id: {0}, loss: {1}".format(batch_id, loss))
                t.set_postfix(epoch='{0}'.format(epoch_id), training_loss='{:05.3f}'.format(running_loss))
                t.update()

    model.eval()
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as t:
            for batch_id, (x, labels) in enumerate(val_loader):
                x = x.to(device, dtype=torch.float)
                labels = labels.to(device)
                out_prob = model(x)
                out_put_predict = torch.cat((out_put_predict, out_prob), dim=0)
                out_put_GT = torch.cat((out_put_GT, labels), dim=0)
                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_np = out_put_predict.cpu().numpy()
    y_pred = np.where(out_put_predict_np > 0.5, 1, 0)

    cls_report = {}
    for i, concept_name in enumerate(concept_names):
        cls_report[concept_name] = {}
    for i, concept_name in enumerate(concept_names):
        cls_report[concept_name]["accuracy"] = metrics.accuracy_score(y_pred=y_pred[i], y_true=y_val[i])
        cls_report[concept_name]["precision"] = metrics.precision_score(y_pred=y_pred[i], y_true=y_val[i])
        cls_report[concept_name]["recall"] = metrics.recall_score(y_pred=y_pred[i], y_true=y_val[i])
        cls_report[concept_name]["f1"] = metrics.f1_score(y_pred=y_pred[i], y_true=y_val[i])

    cls_report["accuracy_overall"] = (y_pred == y_val).sum() / (y_val.shape[0] * y_val.shape[1])
    cav_tensor = next(model.parameters())
    cavs = np.array(cav_tensor.detach().cpu())
    return cavs, cls_report


def fit_logistic_regression_multilabel(x_train, y_train, x_val, y_val, concept_names):
    # if cav_flattening_type == "avg_pooled" or cav_flattening_type == "max_pooled":
    start = time.time()
    concept_model = LogisticRegression(multi_class='ovr', solver='liblinear', verbose=1)
    multi_target_model = MultiOutputClassifier(concept_model, n_jobs=-1)
    multi_target_model.fit(x_train, y_train)
    y_pred = multi_target_model.predict(x_val)
    cavs = []
    cls_report = {}
    num_correct = 0
    for i, concept_name in enumerate(concept_names):
        cls_report[concept_name] = {}
    for i, concept_name in enumerate(concept_names):
        idx = (y_pred == i)
        cls_report[concept_name]["accuracy"] = metrics.accuracy_score(y_pred=y_pred[i], y_true=y_val[i])
        cls_report[concept_name]["precision"] = metrics.precision_score(y_pred=y_pred[i], y_true=y_val[i])
        cls_report[concept_name]["recall"] = metrics.recall_score(y_pred=y_pred[i], y_true=y_val[i])
        cls_report[concept_name]["f1"] = metrics.f1_score(y_pred=y_pred[i], y_true=y_val[i])
        num_correct += (sum(idx) * cls_report[concept_name]["accuracy"])
        cavs.append(multi_target_model.estimators_[i].coef_[0].tolist())
    cls_report["accuracy_overall"] = (y_pred == y_val).sum() / (y_val.shape[0] * y_val.shape[1])
    cavs = np.array(cavs)
    done = time.time()
    elapsed = done - start
    print("Time to train: " + str(elapsed) + " secs")
    return cavs, cls_report


def fit_logistic_regression_multiclass(x_train, y_train, x_val, y_val, concept_names):
    start = time.time()
    concept_model = LogisticRegression(multi_class='ovr', solver='liblinear', verbose=1)
    concept_model.fit(x_train, y_train)
    y_pred = concept_model.predict(x_val)
    cls_report = classification_report(y_val, y_pred, target_names=concept_names)
    print(cls_report)

    accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_val)
    print(f"Val set, Accuracy: {accuracy}")
    print(f"Coeffs: {len(concept_model.coef_)}")

    done = time.time()
    elapsed = done - start
    print("Time to train: " + str(elapsed) + " secs")

    if len(concept_model.coef_) == 1:
        cav = np.array([-concept_model.coef_[0], concept_model.coef_[0]])
        return cav, cls_report
    else:
        cav = np.array(concept_model.coef_)
        return cav, cls_report


# def generate_cavs(
#         logs,
#         model_arch,
#         dataset_name,
#         concept_names,
#         cav_flattening_type
# ):
#     bb_layers = ["layer3"]
#     activations_path = os.path.join(logs, "activations", "BB", model_arch, dataset_name)
#
#     start = time.time()
#     train_activations = load_activations(os.path.join(activations_path, "train_activations.h5"))
#     val_activations = load_activations(os.path.join(activations_path, "val_activations.h5"))
#     test_activations = load_activations(os.path.join(activations_path, "test_activations.h5"))
#
#     train_np_attr_GT = np.load(os.path.join(activations_path, "train_np_attr_GT.npy"))
#     val_np_attr_GT = np.load(os.path.join(activations_path, "val_np_attr_GT.npy"))
#     # test_np_attr_GT = np.load(os.path.join(activations_path, "test_np_attr_GT.npy"))
#     done = time.time()
#     elapsed = done - start
#     print("Time to load from disk: " + str(elapsed) + " secs")
#
#     multiclass_label_train_attr_GT = np.argmax(train_np_attr_GT, axis=1)
#     multiclass_label_val_attr_GT = np.argmax(val_np_attr_GT, axis=1)
#     # multiclass_label_test_attr_GT = np.argmax(test_np_attr_GT, axis=1)
#
#     flatten_train_activations, \
#     flatten_val_activations = flatten_activations(
#         bb_layers,
#         train_activations,
#         val_activations,
#         # test_activations,
#         cav_flattening_type
#     )
#
#     print(f"CAVs using {model_arch} layers: ")
#     train_cavs = {}
#     for l in bb_layers:
#         train_cavs[l] = []
#         # train_cavs[l] = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
#
#     for layer in bb_layers:
#         print(f"Train: {flatten_train_activations[layer].shape}")
#         print(f"Val: {flatten_val_activations[layer].shape}")
#         print(f"========== >> Running Logistic regression for layer: {layer} <<============")
#         cav, cls_report = generate_CAV_vectors(
#             flatten_train_activations[layer],
#             multiclass_label_train_attr_GT,
#             flatten_val_activations[layer],
#             multiclass_label_val_attr_GT,
#             concept_names
#         )
#         train_cavs[layer] = cav
#         print("------")
#
#     if cav_flattening_type == "max_pooled":
#         activations_file_name = "max_pooled_train_cavs.pkl"
#     elif cav_flattening_type == "flattened":
#         activations_file_name = "flattened_train_cavs.pkl"
#     else:
#         activations_file_name = "avg_pooled_train_cavs.pkl"
#     activations_file = open(os.path.join(activations_path, activations_file_name), "wb")
#     pickle.dump(train_cavs, activations_file)
#     activations_file.close()
#     print(f"Activation dictionary is saved in the location: {activations_path}")
#

def generate_cavs_using_pruning(
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
        multi_label=False,
        solver=None,
        multi_class=None
):
    start = time.time()
    train_activations = load_activations(os.path.join(activations_path, train_activations_file))
    val_activations = load_activations(os.path.join(activations_path, val_activation_file))
    # test_activations = load_activations(os.path.join(activations_path, test_activation_file))

    train_np_attr_GT = np.load(os.path.join(activations_path, train_GT_file))
    val_np_attr_GT = np.load(os.path.join(activations_path, val_GT_file))
    # test_np_attr_GT = np.load(os.path.join(activations_path, test_GT_file))
    done = time.time()
    elapsed = done - start
    print("Time to load from disk: " + str(elapsed) + " secs")

    if not multi_label:
        # this is for mnist
        multiclass_label_train_attr_GT = np.argmax(train_np_attr_GT, axis=1)
        multiclass_label_val_attr_GT = np.argmax(val_np_attr_GT, axis=1)
    # multiclass_label_test_attr_GT = np.argmax(test_np_attr_GT, axis=1)
    else:
        # this for other datasets like cubs, MIMIC
        multiclass_label_train_attr_GT = train_np_attr_GT
        multiclass_label_val_attr_GT = val_np_attr_GT

    flatten_train_activations, \
    flatten_val_activations = flatten_activations(
        bb_layers,
        train_activations,
        val_activations,
        # test_activations,
        cav_flattening_type
    )

    print("CAVs using Resnet18 layers: ")
    train_cavs = {}
    train_cav_cls_report = {}
    for l in bb_layers:
        train_cavs[l] = []
        train_cav_cls_report[l] = None

    for layer in bb_layers:
        print(f"Train: {flatten_train_activations[layer].shape}")
        print(f"Val: {flatten_val_activations[layer].shape}")
        # print(f"Test: {flatten_test_activations[layer].shape}")
        print(f"========== >> Running Logistic regression for layer: {layer} <<============")
        cav, cls_report = generate_CAV_vectors(
            flatten_train_activations[layer],
            multiclass_label_train_attr_GT,
            flatten_val_activations[layer],
            multiclass_label_val_attr_GT,
            concept_names,
            multi_label,
            cav_flattening_type
        )
        train_cavs[layer] = cav
        train_cav_cls_report[layer] = cls_report
        print("------")

    return train_cavs, train_cav_cls_report
