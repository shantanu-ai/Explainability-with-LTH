import os
import pickle

import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset.dataset_attributes_mnist import Dataset_attributes_mnist
from dataset.dataset_utils import get_dataset_with_attributes, get_transforms

"""
    Based on TCAV implementation by 
    https://github.com/agil27/TCAV_PyTorch/blob/master/tcav/tcav.py
    
"""


def load_activations(path):
    activations = {}
    with h5py.File(path, 'r') as f:
        for k, v in f.items():
            activations[k] = np.array(v)
    return activations


def directional_derivative(model_meta, cav, layer_name, class_name, outputs, cav_flattening_type):
    gradient = model_meta.calculate_grads(class_name, layer_name, outputs)
    if cav_flattening_type == "max_pooled":
        flatten_gradients = utils.flatten_cnn_activations_using_max_pooled(
            gradient,
            kernel_size=gradient.shape[-1],
            stride=1
        )
    elif cav_flattening_type == "flattened":
        flatten_gradients = utils.flatten_cnn_activations_using_activations(
            gradient
        )
    else:
        flatten_gradients = utils.flatten_cnn_activations_using_avg_pooled(
            gradient,
            kernel_size=gradient.shape[-1],
            stride=1
        )
    return np.dot(flatten_gradients, cav) > 0


def tcav_score(
        model,
        model_meta,
        data_loader,
        cav,
        layer_name,
        class_list,
        concept,
        cav_flattening_type,
        device
):
    derivatives = {}
    for k in class_list:
        derivatives[k] = []

    tcav_bar = tqdm(data_loader)
    tcav_bar.set_description('Calculating tcav score for %s' % concept)
    for x, y in tcav_bar:
        model.eval()
        x = x.to(device)
        outputs = model(x)
        k = 1 if outputs.item() >= 0.5 else 0
        if k in class_list:
            derivatives[k].append(
                directional_derivative(model_meta, cav, layer_name, k, outputs, cav_flattening_type)
            )

    eps = 1e-7
    score = np.zeros(len(class_list))
    for i, k in enumerate(class_list):
        if len(derivatives[k]) > 0:
            score[i] = np.count_nonzero(np.array(derivatives[k])) / len(derivatives[k])
        else:
            score[i] = np.count_nonzero(np.array(derivatives[k])) / (len(derivatives[k]) + eps)

    return score


def get_dir_derivative_binary_classification(
        model_meta,
        activation,
        bb_mid,
        bb_tail,
        cav_flattening_type,
        cav,
        device
):
    gradient = model_meta.estimate_grads_binary_classification(
        activation,
        bb_mid,
        bb_tail,
        device
    )

    if cav_flattening_type == "max_pooled":
        flatten_gradients_0 = utils.flatten_cnn_activations_using_max_pooled(
            gradient[0],
            kernel_size=gradient[0].shape[-1],
            stride=1
        )
        flatten_gradients_1 = utils.flatten_cnn_activations_using_max_pooled(
            gradient[1],
            kernel_size=gradient[1].shape[-1],
            stride=1
        )
    elif cav_flattening_type == "flattened":
        flatten_gradients_0 = utils.flatten_cnn_activations_using_activations(
            gradient[0]
        )
        flatten_gradients_1 = utils.flatten_cnn_activations_using_activations(
            gradient[1]
        )
    else:
        flatten_gradients_0 = utils.flatten_cnn_activations_using_avg_pooled(
            gradient[0],
            kernel_size=gradient[0].shape[-1],
            stride=1
        )
        flatten_gradients_1 = utils.flatten_cnn_activations_using_avg_pooled(
            gradient[1],
            kernel_size=gradient[1].shape[-1],
            stride=1
        )
    return np.dot(flatten_gradients_0, cav), np.dot(flatten_gradients_1, cav)


def get_data_loader_and_cavs(
        size,
        root,
        json_root,
        logs,
        model_arch,
        dataset_name,
        cav_flattening_type,
        batch_size=1,
        num_workers=4
):
    transform = get_transforms(size=size)
    test_set, test_attributes = get_dataset_with_attributes(
        data_root=root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="test",
        attribute_file="attributes.npy"
    )

    test_dataset = Dataset_attributes_mnist(test_set, test_attributes, transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    if cav_flattening_type == "max_pooled":
        cav_file_name = "max_pooled_train_cavs.pkl"
    elif cav_flattening_type == "flattened":
        cav_file_name = "flattened_train_cavs.pkl"
    else:
        cav_file_name = "avg_pooled_train_cavs.pkl"

    activations_path = os.path.join(logs, "activations", "BB", model_arch, dataset_name)
    cav_file = open(
        os.path.join(activations_path, cav_file_name),
        "rb")
    cavs = pickle.load(cav_file)
    return test_dataloader, cavs


def calculate_TCAV_score_binary_classification(
        bb_model,
        bb_model_meta,
        test_loader,
        cavs,
        cav_flattening_type,
        model_arch,
        bb_layer,
        device
):
    pos_cs_count_0 = 0
    pos_cs_count_1 = 0
    sample_size_0 = 0
    sample_size_1 = 0
    tcav_bar = tqdm(test_loader)
    for x, y in tcav_bar:
        x = x.to(device)
        output_prob = bb_model(x)
        if y.item() == 0:
            sample_size_0 += 1
            grads = bb_model_meta.estimate_grads_for_binary_classification(
                (1 - output_prob),
                bb_layer
            )
            flattened_grad = get_flattened_grads(cav_flattening_type, grads)
            flattened_grad = np.squeeze(flattened_grad)
            if np.dot(flattened_grad, cavs) > 0:
                pos_cs_count_0 += 1

        if y.item() == 1:
            sample_size_1 += 1
            grads = bb_model_meta.estimate_grads_for_binary_classification(
                output_prob,
                bb_layer
            )
            flattened_grad = get_flattened_grads(cav_flattening_type, grads)
            flattened_grad = np.squeeze(flattened_grad)
            if np.dot(flattened_grad, cavs) > 0:
                pos_cs_count_1 += 1
    return float(pos_cs_count_0) / float(sample_size_0), float(pos_cs_count_1) / float(sample_size_1)


def calculate_cavs_binary_classification(
        test_loader,
        concept_vectors,
        bb_model,
        bb_model_meta,
        cav_flattening_type,
        bb_layer,
        concept_names,
        class_list,
        model_arch

):
    device = utils.get_device()

    print(f"TCAV for layer: {bb_layer}")
    stat_dict = {}

    for idx, concept in enumerate(concept_names):
        print(f"========>> Concept: {concept} <<=======")
        cavs = concept_vectors[idx]
        cavs /= np.linalg.norm(cavs)
        print(f"Shape of cav: {cavs.shape}")
        print(f"Norm of cav: {np.linalg.norm(cavs)}")
        score = calculate_TCAV_score_binary_classification(
            bb_model,
            bb_model_meta,
            test_loader,
            cavs,
            cav_flattening_type,
            model_arch,
            bb_layer,
            device
        )
        # score = tcav_score(
        #     bb_model,
        #     bb_model_meta,
        #     test_loader,
        #     cavs,
        #     bb_layer,
        #     class_list,
        #     idx,
        #     cav_flattening_type,
        #     device
        # )
        print(f"TCAV for, class 0(Even): {score[0]}, class 1(Odd): {score[1]}")
        stat_dict[concept] = {"class_0": score[0], "class_1": score[1]}

    return stat_dict


def calculate_TCAV_score_multi_class_classification(
        bb_model,
        bb_model_meta,
        data_loader,
        cavs,
        cav_flattening_type,
        model_arch,
        concept,
        class_label_index,
        bb_layer,
        device
):
    tcav_bar = tqdm(data_loader)
    pos_grad = 0
    sample_size = 0
    bb_model.eval()
    for x, y, _ in tcav_bar:
        x = x.to(device)
        output_prob = bb_model(x)
        if class_label_index == y.item():
            sample_size += 1
            grads = bb_model_meta.estimate_grads_multiclass_classification(
                class_label_index,
                bb_layer,
                output_prob,
                device
            )
            flattened_grad = get_flattened_grads(cav_flattening_type, grads)
            flattened_grad = np.squeeze(flattened_grad)
            if np.dot(flattened_grad, cavs) > 0:
                pos_grad += 1

    return float(pos_grad) / float(sample_size)


def get_flattened_grads(cav_flattening_type, grads):
    if cav_flattening_type == "max_pooled":
        return utils.flatten_cnn_activations_using_max_pooled(
            grads,
            kernel_size=grads.shape[-1],
            stride=1
        )

    elif cav_flattening_type == "flattened":
        return utils.flatten_cnn_activations_using_activations(
            grads
        )

    else:
        return utils.flatten_cnn_activations_using_avg_pooled(
            grads,
            kernel_size=grads.shape[-1],
            stride=1
        )


def calculate_cavs_multiclass(
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

):
    device = utils.get_device()
    class_to_compute = class_labels[labels_index_list_TCAV[0]]
    concept_arr = []
    print(f"########## TCAV for model: {model_arch} ||"
          f" layer: {bb_layer} || "
          f"class: {class_to_compute} #############")
    for concept_id in concept_index_list_TCAV:
        concept_name = concept_names[concept_id]
        cavs = concept_vectors[concept_id]
        cavs /= np.linalg.norm(cavs)
        print(f"========>> Concept: {concept_name} <<=======")
        print(f"Shape of cav: {cavs.shape}")
        print(f"Norm of cav: {np.linalg.norm(cavs)}")

        TCAV_score = calculate_TCAV_score_multi_class_classification(
            bb_model,
            bb_model_meta,
            test_loader,
            cavs,
            cav_flattening_type,
            model_arch,
            concept_name,
            labels_index_list_TCAV[0],
            bb_layer,
            device
        )
        print(f"TCAV score for concept {concept_name}: {TCAV_score}")
        concept_arr.append({
            "concept_id": concept_id,
            "concept_name": concept_name,
            "TCAV_score": TCAV_score
        })

    stat = {
        "class_to_predict_name": class_to_compute,
        "class_to_predict_id": labels_index_list_TCAV[0],
        "concept_arr": concept_arr
    }
    return stat
