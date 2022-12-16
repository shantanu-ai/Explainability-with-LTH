import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset_cubs import Dataset_cub
from dataset.dataset_mnist import Dataset_mnist
from dataset.dataset_utils import get_dataset, get_transforms, get_transform_cub, get_dataset_with_image_and_attributes


def dissect_bb_model(model_arch, bb_model):
    if model_arch == "Resnet_18":
        print("Resnet_18")
        resnet = list(list(bb_model.children())[0].children())
        mid = torch.nn.Sequential(*resnet[7:9])
        tail = torch.nn.Sequential(*resnet[9])
        return mid, tail
    elif model_arch == "Resnet_50":
        resnet = list(list(bb_model.children())[0].children())
        mid = torch.nn.Sequential(*resnet[7:9])
        tail = torch.nn.Sequential(resnet[9])
        return mid, tail


def get_normalized_vc_using_pooling(
        activations,
        torch_concept_vector,
        th,
        val_after_th):
    bs, ch = activations.size(0), activations.size(1)
    vc = torch.matmul(
        activations.reshape((bs, ch, -1)).permute((0, 2, 1)),
        torch_concept_vector.T
    ).reshape((bs, -1))
    th_fn = torch.nn.Threshold(threshold=th, value=val_after_th)
    th_vc = th_fn(vc)
    norm_vc = torch.nn.functional.normalize(th_vc, p=2, dim=1)
    return norm_vc


def get_normalized_vc_using_flattening(
        activations,
        torch_concept_vector,
        th,
        val_after_th):
    bs = activations.size(0)
    vc = torch.matmul(
        activations.reshape((bs, -1)),
        torch_concept_vector.T
    )
    th_fn = torch.nn.Threshold(threshold=th, value=val_after_th)
    th_vc = th_fn(vc)
    norm_vc = torch.nn.functional.normalize(th_vc, p=2, dim=1)
    return norm_vc


def get_normalized_vc(
        activations,
        torch_concept_vector,
        th,
        val_after_th,
        cav_flattening_type):
    if cav_flattening_type == "max_pooled" or cav_flattening_type == "avg_pooled":
        return get_normalized_vc_using_pooling(
            activations,
            torch_concept_vector,
            th,
            val_after_th)
    elif cav_flattening_type == "flattened":
        return get_normalized_vc_using_flattening(
            activations,
            torch_concept_vector,
            th,
            val_after_th)


def get_concept_to_pred(
        concept_to_act,
        bs,
        activations,
        bb_model_mid,
        bb_model_tail):
    concept_to_act = concept_to_act.reshape(
        bs,
        activations.size(1),
        activations.size(2),
        activations.size(3)
    )

    prob_mid = bb_model_mid(concept_to_act)
    bs, ch, h, w = prob_mid.size()
    prob_mid = prob_mid.reshape(bs, ch * h * w)
    concept_to_pred = bb_model_tail(prob_mid)
    return concept_to_pred


def get_g_model_ip_op_size(
        loader,
        device,
        bb_model,
        bb_model_meta,
        torch_concept_vector,
        bb_layer,
        cav_flattening_type,
        dataset_name
):
    if dataset_name == "mnist":
        images, _ = next(iter(loader))
    elif dataset_name == "cub":
        images, _, _ = next(iter(loader))
    _ = bb_model(images.to(device))
    activations = bb_model_meta.model_activations_store[bb_layer]
    if cav_flattening_type == "max_pooled" or cav_flattening_type == "avg_pooled":
        g_model_ip_size = activations.size(-1) * activations.size(-2) * torch_concept_vector.size(0)
        g_model_op_size = activations.size(-1) * activations.size(-2) * torch_concept_vector.size(1)
        return g_model_ip_size, g_model_op_size
    elif cav_flattening_type == "flattened":
        g_model_ip_size = torch_concept_vector.size(0)
        g_model_op_size = activations.size(1) * activations.size(2) * activations.size(3)
        return g_model_ip_size, g_model_op_size


def get_dataloader_mnist(data_root, json_root, dataset_name, img_size, batch_size):
    train_set = get_dataset(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="train"
    )

    val_set = get_dataset(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="val"
    )

    transform = get_transforms(size=img_size)
    train_dataset = Dataset_mnist(train_set, transform)
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataset = Dataset_mnist(val_set, transform)
    val_loader = DataLoader(
        val_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def get_dataloader_cub(data_root, json_root, dataset_name, img_size, batch_size, attribute_file_name):
    train_transform = get_transform_cub(size=img_size, data_augmentation=True)
    train_set, train_attributes = get_dataset_with_image_and_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="train",
        attribute_file=attribute_file_name
    )

    train_dataset = Dataset_cub(train_set, train_attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    val_transform = get_transform_cub(size=img_size, data_augmentation=False)
    val_set, val_attributes = get_dataset_with_image_and_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="val",
        attribute_file=attribute_file_name
    )

    val_dataset = Dataset_cub(val_set, val_attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_dataloader, val_dataloader


def get_concept_vectors(logs, bb_layer, cav_flattening_type, model_arch, dataset_name):
    start = time.time()
    cav_vector_file = ""
    if cav_flattening_type == "max_pooled":
        cav_vector_file = "max_pooled_train_cavs.pkl"
    elif cav_flattening_type == "flattened":
        cav_vector_file = "flattened_train_cavs.pkl"
    elif cav_flattening_type == "avg_pooled":
        cav_vector_file = "avg_pooled_train_cavs.pkl"

    cav_path = os.path.join(logs, "activations", "BB", model_arch, dataset_name)
    cav_file = open(
        os.path.join(cav_path, cav_vector_file),
        "rb")
    cavs = pickle.load(cav_file)[bb_layer]
    for i in range(cavs.shape[0]):
        cavs[i] /= np.linalg.norm(cavs[i])
    done = time.time()
    elapsed = done - start
    print("Time to load the concepts from disk: " + str(elapsed) + " secs")
    return cavs


def get_concept_vectors_for_pruning(_ite, cav_path, bb_layer, cav_flattening_type):
    start = time.time()
    cav_vector_file = ""
    if cav_flattening_type == "max_pooled":
        cav_vector_file = f"max_pooled_train_cavs_prune_iteration_{_ite}.pkl"
    elif cav_flattening_type == "flattened":
        cav_vector_file = f"flattened_train_cavs_prune_iteration_{_ite}.pkl"
    elif cav_flattening_type == "avg_pooled":
        cav_vector_file = f"avg_pooled_train_cavs_prune_iteration_{_ite}.pkl"

    cav_file = open(
        os.path.join(cav_path, cav_vector_file),
        "rb")

    print("Cav vector is loaded from: \n")
    print(os.path.join(cav_path, cav_vector_file))
    cavs = pickle.load(cav_file)[bb_layer]
    for i in range(cavs.shape[0]):
        cavs[i] /= np.linalg.norm(cavs[i])
    done = time.time()
    elapsed = done - start
    print("Time to load the concepts from disk: " + str(elapsed) + " secs")
    return cavs
