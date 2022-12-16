import os
import time

import matplotlib as mpl
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

import utils
from Saliency_Visualization.Grad_cam_Resnet_50 import GradCamModel
from dataset.dataset_cubs import Dataset_cub
from dataset.dataset_utils import get_transform_cub, get_dataset_with_image_and_attributes
from lth_pruning import pruning_utils


def visualize_w_GRAD_CAM(
        ite,
        dataset_name,
        checkpoint_path,
        img_arr,
        label_arr,
        img_id,
        num_classes,
        loss,
        device
):
    # for ite in range(15):
    chk_pt_file = f"best_val_prune_iteration_{ite}_model_lt.pth.tar"
    chk_pt_file_name = os.path.join(checkpoint_path, chk_pt_file)
    gcmodel = GradCamModel(chk_pt_file_name, dataset_name, num_classes).to(device)

    model_chk_pt = torch.load(chk_pt_file_name)
    gcmodel.load_state_dict(model_chk_pt)
    img = img_arr[img_id].to(device)
    img_label = label_arr[img_id].to(device)
    out, acts = gcmodel(img)
    acts = acts.detach().cpu()
    test_loss = loss(out, img_label).to(device)
    test_loss.backward()
    grads = gcmodel.get_act_grads().detach().cpu()
    pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()
    for i in range(acts.shape[1]):
        acts[:, i, :, :] += pooled_grads[i]
    heatmap_j = torch.mean(acts, dim=1).squeeze()
    heatmap_j_max = heatmap_j.max(axis=0)[0]
    heatmap_j /= heatmap_j_max
    heatmap_j = resize(heatmap_j, (224, 224), preserve_range=True)
    cmap = mpl.cm.get_cmap("jet", 256)
    heatmap_j2 = cmap(heatmap_j, alpha=0.5)
    heatmap_j3 = (heatmap_j > 0.75)
    heatmap_j3 = np.expand_dims(heatmap_j3, 2)

    return heatmap_j2, heatmap_j3


def get_img_label_arr(data_loader, label_id):
    img_arr = []
    label_arr = []
    for idx, (img, label, _) in enumerate(data_loader):
        if label.item() == label_id:
            img_arr.append(img)
            label_arr.append(label)

    return img_arr, label_arr


def get_img_label_arr_mnist(data_loader, label_id):
    img_arr = []
    label_arr = []
    tcav_bar = tqdm(data_loader)
    for img, label in tcav_bar:
        if label.item() == label_id:
            img_arr.append(img)
            label_arr.append(label)

    return img_arr, label_arr


def generate_heatmap_cub(
        ite,
        img_size,
        data_root,
        json_root,
        dataset_name,
        attribute_file_name,
        logs,
        model_arch,
        prune_type,
        cav_flattening_type,
        labels_for_tcav,
        class_labels,
        img_id,
        num_classes,
        device
):
    print("------------------------------")
    print(f"Pruning iteration: {ite}")
    print("------------------------------")

    img_dataset, attributes = get_dataset_with_image_and_attributes(
        data_root,
        json_root,
        dataset_name,
        "test",
        attribute_file_name
    )
    transform = get_transform_cub(size=img_size, data_augmentation=False)
    dataset = Dataset_cub(img_dataset, attributes, transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    label_id = pruning_utils.get_class_index_for_TCAV(labels_for_tcav, class_labels)[0]
    heatmap_save_path = os.path.join(
        logs,
        "predictions",
        "prune-statistics",
        model_arch,
        dataset_name,
        f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}",
        labels_for_tcav[0],
        "heatmaps"
    )

    utils.create_dir(
        path_dict={
            "path_name": heatmap_save_path,
            "path_type": "heatmap_save_path"
        })

    loss = torch.nn.CrossEntropyLoss()
    checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
    img_arr, label_arr = get_img_label_arr(data_loader, label_id)

    heatmap_j2_arr, heatmap_j3_arr = visualize_w_GRAD_CAM(
        ite,
        dataset_name,
        checkpoint_path,
        img_arr,
        label_arr,
        img_id,
        num_classes,
        loss,
        device
    )

    np.save(
        os.path.join(heatmap_save_path, f"heatmap_j2_prune_iteration_{ite}_image_id_{img_id}.npy"),
        heatmap_j2_arr
    )
    np.save(
        os.path.join(heatmap_save_path, f"heatmap_j3_prune_iteration_{ite}_image_id_{img_id}.npy"),
        heatmap_j3_arr
    )


def generate_heatmap_mnist(
        ite,
        img_size,
        data_root,
        json_root,
        dataset_name,
        attribute_file_name,
        logs,
        model_arch,
        prune_type,
        cav_flattening_type,
        label_id,
        img_id,
        num_classes,
        device
):
    print("------------------------------")
    print(f"Pruning iteration: {ite}")
    print("------------------------------")

    transform_params = {
        "img_size": img_size
    }
    start = time.time()
    data_loader = pruning_utils.get_test_dataloader(
        dataset_name,
        data_root,
        json_root,
        batch_size=1,
        transform_params=transform_params
    )
    done = time.time()
    elapsed = done - start
    print(f"Time to load dataset: {elapsed}")
    heatmap_save_path = os.path.join(
        logs,
        "predictions",
        "prune-statistics",
        model_arch,
        dataset_name,
        f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}"
        "heatmaps"
    )

    utils.create_dir(
        path_dict={
            "path_name": heatmap_save_path,
            "path_type": "heatmap_save_path"
        })

    loss = torch.nn.CrossEntropyLoss()
    checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
    img_arr, label_arr = get_img_label_arr_mnist(data_loader, label_id)

    heatmap_j2_arr, heatmap_j3_arr = visualize_w_GRAD_CAM(
        ite,
        dataset_name,
        checkpoint_path,
        img_arr,
        label_arr,
        img_id,
        num_classes,
        loss,
        device
    )

    np.save(
        os.path.join(heatmap_save_path, f"heatmap_j2_prune_iteration_{ite}_image_id_{img_id}.npy"),
        heatmap_j2_arr
    )
    np.save(
        os.path.join(heatmap_save_path, f"heatmap_j3_prune_iteration_{ite}_image_id_{img_id}.npy"),
        heatmap_j3_arr
    )
