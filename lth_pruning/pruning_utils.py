import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader

from dataset.dataset_cubs import Dataset_cub
from dataset.dataset_mnist import Dataset_mnist
from dataset.dataset_utils import get_dataset, get_transforms, get_dataset_with_image_and_attributes, get_transform_cub
from model_factory.models import Classifier


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def print_parameters(model):
    for name, param in model.named_parameters():
        print(name, param.size())


def create_mask_for_BB(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    # mask will be a 2D array where the 1st dimension will store
    # the number of layers which has weight paramters.
    # Ex: in ResNet18, there are 18 layers which has weights, so
    # the 1st dimension has a size of 41
    # The 2nd dimension will store 1's and has a size of the weights
    # from each layer
    mask = [None] * step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1

    print("Parameters with weights of BB along with masks: ")
    idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"name: {name}, param_size: {param.size()}, mask_size: {np.array(mask[idx]).shape}")
            idx += 1
    return mask


def print_non_zeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(
            f'{name:20} | '
            f'nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | '
            f'total_pruned = {total_params - nz_count :7} | '
            f'shape = {tensor.shape}')
    print(
        f'alive: {nonzero}, '
        f'pruned : {total - nonzero}, '
        f'total: {total}, '
        f'Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')
    return round((nonzero / total) * 100, 1)


def get_dataloader(data_root, json_root, dataset_name, img_size, batch_size):
    if dataset_name == "mnist":
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

    elif dataset_name == "cub":
        train_set, train_attributes = get_dataset_with_image_and_attributes(
            data_root=data_root,
            json_root=json_root,
            dataset_name=dataset_name,
            mode="train",
            attribute_file="attributes.npy"
        )

        val_set, val_attributes = get_dataset_with_image_and_attributes(
            data_root=data_root,
            json_root=json_root,
            dataset_name=dataset_name,
            mode="val",
            attribute_file="attributes.npy"
        )

        train_transform = get_transform_cub(size=img_size, data_augmentation=True)
        val_transform = get_transform_cub(size=img_size, data_augmentation=False)

        train_dataset = Dataset_cub(train_set, train_attributes, train_transform)
        train_loader = DataLoader(
            train_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        val_dataset = Dataset_cub(val_set, val_attributes, val_transform)
        val_loader = DataLoader(
            val_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        return train_loader, val_loader


def get_image_labels(data_tuple, dataset_name):
    if dataset_name == "cub":
        data, _, attribute = data_tuple
        return data, attribute
    elif dataset_name == "mnist":
        data, attribute = data_tuple
        return data, attribute


def get_image_target(data_tuple, dataset_name, device):
    if dataset_name == "cub":
        data, target, _ = data_tuple
        data, target = data.to(device), target.to(torch.long).to(device)
        return data, target
    elif dataset_name == "mnist":
        data, target = data_tuple
        data, target = data.to(device), target.to(torch.float32).to(device)
        target = target.reshape((target.shape[0], 1))
        return data, target


def load_BB_model_w_pruning(
        model_arch,
        num_classes,
        pretrained,
        transfer_learning,
        dataset_name,
        device,
        _ite,
        checkpoint_path
):
    model = Classifier(model_arch, num_classes, dataset_name, pretrained, transfer_learning)
    model.to(device)
    chk_pt_file = f"best_val_prune_iteration_{_ite}_model_lt.pth.tar"
    chk_pt_file_name = os.path.join(checkpoint_path, chk_pt_file)
    print("BB Model loaded from:")
    print(os.path.join(checkpoint_path, chk_pt_file))
    model_chk_pt = torch.load(chk_pt_file_name)
    model.load_state_dict(model_chk_pt)
    return model


def get_test_dataloader(
        dataset_name,
        data_root,
        json_root,
        batch_size,
        transform_params
):
    if dataset_name == "mnist":
        test_set = get_dataset(
            data_root=data_root,
            json_root=json_root,
            dataset_name=dataset_name,
            mode="test"
        )
        transform = get_transforms(size=transform_params["img_size"])
        test_dataset = Dataset_mnist(test_set, transform)
        return DataLoader(
            test_dataset,
            num_workers=4,
            batch_size=batch_size,
            shuffle=False
        )


def get_test_dataloader_cub(
        dataset_name,
        data_root,
        json_root,
        batch_size,
        img_size,
        attribute_file_name
):
    test_transform = get_transform_cub(size=img_size, data_augmentation=False)
    test_set, test_attributes = get_dataset_with_image_and_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="test",
        attribute_file=attribute_file_name
    )

    test_dataset = Dataset_cub(test_set, test_attributes, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    return test_dataloader


def get_class_index_for_TCAV(list1, list2):
    index_for_items_in_list1 = []
    for item in list1:
        index_for_items_in_list1.append(list2.index(item))

    return index_for_items_in_list1


def get_percent_weight_remains(dataset_name, model_arch):
    if dataset_name == "mnist" and model_arch == "Resnet_18":
        return [
            100.0, 90.0, 81.0, 72.9, 65.6, 59.1, 53.2, 47.8, 43.1, 38.8, 34.9, 31.4,
            28.3, 25.4, 22.9, 20.6, 18.6, 16.7, 15.0, 13.5, 12.2, 11.0, 9.9, 8.9,
            8.0, 7.2, 6.5, 5.9, 5.3, 4.7, 4.3, 3.9, 3.5, 3.1, 2.8
        ]
    elif dataset_name == "cub" and model_arch == "Resnet_50":
        return [
            100.0, 90.0, 81.0, 72.9, 65.6, 59.1, 53.2, 47.9, 43.1, 38.8, 34.9, 31.5,
            28.3, 25.5, 23.0, 20.7, 18.6, 16.8, 15.1, 13.6, 12.3, 11.0, 9.9, 9.0,
            8.1
        ]
