import json
import os

import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder


def clean_names(names: list) -> list:
    names = \
        [
            name.replace("::", "_")
                .replace(":", "_")
                .replace(".", "_")
                .replace(" ", "_")
                .replace("\n", "")
                .replace("-", "")
                .replace("(", "")
                .replace(")", "")
            for name in names
        ]
    return names


def get_dataset(
        data_root,
        json_root,
        dataset_name,
        mode
):
    data_json = os.path.join(
        json_root,
        f"{mode}_samples_{dataset_name}.json"
    )

    if os.path.isfile(data_json):
        with open(os.path.join(data_json), "r") as f:
            json_file = json.load(f)
            data_samples = json_file["samples"]

    print(f"Length of the [{mode}] dataset: {len(data_samples)}")
    img_set = ImageFolder(data_root)
    dataset = [img_set[index] for index in data_samples]

    return dataset


def get_transform_cub(size, data_augmentation=False):
    resize = int(size * 0.9)
    if data_augmentation:
        train_transform_list = [
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )]
        return transforms.Compose(train_transform_list)
    else:
        return transforms.Compose([
            transforms.Resize(size=resize),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataset_with_image_and_attributes(
        data_root,
        json_root,
        dataset_name,
        mode,
        attribute_file
):
    data_json = os.path.join(
        json_root,
        f"{mode}_samples_{dataset_name}.json"
    )

    if os.path.isfile(data_json):
        with open(os.path.join(data_json), "r") as f:
            json_file = json.load(f)
            data_samples = json_file["samples"]

    print(f"Length of the [{mode}] dataset: {len(data_samples)}")
    img_set = ImageFolder(data_root)
    img_dataset = [img_set[index] for index in data_samples]
    attributes = np.load(os.path.join(data_root, attribute_file))[data_samples]

    return img_dataset, attributes


def get_dataset_with_attributes(
        data_root,
        json_root,
        dataset_name,
        mode,
        attribute_file
):
    data_json = os.path.join(
        json_root,
        f"{mode}_samples_{dataset_name}.json"
    )

    if os.path.isfile(data_json):
        with open(os.path.join(data_json), "r") as f:
            json_file = json.load(f)
            data_samples = json_file["samples"]

    attributes = np.load(os.path.join(data_root, attribute_file))[data_samples]
    print(f"Length of the [{mode}] dataset: {len(data_samples)}")
    img_set = ImageFolder(data_root)
    img_dataset = [img_set[index] for index in data_samples]
    print("")

    return img_dataset, attributes


def get_transforms(size=224, dataset="mnist"):
    resize = int(size * 0.9)
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(size=resize),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        return transform
