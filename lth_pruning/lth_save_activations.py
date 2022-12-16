import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset.dataset_attributes_mnist import Dataset_attributes_mnist
from dataset.dataset_cubs import Dataset_cub
from dataset.dataset_utils import get_transforms, get_dataset_with_attributes, \
    get_dataset_with_image_and_attributes, get_transform_cub
from lth_pruning.pruning_utils import load_BB_model_w_pruning, get_image_labels
from model_factory.model_meta import Model_Meta


def create_activation_DB(dataloader, bb_layers, device, model, model_meta, dataset_name):
    attr_GT = torch.FloatTensor()
    activations = {}
    for l in bb_layers:
        activations[l] = []

    with torch.no_grad():
        with tqdm(total=len(dataloader)) as t:
            for batch_id, data_tuple in enumerate(dataloader):
                image, attribute = get_image_labels(data_tuple, dataset_name)
                image = image.to(device)
                _ = model(image).cpu().detach()
                for l in bb_layers:
                    z = model_meta.model_activations_store[l].cpu().detach().numpy()
                    activations[l].append(z)
                t.set_postfix(batch_id='{0}'.format(batch_id))
                attr_GT = torch.cat((attr_GT, attribute), dim=0)
                t.update()

    print("Activations are generated..")
    for l in bb_layers:
        activations[l] = np.concatenate(activations[l], axis=0)
        print(activations[l].shape)
    print(attr_GT.cpu().numpy().shape)
    return activations, attr_GT.cpu().numpy()


def get_dataloaders(dataset_name, img_size, data_root, json_root, batch_size, attribute_file_name):
    if dataset_name == "mnist":
        return load_datasets_mnist(
            img_size,
            data_root,
            json_root,
            dataset_name,
            batch_size,
            attribute_file_name
        )
    elif dataset_name == "cub":
        return load_datasets_cub(
            img_size,
            data_root,
            json_root,
            dataset_name,
            batch_size,
            attribute_file_name
        )


def load_datasets_cub(img_size, data_root, json_root, dataset_name, batch_size, attribute_file_name):
    start = time.time()

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
    done = time.time()
    elapsed = done - start
    print("Time to load the train dataset from disk: " + str(elapsed) + " secs")

    start = time.time()
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
    done = time.time()
    elapsed = done - start
    print("Time to load the val dataset from disk: " + str(elapsed) + " secs")

    start = time.time()
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
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")

    return train_dataloader, val_dataloader, test_dataloader


def load_datasets_mnist(img_size, data_root, json_root, dataset_name, batch_size, attribute_file_name):
    start = time.time()
    transform = get_transforms(size=img_size)
    train_set, train_attributes = get_dataset_with_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="train",
        attribute_file=attribute_file_name
    )

    train_dataset = Dataset_attributes_mnist(train_set, train_attributes, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    done = time.time()
    elapsed = done - start
    print("Time to load the train dataset from disk: " + str(elapsed) + " secs")

    start = time.time()
    val_set, val_attributes = get_dataset_with_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="val",
        attribute_file=attribute_file_name
    )

    val_dataset = Dataset_attributes_mnist(val_set, val_attributes, transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    done = time.time()
    elapsed = done - start
    print("Time to load the val dataset from disk: " + str(elapsed) + " secs")

    start = time.time()
    test_set, test_attributes = get_dataset_with_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="test",
        attribute_file=attribute_file_name
    )

    test_dataset = Dataset_attributes_mnist(test_set, test_attributes, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")

    return train_dataloader, val_dataloader, test_dataloader


def save_activations_with_Pruning(
        seed,
        num_classes,
        pretrained,
        transfer_learning,
        prune_type,
        dataset_name,
        data_root,
        json_root,
        batch_size,
        img_size,
        start_iter,
        prune_iterations,
        logs,
        model_arch,
        bb_layers,
        attribute_file_name,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
    activations_path = os.path.join(
        logs,
        "activations",
        "Pruning",
        model_arch,
        dataset_name,
        "BB_act",
        f"Prune_type_{prune_type}"
    )

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        dataset_name,
        img_size,
        data_root,
        json_root,
        batch_size,
        attribute_file_name,
    )

    utils.create_dir(
        path_dict={
            "path_name": activations_path,
            "path_type": "activations-of-BB"
        })

    for _ite in range(start_iter, prune_iterations):
        print(f"Prune iteration: {_ite} =======================================>")
        bb_model = load_BB_model_w_pruning(
            model_arch,
            num_classes,
            pretrained,
            transfer_learning,
            dataset_name,
            device,
            _ite,
            checkpoint_path
        )

        bb_model.eval()
        model_meta = Model_Meta(bb_model, bb_layers)
        start = time.time()
        train_activations, train_np_attr_GT = create_activation_DB(
            train_dataloader,
            bb_layers,
            device,
            bb_model,
            model_meta,
            dataset_name
        )

        utils.save_activations(
            activations_path,
            f"train_activations_prune_iteration_{_ite}.h5",
            bb_layers,
            train_activations
        )
        np.save(os.path.join(activations_path, f"train_np_attr_GT_prune_iteration_{_ite}.npy"), train_np_attr_GT)
        done = time.time()
        elapsed = done - start
        print("Time to create train activations: " + str(elapsed) + " secs")

        start = time.time()
        val_activations, val_np_attr_GT = create_activation_DB(
            val_dataloader,
            bb_layers,
            device,
            bb_model,
            model_meta,
            dataset_name
        )
        utils.save_activations(
            activations_path,
            f"val_activations_prune_iteration_{_ite}.h5",
            bb_layers,
            val_activations
        )
        np.save(os.path.join(activations_path, f"val_np_attr_GT_prune_iteration_{_ite}.npy"), val_np_attr_GT)
        done = time.time()
        elapsed = done - start
        print("Time to create val activations: " + str(elapsed) + " secs")

        start = time.time()
        test_activations, test_np_attr_GT = create_activation_DB(
            test_dataloader,
            bb_layers,
            device,
            bb_model,
            model_meta,
            dataset_name
        )
        utils.save_activations(
            activations_path,
            f"test_activations_prune_iteration_{_ite}.h5",
            bb_layers,
            test_activations
        )
        np.save(os.path.join(activations_path, f"test_np_attr_GT_prune_iteration_{_ite}.npy"), test_np_attr_GT)
        done = time.time()
        elapsed = done - start
        print("Time to create test activations: " + str(elapsed) + " secs")
