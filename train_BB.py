import argparse
import os
import random
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset.dataset_mnist import Dataset_mnist
from dataset.dataset_utils import get_dataset, get_transforms
from model_factory.models import Classifier
from run_manager import RunManager

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))


def fit(
        img_size,
        batch_size,
        data_root,
        json_root,
        dataset_name,
        model_arch,
        num_classes,
        pretrained,
        transfer_learning,
        lr,
        epochs,
        logs,
        seed
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    final_parameters = OrderedDict(
        arch=[model_arch],
        dataset=[dataset_name],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    run_id = utils.get_runs(final_parameters)[0]
    device = utils.get_device()
    print(f"Device: {device}")

    model = Classifier(model_arch, num_classes, pretrained, transfer_learning)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Loss Function
    criterion = nn.BCELoss()

    # Loading model to device
    checkpoint_path = os.path.join(logs, "chk_pt", "BB", run_id.arch, run_id.dataset)
    tb_path = os.path.join(logs, "tensorboard_logs", "BB")

    try:
        os.makedirs(checkpoint_path, exist_ok=True)
        print("Checkpoint directory is created successfully at:")
        print(checkpoint_path)
    except OSError as error:
        print(f"Checkpoint directory {checkpoint_path} can not be created")

    try:
        os.makedirs(tb_path, exist_ok=True)
        print("Tensorboard_path directory is created successfully at:")
        print(tb_path)
    except OSError as error:
        print(f"Tensorboard_path directory {tb_path} can not be created")

    run_manager = RunManager(checkpoint_path, tb_path, train_loader, val_loader)
    run_manager.begin_run(run_id)

    best_val_acc = 0
    for epoch in range(epochs):
        run_manager.begin_epoch()
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(torch.float32)
                labels = labels.to(device)
                labels = labels.reshape((labels.shape[0], 1))
                optimizer.zero_grad()

                # Forward
                y_hat = model(images)
                # Calculating Loss
                train_loss = criterion(y_hat, labels)

                # Backward
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(y_hat, labels)

                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (images, labels) in enumerate(val_loader):
                    images = images.to(device)
                    labels = labels.to(torch.float32)
                    labels = labels.to(device)
                    labels = labels.reshape((labels.shape[0], 1))
                    y_hat = model(images)

                    val_loss = criterion(y_hat, labels)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(y_hat, labels)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(model)

        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} "
              f"Val_Accuracy: {round(run_manager.get_final_val_accuracy(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} ")

    run_manager.end_run()


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config/BB_mnist.yaml")
    parser.add_argument(
        "--main_dir", "-m", default="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/")
    args = parser.parse_args()
    main_dir = args.main_dir

    # run config
    with open(os.path.join(main_dir, args.config)) as config_file:
        config = yaml.safe_load(config_file)
        _img_size = config["img_size"]
        _seed = config["seed"]
        _dataset_name = config["dataset_name"]
        _data_root = config["data_root"]
        _json_root = config["json_root"]
        _model_arch = config["model_arch"]
        _pretrained = config["pretrained"]
        _transfer_learning = config["transfer_learning"]
        _batch_size = config["batch_size"]
        _lr = config["lr"]
        _logs = config["logs"]
        _num_classes = config["num_classes"]
        _epochs = config["epochs"]

        fit(
            _img_size,
            _batch_size,
            _data_root,
            _json_root,
            _dataset_name,
            _model_arch,
            _num_classes,
            _pretrained,
            _transfer_learning,
            _lr,
            _epochs,
            _logs,
            _seed
        )
