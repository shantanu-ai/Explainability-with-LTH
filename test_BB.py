import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset.dataset_mnist import Dataset_mnist
from dataset.dataset_utils import get_dataset, get_transforms
from model_factory.models import Classifier

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))


def eval(
        img_size,
        batch_size,
        data_root,
        json_root,
        dataset_name,
        model_arch,
        num_classes,
        pretrained,
        transfer_learning,
        logs,
        chk_pt_path,
        seed
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_set = get_dataset(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="test"
    )

    transform = get_transforms(size=img_size)
    test_dataset = Dataset_mnist(test_set, transform)
    test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False
    )

    device = utils.get_device()
    print(f"Device: {device}")

    model = Classifier(model_arch, num_classes, pretrained, transfer_learning).to(device)
    checkpoint_path = os.path.join(logs, "chk_pt", "BB", model_arch, dataset_name, chk_pt_path)
    output_path = os.path.join(logs, "predictions", "BB", model_arch, dataset_name)

    try:
        os.makedirs(output_path, exist_ok=True)
        print("output prediction directory is created successfully at:")
        print(output_path)
    except OSError as error:
        print(f"output prediction directory {output_path} can not be created")

    model_chk_pt = torch.load(checkpoint_path)
    model.load_state_dict(model_chk_pt)
    model.eval()

    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict = torch.FloatTensor().cuda()
    out_prob_arr = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(torch.float32)
                labels = labels.to(device)
                labels = labels.reshape((labels.shape[0], 1))
                y_hat = model(images)
                out_prob_arr.append(y_hat)
                out_put_predict = torch.cat((out_put_predict, y_hat), dim=0)
                out_put_GT = torch.cat((out_put_GT, labels), dim=0)
                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    test_accuracy = utils.get_correct(out_put_predict, out_put_GT) / out_put_GT.size(0)
    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_np = out_put_predict.cpu().numpy()
    y_hat = np.where(out_put_predict_np > 0.5, 1, 0)
    np.save(os.path.join(output_path, "out_put_GT.npy"), out_put_GT_np)
    np.save(os.path.join(output_path, "out_put_predict_np.npy"), out_put_predict_np)

    print(f"Accuracy: {utils.cal_accuracy(out_put_GT_np, y_hat)}")
    print(f"Precision: {utils.cal_precision(out_put_GT_np, y_hat)}")
    print(f"Recall: {utils.cal_recall(out_put_GT_np, y_hat)}")
    print(f"RocAUC: {utils.cal_roc_auc(out_put_GT_np, y_hat)}")
    print(f"F1 score: {utils.cal_f1_score(out_put_GT_np, y_hat)}")


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
        _chk_pt_path = config["chk_pt_path"]
        eval(
            _img_size,
            _batch_size,
            _data_root,
            _json_root,
            _dataset_name,
            _model_arch,
            _num_classes,
            _pretrained,
            _transfer_learning,
            _logs,
            _chk_pt_path,
            _seed
        )
