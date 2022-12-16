import copy
import os
import pickle
import random
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import lth_pruning.pruning_utils as pruning_utils
import utils
from model_factory.models import Classifier


class LTH:
    """
     Motivated from:
     1) LTH: https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch/
     2) Finetuning CUB using Resnet50: https://github.com/zhangyongshun/resnet_finetune_cub
    """

    def __init__(
            self,
            model_arch,
            num_classes,
            dataset_name,
            pretrained,
            transfer_learning,
            logs,
            prune_type,
            device,
            initialized_BB_weights=False,
            continue_pruning=False,
            last_check_pt=None,
            last_mask=None
    ):
        self.mask = None
        self.bb_initial_state_dict = None
        self.model = None
        self.continue_pruning = continue_pruning
        self.num_classes = num_classes
        self.device = device
        self.dataset_name = dataset_name
        self.prune_type = prune_type
        tb_path = os.path.join(
            logs,
            "tensorboard_logs",
            "Pruning",
            model_arch,
            dataset_name
        )
        utils.create_dir(
            path_dict={
                "path_name": tb_path,
                "path_type": "tensorboard"
            })
        now = datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')
        self.tb = SummaryWriter(f"{tb_path}/prune_type_{prune_type}_{now}")

        self.checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
        self.mask_path = os.path.join(self.checkpoint_path, "mask")

        if continue_pruning:
            # This step is for training when the full pruning iteration won't fit in a single job for 48 hours
            # For this step, the checkpoint of the last model needs to get passed.
            # **vvi: self.model will hold the parameters from the last saved model checkpoint and
            # self.bb_initial_state_dict will hold the parameters with which the original model was initialized
            # before the start of the training and pruning.
            # Also self.mask will hold the masks corresponding to the last self.model
            # ** load(self.model/self.mask) -> prune(self.model/self.mask) -> copy(self.bb_initial_state_dict)
            #    ->train(self.model) -> save(self.model) .. continue
            self.initialize_models_for_any_iteration_other_than_zero(
                model_arch,
                last_check_pt,
                last_mask,
                pretrained,
                transfer_learning
            )
        else:
            self.initialize_model_params_for_iteration_0(initialized_BB_weights, model_arch, pretrained,
                                                         transfer_learning)
            self.mask = pruning_utils.create_mask_for_BB(self.model)

    def initialize_models_for_any_iteration_other_than_zero(
            self,
            model_arch,
            last_check_pt,
            last_mask,
            pretrained,
            transfer_learning
    ):
        self.model = Classifier(
            model_arch,
            n_classes=self.num_classes,
            dataset_name=self.dataset_name,
            pretrained=pretrained,
            transfer_learning=transfer_learning
        ).to(self.device)

        # self.model holds the parameters from the last checkpoint
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, last_check_pt)))

        # the mask should be the same for the last loaded model
        with open(os.path.join(self.checkpoint_path, "mask", last_mask), "rb") as last_mask_path:
            self.mask = pickle.load(last_mask_path)

        initial_state = torch.load(os.path.join(
            self.checkpoint_path,
            f"initial_state_dict_prune_type_{self.prune_type}.pth.tar")
        )

        # self.bb_initial_state_dict holds the parameters from which was used to initialize the model
        # even before pruning and training
        self.bb_initial_state_dict = copy.deepcopy(initial_state.state_dict())

    def initialize_model_params_for_iteration_0(self, initialized_BB_weights, model_arch, pretrained,
                                                transfer_learning):
        if initialized_BB_weights:
            # this step is to train LTH by iterative pruning for iteration 0 onwards when the initialization
            # was separately done *** vvi to use iteration as 0
            # after this step self.model and self.bb_initial_state_dict will hold the same value and this is true
            # only after training the model for iteration 0 with 100% weights.
            # So for this step, continue_pruning=False and last_check_pt=None
            self.model = torch.load(os.path.join(
                self.checkpoint_path,
                f"initial_state_dict_prune_type_{self.prune_type}.pth.tar")
            )
            self.bb_initial_state_dict = copy.deepcopy(self.model.state_dict())
            print(f"Initial check point was loaded from: {self.checkpoint_path}")
            print(f"Initial parameters were loaded from: initial_state_dict_prune_type_{self.prune_type}.pth.tar")
        else:
            self.model = Classifier(
                model_arch, self.num_classes,
                dataset_name=self.dataset_name,
                pretrained=pretrained,
                transfer_learning=transfer_learning
            )
            self.model.to(self.device)

            # LTH (1) Init
            if self.dataset_name == "mnist":
                self.model.apply(pruning_utils.weight_init)
            utils.create_dir(
                path_dict={
                    "path_name": self.checkpoint_path,
                    "path_type": "checkpoint"
                })

            self.bb_initial_state_dict = copy.deepcopy(self.model.state_dict())
            torch.save(
                self.model,
                os.path.join(
                    self.checkpoint_path,
                    f"initial_state_dict_prune_type_{self.prune_type}.pth.tar")
            )

    def prune_by_percentile(self, percent, resample=False, reinit=False, **kwargs):
        # Calculate percentile value
        step = 0
        for name, param in self.model.named_parameters():
            # Prune weights, not biases
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)
                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, self.mask[step])
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                self.mask[step] = new_mask
                step += 1

    def original_initialization(self):
        step = 0
        for name, param in self.model.named_parameters():
            if "weight" in name:
                weight_device = param.device
                param.data = torch.from_numpy(
                    self.mask[step] * self.bb_initial_state_dict[name].cpu().numpy()
                ).to(
                    weight_device
                )
                step = step + 1
            if "bias" in name:
                param.data = self.bb_initial_state_dict[name]

    def get_model_optimization_params(self, lr):
        if self.dataset_name == "mnist":
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = None
            return criterion, optimizer, scheduler
        elif self.dataset_name == "cub":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            criterion = torch.nn.CrossEntropyLoss()
            return criterion, optimizer, scheduler

    def prune_and_train_BB(
            self,
            seed,
            epsilon,
            data_root,
            json_root,
            dataset_name,
            lr,
            img_size,
            batch_size,
            resample,
            prune_percent,
            prune_type,
            prune_iterations,
            start_iter,
            end_iter
    ):
        """
        LTH main steps:
        (1) Init
        (2) Train
        (3) Prune
        (4) Copy weights from (1)
        continue ..
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        start = time.time()
        train_loader, val_loader = pruning_utils.get_dataloader(
            data_root,
            json_root,
            dataset_name,
            img_size,
            batch_size
        )
        done = time.time()
        elapsed = done - start
        print("Time to load the dataset: " + str(elapsed) + " secs")

        non_zero_params = np.zeros(prune_iterations, float)
        reinit = True if prune_type == "reinit" else False
        prune_statistics = []

        prune_stat_path = os.path.join(self.checkpoint_path, "prune-statistics", "train")

        utils.create_dir(
            path_dict={
                "path_name": self.mask_path,
                "path_type": "mask-for-each-prune-iteration"
            })

        utils.create_dir(
            path_dict={
                "path_name": prune_stat_path,
                "path_type": "prune_stat_path-for-each-prune-iteration"
            })

        ITE = 1
        criterion, optimizer, scheduler = self.get_model_optimization_params(lr)
        for _ite in range(start_iter, prune_iterations):
            if _ite > 0:
                # (3) Prune
                # if all the pruning iteration won't fit in the current job, this setting needs to be used
                # if self.continue_pruning:
                #     no_of_times_prior_prune = start_iter - 1
                #     for i in range(no_of_times_prior_prune):
                #         self.prune_by_percentile(prune_percent, resample=resample, reinit=reinit)
                #     self.continue_pruning = False

                self.prune_by_percentile(prune_percent, resample=resample, reinit=reinit)

                if reinit:
                    self.model.apply(pruning_utils.weight_init)
                    step = 0
                    for name, param in self.model.named_parameters():
                        if 'weight' in name:
                            weight_device = param.device
                            param.data = torch.from_numpy(
                                param.data.cpu().numpy() * self.mask[step]
                            ).to(weight_device)
                            step = step + 1
                else:
                    # LTH (4) Copy weights from (1)
                    self.original_initialization()

                if self.dataset_name == "mnist":
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
                elif self.dataset_name == "cub":
                    optimizer = torch.optim.SGD(
                        self.model.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=1e-4
                    )
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

            print(f"\n------------------ Pruning Level [{ITE}:{_ite}/{prune_iterations}]: ----------------------")
            non_zero_params_ite = pruning_utils.print_non_zeros(self.model)
            non_zero_params[_ite] = non_zero_params_ite
            print(non_zero_params)
            with open(os.path.join(
                    self.mask_path,
                    f"{prune_type}_mask_non_zero_params_{non_zero_params_ite}_ite_{_ite}.pkl"
            ), 'wb') as fp:
                pickle.dump(self.mask, fp)
            # LTH (2)/(4) Train/ Retrain
            prune_iter_stats = self.train(
                _ite,
                end_iter,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                prune_type,
                scheduler,
                epsilon
            )

            prune_statistics.append(prune_iter_stats)
        with open(os.path.join(
                prune_stat_path,
                f"prune_stats_for_prune_type{prune_type}.pkl"
        ), 'wb') as fp:
            pickle.dump(prune_statistics, fp)

        print("Prune Statistics:=====>>")
        print(prune_statistics)

    def train(
            self,
            prune_ite,
            end_iter,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            prune_type,
            scheduler,
            epsilon
    ):
        utils.create_dir(
            path_dict={
                "path_name": os.path.join(self.checkpoint_path, "iter"),
                "path_type": "checkpoint-for-each-iteration"
            })
        best_accuracy = 0
        all_train_loss = np.zeros(end_iter, float)
        all_train_accuracy = np.zeros(end_iter, float)
        all_val_loss = np.zeros(end_iter, float)
        all_val_accuracy = np.zeros(end_iter, float)

        for iter_ in range(end_iter):
            # Validation
            iter_val_loss, iter_val_accuracy = self.validate(val_loader, criterion, iter_)
            # Save Weights
            if iter_val_accuracy > best_accuracy:
                print(f"[Saving best model] "
                      f"Best val accuracy changed from {best_accuracy} to {iter_val_accuracy}")
                best_accuracy = iter_val_accuracy
                # print(self.model.state_dict())
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.checkpoint_path,
                        f"best_val_prune_iteration_{prune_ite}_model_{prune_type}.pth.tar"
                    )
                )

            # Training
            iter_train_loss = 0
            iter_train_correct = 0
            self.model.train()
            with tqdm(total=len(train_loader)) as t:
                for batch_idx, data_tuple in enumerate(train_loader):
                    imgs, targets = self.get_image_label(data_tuple)
                    optimizer.zero_grad()
                    # imgs, targets = next(train_loader)
                    output = self.model(imgs)
                    train_loss = criterion(output, targets)
                    train_loss.backward()
                    iter_train_loss += train_loss.item()
                    iter_train_correct += utils.get_correct(output, targets, self.num_classes)

                    # Freezing Pruned weights by making their gradients Zero
                    for name, p in self.model.named_parameters():
                        if 'weight' in name:
                            tensor = p.data.cpu().numpy()
                            grad_tensor = p.grad.data.cpu().numpy()
                            # This sometimes does not make the grads zero, so
                            # we check if the grads are lower than a desired value
                            # grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                            grad_tensor = np.where(tensor < epsilon, 0, grad_tensor)
                            p.grad.data = torch.from_numpy(grad_tensor).to(self.device)
                    optimizer.step()

                    t.set_postfix(
                        iteration=f"{iter_}",
                        train_loss=f"{iter_train_loss:.6f}")
                    t.update()

            iter_train_accuracy = 100. * iter_train_correct / len(train_loader.dataset)
            iter_train_loss /= len(train_loader.dataset)
            all_train_loss[iter_] = iter_train_loss
            all_train_accuracy[iter_] = iter_train_accuracy
            all_val_loss[iter_] = iter_val_loss
            all_val_accuracy[iter_] = iter_val_accuracy
            if scheduler is not None:
                scheduler.step()

            print(f"Epoch: [{iter_ + 1}/{end_iter}] "
                  f"Train_loss: {iter_train_loss:.6f} "
                  f"Val_loss: {iter_val_loss:.6f} "
                  f"Train_Accuracy: {iter_train_accuracy:.6f} "
                  f"Val_Accuracy:{iter_val_accuracy:.6f} "
                  f"Best_Val_Accuracy: {best_accuracy:.6f} ")

            self.tb.add_scalar(
                f"Prune_iteration: {prune_ite}/Train_Loss",
                iter_train_loss,
                iter_ + 1
            )

            self.tb.add_scalar(
                f"Prune_iteration: {prune_ite}/Val_loss",
                iter_val_loss,
                iter_ + 1
            )

            self.tb.add_scalar(
                f"Prune_iteration: {prune_ite}/Train_Accuracy",
                iter_train_accuracy,
                iter_ + 1
            )

            self.tb.add_scalar(
                f"Prune_iteration: {prune_ite}/Val_Accuracy",
                iter_val_accuracy,
                iter_ + 1
            )

            torch.save(
                self.model.state_dict(),
                os.path.join(
                    os.path.join(self.checkpoint_path, "iter"),
                    f"prune_iteration_{prune_ite}_main_iteration_{iter_}_model_{prune_type}.pth.tar"
                )
            )

        return {
            "all_train_loss": all_train_loss,
            "all_train_accuracy": all_train_accuracy,
            "all_val_loss": all_val_loss,
            "all_val_accuracy": all_val_accuracy,
            "best_accuracy": best_accuracy
        }

    def validate(self, val_loader, criterion, iter_):
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_idx, data_tuple in enumerate(val_loader):
                    data, target = self.get_image_label(data_tuple)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    correct += utils.get_correct(output, target, self.num_classes)

                    t.set_postfix(
                        iteration=f"{iter_}",
                        val_loss=f"{val_loss:.6f}")
                    t.update()
            val_loss /= len(val_loader.dataset)
            accuracy = 100. * correct / len(val_loader.dataset)
        return val_loss, accuracy

    def get_image_label(self, data_tuple):
        if self.dataset_name == "cub":
            data, target, _ = data_tuple
            data, target = data.to(self.device), target.to(torch.long).to(self.device)
            return data, target
        elif self.dataset_name == "mnist":
            data, target = data_tuple
            data, target = data.to(self.device), target.to(torch.float32).to(self.device)
            target = target.reshape((target.shape[0], 1))
            return data, target
