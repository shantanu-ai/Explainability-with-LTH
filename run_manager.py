import time

from torch.utils.tensorboard import SummaryWriter

from utils import *


class RunManager:
    """
    This class creates manages different parameters based on each run.
    """

    def __init__(self, prune_ite, checkpoint_path, tb_path, train_loader, val_loader):
        """
        Initialized each parameters of each run.
        """
        self.prune_ite = prune_ite
        self.checkpoint_path = checkpoint_path
        self.tb_path = tb_path
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epoch_id = 0
        self.epoch_train_loss = 0
        self.epoch_val_loss = 0
        self.epoch_id_total_train_correct = 0
        self.epoch_id_total_val_correct = 0
        self.best_val_accuracy = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_id = 0
        self.run_data = []
        self.run_start_time = None
        self.epoch_duration = None

        self.tb = None
        self.train_loss = None
        self.val_loss = None
        self.train_accuracy = None
        self.val_accuracy = None

    def begin_run(self, run):
        """
        Records all the parameters at the start of each run.

        :param run:
        :param network: cnn model
        :param loader: pytorch data loader
        :param device: {cpu or gpu}
        :param type_of_bn: whether {batch normalization, no batch normalization or dropout}

        :return: none
        """
        self.run_start_time = time.time()

        self.run_id += 1
        self.run_params = run
        self.tb = SummaryWriter(f"{self.tb_path}/{run}")

    def end_run(self):
        """
        Records all the parameters at the end of each run.

        :return: none
        """
        self.tb.close()
        self.epoch_id = 0

    def begin_epoch(self):
        """
        Records all the parameters at the start of each epoch.

        :return: none
        """
        self.epoch_start_time = time.time()

        self.epoch_id += 1
        self.epoch_train_loss = 0
        self.epoch_val_loss = 0
        self.epoch_id_total_train_correct = 0
        self.epoch_id_total_val_correct = 0

    def end_epoch(self, model):
        """
        Records all the parameters at the end of each epoch.

        :return: none
        """
        self.epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        self.train_loss = self.epoch_train_loss / len(self.train_loader.dataset)
        self.val_loss = self.epoch_val_loss / len(self.val_loader.dataset)
        self.train_accuracy = self.epoch_id_total_train_correct / len(self.train_loader.dataset)
        self.val_accuracy = self.epoch_id_total_val_correct / len(self.val_loader.dataset)

        self.tb.add_scalar(
            "Plots/Train_Loss",
            self.train_loss,
            self.epoch_id
        )
        self.tb.add_scalar(
            "Plots/Val_Loss",
            self.val_loss,
            self.epoch_id
        )

        self.tb.add_scalar(
            "Plots/Train_correct",
            self.epoch_id_total_train_correct,
            self.epoch_id
        )
        self.tb.add_scalar(
            "Plots/Val_correct",
            self.epoch_id_total_val_correct,
            self.epoch_id
        )

        self.tb.add_scalar("Plots/Train_accuracy", 100 * self.train_accuracy, self.epoch_id)
        self.tb.add_scalar("Plots/Val_accuracy", 100 * self.val_accuracy, self.epoch_id)

        # for name, param in model.named_parameters():
        #     self.tb.add_histogram(name, param, self.epoch_id)
        #     self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_id)

        torch.save(
            model.state_dict(),
            f"{self.checkpoint_path}/"
            f"seq_epoch_{self.epoch_id}_prune_iteration_{self.prune_ite}.pth.tar"
        )
        if self.val_accuracy > self.best_val_accuracy:
            torch.save(
                model.state_dict(),
                f"{self.checkpoint_path}/"
                f"best_prune_iteration_{self.prune_ite}.pth.tar"
            )
            print(f"\n Old best val accuracy: {self.best_val_accuracy} || "
                  f"New best val accuracy: {self.val_accuracy} , and new model saved..\n")
            self.best_val_accuracy = self.val_accuracy

    def track_train_loss(self, loss):
        """
        Calculates the loss at the each iteration of batch.

        :param loss:

        :return: calculated loss
        """
        self.epoch_train_loss += loss * self.train_loader.batch_size

    def track_total_train_correct_per_epoch(self, preds, labels, num_classes):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        self.epoch_id_total_train_correct += get_correct(preds, labels, num_classes)

    def track_val_loss(self, loss):
        """
        Calculates the loss at the each iteration of batch.

        :param loss:

        :return: calculated loss
        """
        self.epoch_val_loss += loss * self.val_loader.batch_size

    def track_total_val_correct_per_epoch(self, preds, labels, num_labels):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        self.epoch_id_total_val_correct += get_correct(preds, labels, num_labels)

    def get_final_val_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.val_loss

    def get_final_train_loss(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.train_loss

    def get_final_best_val_accuracy(self):
        """
        Gets the final loss value.

        :return: the final loss value
        """
        return self.best_val_accuracy

    def get_final_val_accuracy(self):
        """
        Gets the final accuracy value.

        :return: the final accuracy value
        """
        return self.val_accuracy

    def get_final_train_accuracy(self):
        """
        Gets the final accuracy value.

        :return: the final accuracy value
        """
        return self.train_accuracy

    def get_epoch_duration(self):
        return self.epoch_duration
