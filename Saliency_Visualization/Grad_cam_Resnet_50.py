import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import torch

def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)


class GradCamModel(nn.Module):
    def __init__(self, chkpt_file, dataset_name, n_classes):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        # PRETRAINED MODEL
        self.model = models.resnet50(pretrained=True)
        feat_dim = self.model.fc.weight.shape[1]
        if dataset_name == "mnist":
            self.model.fc = nn.Sequential(
                nn.Linear(
                    in_features=feat_dim, out_features=n_classes,
                    bias=True
                ),
                nn.Sigmoid()
            )
        elif dataset_name == "cub":
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.model.fc = nn.Linear(in_features=feat_dim, out_features=n_classes)
            self.model.fc.apply(weight_init_kaiming)

        self.layerhook.append(self.model.layer4.register_forward_hook(self.forward_hook()))

        for p in self.model.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        out = self. model(x)
        return out, self.selected_out
