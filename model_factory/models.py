import torch
import torch.nn.init as init
import torchvision
from torch import nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)


AlexNet = "AlexNet"
RESNET10 = "Resnet_10"
RESNET18 = "Resnet_18"
RESNET34 = "Resnet_34"
RESNET50 = "Resnet_50"
RESNET101 = "Resnet_101"


def ResNet10(pretrained=False):
    assert pretrained == False, "No pretrained weights available for ResNet10"
    return ResNet(BasicBlock, [1, 1, 1, 1])


CNN_MODELS = {
    RESNET10: ResNet10,
    AlexNet: torchvision.models.AlexNet,
    RESNET18: torchvision.models.resnet18,
    RESNET34: torchvision.models.resnet34,
    RESNET50: torchvision.models.resnet50,
    RESNET101: torchvision.models.resnet101
}


class Classifier(nn.Module):
    def __init__(
            self,
            model_name,
            n_classes,
            dataset_name,
            pretrained=True,
            transfer_learning=False
    ):
        super(Classifier, self).__init__()
        model_dict = CNN_MODELS[model_name]
        if model_name == "AlexNet":
            self.model = model_dict()
            feat_dim = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(
                    in_features=feat_dim,
                    out_features=n_classes,
                    bias=True
                ),
                nn.Sigmoid()
            )
        elif model_name == "Resnet_10" or model_name == "Resnet_18" or model_name == "Resnet_34" \
                or model_name == "Resnet_50" or model_name == "Resnet_101":
            self.model = model_dict(pretrained=pretrained)
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

        if transfer_learning:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x


def test_model():
    img = torch.rand(1, 3, 448, 448)
    # img = torch.rand(1, 3, 28, 28)
    model = Classifier("Resnet_50", 200, "cub", False)
    # model = Classifier("AlexNet", 1, False)
    pred = model(img)
    print(model)
    print(pred)
    print(pred.size())


if __name__ == '__main__':
    test_model()
