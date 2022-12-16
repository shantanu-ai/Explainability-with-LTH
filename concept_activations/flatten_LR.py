import torch


class Flatten_LR(torch.nn.Module):
    def __init__(self, ip_size, op_size):
        super(Flatten_LR, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=ip_size, out_features=op_size, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    x = torch.rand(1, 80176)
    model = Flatten_LR(80176, 108)

    print(model.parameters())
    for name, params in model.named_parameters():
        print(name, params)
        print(params.size())
