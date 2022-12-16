import torch


class G(torch.nn.Module):
    def __init__(
            self,
            g_model_ip_size,
            g_model_op_size,
            hidden_features=500
    ):
        super(G, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=g_model_ip_size, out_features=hidden_features, bias=True),
            torch.nn.ReLU(),
            # torch.nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=True),
            # torch.nn.ReLU(),
            # torch.nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=True),
            # torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_features, out_features=g_model_op_size, bias=True),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x
