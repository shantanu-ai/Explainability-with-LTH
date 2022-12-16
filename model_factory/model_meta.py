import torch


class Model_Meta:
    def __init__(self, model, layers):
        self.grads = None
        self.model = model
        self.model_activations_store = {}

        def save_activation(layer):
            def hook(module, input, output):
                self.model_activations_store[layer] = output

            return hook

        for layer in layers:
            # get the activations for 3rd Dense block
            if layer == "layer2":
                module = model.model.layer2
                module.register_forward_hook(save_activation(layer))

            elif layer == "layer3":
                module = model.model.layer3
                module.register_forward_hook(save_activation(layer))

            elif layer == "layer4":
                module = model.model.layer4
                module.register_forward_hook(save_activation(layer))

    def get_gradients(self, grad):
        self.grads = grad

    def calculate_grads(self, class_name, layer, output_probs):
        activation = self.model_activations_store[layer]
        activation.register_hook(self.get_gradients)
        logit = output_probs
        logit.backward(torch.ones_like(logit), retain_graph=True)

        gradients = self.grads.cpu().detach().numpy()
        return gradients

    def estimate_grads_binary_classification(self, activation, bb_mid, bb_tail, device):
        torch_acts = torch.unsqueeze(
            torch.autograd.Variable(torch.from_numpy(activation).to(device), requires_grad=True),
            dim=0)
        prob_mid = bb_mid(torch_acts)
        bs, ch, h, w = prob_mid.size()
        prob_mid = prob_mid.reshape(bs, ch * h * w)
        outputs = bb_tail(prob_mid)
        prob_1 = outputs
        prob_0 = 1 - outputs
        grads_1 = torch.autograd.grad(prob_1, torch_acts, retain_graph=True)[0]
        grads_0 = torch.autograd.grad(prob_0, torch_acts, retain_graph=True)[0]

        return grads_0.cpu().detach().numpy(), grads_1.cpu().detach().numpy()

    def estimate_grads_for_binary_classification(self, logits, bb_layer):
        activation = self.model_activations_store[bb_layer]
        grads = torch.autograd.grad(logits, activation, retain_graph=True)[0]
        return grads.cpu().detach().numpy()

    def estimate_grads_multiclass_classification(self, class_label_index, bb_layer, output_prob, device):
        activation = self.model_activations_store[bb_layer]
        logit = output_prob[:, class_label_index]
        grads = torch.autograd.grad(logit, activation, retain_graph=True)[0]
        return grads.cpu().detach().numpy()
