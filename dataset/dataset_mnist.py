import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset.dataset_utils import get_dataset
from model_factory.models import Classifier


class Dataset_mnist(Dataset):
    def __init__(self, dataset, transform=None, show_image=False):
        self.dataset = dataset
        self.show_image = show_image
        self.transform = transform

    def __getitem__(self, item):
        image = self.dataset[item][0]
        if self.transform:
            image = self.transform(image)
        return image, self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)


def test_dataset():
    print("MNIST")
    train_set = get_dataset(
        data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
        json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
        dataset_name="mnist",
        mode="train"
    )

    # train_transform = get_transforms(size=224)
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    dataset = Dataset_mnist(train_set, transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    data = iter(dataloader)
    d = next(data)
    print(d[0].size())
    print(d[1])

    model = Classifier("Resnet_10", 1, False)
    # model = Classifier("AlexNet", 1, False)
    pred = model(d[0])
    print(pred.size())
    print(d[1].size())

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(np.transpose(d[0].numpy()[0], (1, 2, 0)), cmap='gray')
    ax.text(10, 0, f"Label: {d[1][0].item()}", style='normal', size='xx-large')
    plt.show()


if __name__ == '__main__':
    test_dataset()
