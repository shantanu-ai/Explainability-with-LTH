import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset.dataset_utils import get_transforms, get_dataset_with_attributes
from model_factory.models import Classifier


class Dataset_attributes_mnist(Dataset):
    def __init__(self, dataset, attributes, transform=None, show_image=False):
        self.dataset = dataset
        self.show_image = show_image
        self.transform = transform
        self.attributes = attributes

    def __getitem__(self, item):
        image = self.dataset[item][0]
        attributes = self.attributes[item]
        if self.transform:
            image = self.transform(image)
        return image, attributes

    def __len__(self):
        return len(self.dataset)


def test_dataset():
    print("MNIST")
    size = 224
    train_transform = get_transforms(size=size)
    train_set, train_attributes = get_dataset_with_attributes(
        data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/MNIST_EVEN_ODD",
        json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
        dataset_name="mnist",
        mode="train",
        attribute_file="attributes.npy"
    )

    dataset = Dataset_attributes_mnist(train_set, train_attributes, train_transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"Train set size: {len(dataloader)}")
    data = iter(dataloader)
    image, attribute = next(data)
    print(image.size())
    print(attribute)

    model = Classifier("Resnet_18", 1, False)
    # model = Classifier("AlexNet", 1, False)
    pred = model(image)
    print(pred.size())
    print(attribute.size())

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(np.transpose(image.numpy()[0], (1, 2, 0)), cmap='gray')
    ax.text(
        10,
        0,
        f"Label: {(attribute[0] == 1).nonzero().item()}",
        style='normal',
        size='xx-large'
    )
    plt.show()


if __name__ == '__main__':
    test_dataset()
