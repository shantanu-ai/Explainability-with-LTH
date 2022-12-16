import torch
from torch.utils.data import Dataset, DataLoader

from dataset.dataset_utils import get_dataset_with_image_and_attributes, get_transform_cub
from model_factory.models import Classifier


class Dataset_cub(Dataset):
    def __init__(self, dataset, attributes, transform=None, show_image=False):
        self.dataset = dataset
        self.show_image = show_image
        self.transform = transform
        self.attributes = attributes

    def __getitem__(self, item):
        image = self.dataset[item][0]
        label = self.dataset[item][1]
        attributes = self.attributes[item]
        if self.transform:
            image = self.transform(image)
        return image, label, attributes

    def __len__(self):
        return len(self.dataset)


def test_dataset():
    print("CUB")
    img_size = 224
    train_set, attributes = get_dataset_with_image_and_attributes(
        data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/CUB_200_2011",
        json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
        dataset_name="cub",
        mode="train",
        attribute_file="attributes.npy"
    )

    transform = get_transform_cub(size=img_size, data_augmentation=True)
    dataset = Dataset_cub(train_set, attributes, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    data = iter(dataloader)
    d = next(data)
    print(d[0].size())
    print(d[1])
    print(d[2])

    model = Classifier("Resnet_50", n_classes=200, dataset_name="cub", pretrained=True)

    # model = Classifier("AlexNet", 1, False)
    print(type(d[0]))
    print("-----------")
    output = model(d[0])
    _, pred = torch.max(output, 1)
    print(pred.size())
    print("=======")
    print(d[1])
    print(pred)
    print(torch.sum(pred == d[1].detach_()))
    print(d[1].size())

    print(model)
    # fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    # ax.imshow(np.transpose(d[0].numpy()[0], (1, 2, 0)), cmap='gray')
    # ax.text(10, 0, f"Label: {d[1][0]}", style='normal', size='xx-large')
    # plt.show()

    val_set, attributes = get_dataset_with_image_and_attributes(
        data_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/CUB_200_2011",
        json_root="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data",
        dataset_name="cub",
        mode="val",
        attribute_file="attributes.npy"
    )
    val_transform = get_transform_cub(size=img_size, data_augmentation=False)
    dataset = Dataset_cub(val_set, attributes, val_transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    data = iter(dataloader)
    d = next(data)
    print(d[0].size())
    print(d[1])
    print(d[2])

    model = Classifier("Resnet_50", n_classes=200, dataset_name="cub", pretrained=True)
    # model = Classifier("AlexNet", 1, False)
    print(type(d[0]))
    print("-----------")
    output = model(d[0])
    _, pred = torch.max(output, 1)
    print(pred.size())
    print("=======")
    print(d[1])
    print(pred)
    print(torch.sum(pred == d[1].detach_()))
    print(d[1].size())


if __name__ == '__main__':
    test_dataset()
