import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from ImageNet100 import ImageNet100
from tqdm import tqdm

def get_mnist_dataloaders():
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
    )
    batch_size = 512
    ds_train = MNIST("./data/mnist", download=True, train=True, transform=transform)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)

    ds_test = MNIST("./data/mnist", download=True, train=False, transform=transform)
    dl_test = DataLoader(ds_test, batch_size=batch_size, num_workers=4)
    return dl_train, dl_test

def get_imagenet_dataloaders(data_folder='/home/dl_class/data/ILSVRC/Data/CLS-LOC/'):
    train_dataset = ImageNet100(data_folder, split="train", remap_labels=True, transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]))
    test_dataset = ImageNet100(data_folder, split="val", remap_labels=True, transform=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=8, shuffle=False)
    return train_dataloader, test_dataloader

def get_preds(classifier, dataloader, transform=None):
    Y = []
    Y_pred = []
    for x, y in dataloader:
        if transform is not None:
            x = transform(x, y)
        x = x.to(classifier.device)
        y_pred = classifier(x)
        label_pred = torch.argmax(y_pred, dim=1)
        Y.extend(y)
        Y_pred.extend(label_pred)
    Y = list(map(lambda y : y.item(), Y))
    Y_pred = list(map(lambda y : y.item(), Y_pred))
    return Y, Y_pred