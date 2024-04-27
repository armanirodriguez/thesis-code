import lightning.pytorch as pl

import torch
from torch import nn, optim
from torch.nn import init
import torch.nn.functional as F

from tqdm import tqdm

from data import get_mnist_dataloaders

class LeNet5(pl.LightningModule):
    def __init__(self, lr=1e-2, num_classes = 10):
        super(LeNet5, self).__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding = 2, stride=1)    
        conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding = 0, stride=1)
        conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding = 0, stride=1)
        fc1 = nn.Linear(120, out_features=84)
        fc2 = nn.Linear(84, out_features=num_classes)
        
        init.xavier_normal_(conv1.weight)
        init.xavier_normal_(conv2.weight)
        init.xavier_normal_(conv3.weight)
        init.xavier_normal_(fc1.weight)
        init.xavier_normal_(fc2.weight)
        init.zeros_(conv1.bias)
        init.zeros_(conv2.bias) 
        init.zeros_(conv3.bias)
        init.zeros_(fc1.bias)
        init.zeros_(fc2.bias)

        self.cnn = nn.Sequential(
            conv1,
            nn.GELU(), 
            nn.MaxPool2d(kernel_size=2, stride = 2),
            conv2,
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            conv3,
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride = 1)
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            fc1,
            nn.GELU(), 
            nn.Dropout(),
            fc2
        )

        self.lr = lr
    
    def forward(self, x):
        feature_map = self.cnn(x)
        y_pred = self.mlp(feature_map)
        return y_pred

    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred ,y)
        self.log("loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def train_lenet_mnist(**trainer_args):
    torch.set_float32_matmul_precision("medium")
    dl_train, dl_test = get_mnist_dataloaders()
    model = LeNet5()
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, dl_train, dl_test)
    return model

def get_preds(model, dataloader, transform=None):
    Y = []
    Y_pred = []
    for x, y in tqdm(dataloader):
        if transform is not None:
            x = transform(x, y)
        x = x.to(model.device)
        y_pred = model(x)
        label_pred = torch.argmax(y_pred, dim=1)
        Y.extend(y)
        Y_pred.extend(label_pred)
    Y = list(map(lambda y : y.item(), Y))
    Y_pred = list(map(lambda y : y.item(), Y_pred))
    return Y, Y_pred
    
if __name__ == '__main__':
    model = train_lenet_mnist(max_epochs=20)
    torch.save(model, "./checkpoints/lenet_mnist.pt")
    
        