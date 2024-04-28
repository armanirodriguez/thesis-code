"""
A PyTorch lightning wrapper for fine tuning ResNet50

Author: Armani Rodriguez
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import lightning.pytorch as pl

from data import get_imagenet_dataloaders

class ResNet50(pl.LightningModule):
    def __init__(self, lr=1e-2, num_classes=100):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Replace final layer such that it has num_classes outputs
        self.resnet.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.lr = lr
    
    def forward(self, x):
        return self.resnet(x)
    
    def training_step(self, batch):
        x, y = batch
        y_pred = self.resnet(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_pred = self.resnet(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            self.trainer.max_epochs * self.trainer.num_training_batches, 
            eta_min=self.lr * 0.01
        )
        return [optimizer], [scheduler]

def train_reset_imagenet(**trainer_args):
    torch.set_float32_matmul_precision("medium")
    dl_train, dl_test = get_imagenet_dataloaders()
    model = ResNet50()
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, dl_train, dl_test)
    return model
        
if __name__ == '__main__':
    model = train_reset_imagenet(max_epochs=10)
    torch.save(model, "./checkpoints/resnet_imagenet.pt")