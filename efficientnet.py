import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import optim

from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4

class EfficientNetClassifier(pl.LightningModule):
    def __init__(self, model_save_path="efficientnet_b4.pt", num_classes=6, lr=0.001):
        super(EfficientNetClassifier, self).__init__()
        model = efficientnet_b4(
            pretrained=True,
            path=model_save_path,
            num_classes=num_classes
        )
        self.mdl = model

        # Hyperparameters
        self.lr = lr
    
    def on_train_start(self):
        self.mdl = self.mdl.to(self.device)

    def forward(self, x):
        return self.mdl(x)

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        return [[optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]]

    def training_step(self, batch, batch_nb):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        self.log("loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        val_loss = F.cross_entropy(self(x.float()), y)
        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss}

