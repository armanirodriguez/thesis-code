"""
Armani Rodriguez
"""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from hqa.model import HQA2D as HQA_Original
from hqa.scheduler import FlatCA
from hqa.optimizer import RAdam

from data import get_mnist_dataloaders, get_imagenet_dataloaders


class HQA2D_Lightning(pl.LightningModule):
    def __init__(self, lr=4e-4, *args, **kwargs):
        super(HQA2D_Lightning, self).__init__()
        self.hqa_model = HQA_Original(*args, **kwargs)
        self.lr = lr

        self.temp_base = self.hqa_model.codebook.temperature

        # Tells pytorch lightinig to use our custom training loop
        self.automatic_optimization = False

    def on_train_start(self):
        # Register a buffer to track codeword usage
        self.register_buffer(
            "code_count",
            torch.zeros(
                self.hqa_model.codebook.codebook_slots,
                device=self.device,
                dtype=torch.float64,
            )
        )
        self.total_steps = self.trainer.max_epochs * self.trainer.num_training_batches

    def training_step(self, batch, batch_ndx):
        x, _ = batch

        # anneal temperature
        self.hqa_model.codebook.temperature = self._decay_temp_linear(
            self.global_step + 1, self.total_steps, self.temp_base, temp_min=0.001
        )

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        recon, original, z_q, z_e, indices, KL, commit_loss = self.hqa_model(x)
        recon_loss = self.hqa_model.recon_loss(recon, original)

        # calculate loss
        dims = np.prod(recon.shape[1:])  # orig_w * orig_h * num_channels
        loss = recon_loss / dims + 0.001 * KL / dims + 0.001 * (commit_loss) / dims

        optimizer.zero_grad()

        self.manual_backward(loss)

        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Code Reset
        indices_onehot = F.one_hot(
            indices, num_classes=self.hqa_model.codebook.codebook_slots
        ).float()
        self.code_count = self.code_count + indices_onehot.sum(dim=(0, 1, 2))
        if self.global_step % 20 == 0:
            self._reset_codeword_if_needed()
               

        self.log_dict(
            {"loss": loss, "kl": KL, "recon": recon_loss, "commit": commit_loss},
            prog_bar=True,
        )

    def validation_step(self, batch):
        x, _ = batch

        recon, original, z_q, z_e, indices, KL, commit_loss = self.hqa_model(x)
        recon_loss = self.hqa_model.recon_loss(recon, original)

        # calculate loss
        dims = np.prod(recon.shape[1:])  # orig_w * orig_h * num_channels
        loss = recon_loss / dims + 0.001 * KL / dims + 0.001 * (commit_loss) / dims

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr)
        lr_scheduler = FlatCA(
            optimizer,
            steps=self.trainer.max_epochs * self.trainer.num_training_batches,
            eta_min=4e-5,
        )
        return [optimizer], [lr_scheduler]
    
    @torch.no_grad()
    def _reset_codeword_if_needed(self):
        max_count, most_used_code = torch.max(self.code_count, dim=0)
        frac_usage = self.code_count / max_count
        z_q_most_used = self.hqa_model.codebook.lookup(
            most_used_code.view(1, 1, 1)
        ).squeeze()

        min_frac_usage, min_used_code = torch.min(frac_usage, dim=0)
        if min_frac_usage < 0.03:
            #print(f"reset code {min_used_code}")
            moved_code = z_q_most_used + torch.randn_like(z_q_most_used) / 100
            self.on_codeword_reset(min_used_code, moved_code)
            self.hqa_model.codebook.codebook[min_used_code] = moved_code
        self.code_count = torch.zeros_like(self.code_count)

    def _decay_temp_linear(cls, step, total_steps, temp_base, temp_min=0.001):
        factor = 1.0 - (step / total_steps)
        return temp_min + (temp_base - temp_min) * factor

    @classmethod
    def init_higher(cls, prev_model, **kwargs):
        model = HQA2D_Lightning(
            lr = prev_model.lr,
            prev_model=prev_model.hqa_model,
            input_feat_dim=prev_model.hqa_model.codebook.codebook_dim,
            **kwargs,
        )
        prev_model.eval()
        return model
        
        

    @classmethod
    def init_bottom(cls, input_feat_dim, **kwargs):
        return HQA2D_Lightning(
            prev_model=None, 
            input_feat_dim=input_feat_dim, 
            **kwargs
        )
    
    @classmethod
    def train_full_stack(cls,
                        train_dataloader,
                        val_dataloader,
                        input_feat_dim,
                        configs,
                        trainer_args={
                            'max_epochs':20
                        }):
        n_layers = len(configs)
        for i in range(n_layers):
            if i == 0:
                hqa = HQA2D_Lightning.init_bottom(input_feat_dim=input_feat_dim,**configs[i])
            else:
                hqa = HQA2D_Lightning.init_higher(hqa_prev, **configs[i])
            trainer = pl.Trainer(**trainer_args, 
                                 strategy=pl.strategies.DDPStrategy(find_unused_parameters=i > 0)
                                )
            trainer.fit(model=hqa, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            hqa.eval()
            hqa_prev = hqa
        return hqa  
    
    def on_codeword_reset(self, moved_index, new_codeword):
        pass

    def encode(self, x):
        return self.hqa_model.encode(x)

    def decode(self, z_q):
        return self.hqa_model.decode(z_q)
        
    def reconstruct(self, x):
        return self.hqa_model.reconstruct(x)

    def __len__(self):
        return len(self.hqa_model)

    def __getitem__(self, index):
        return self.hqa_model[index]

    def parameters(self):
        return self.hqa_model.parameters()


def train_hqa_mnist(**trainer_args):
    torch.set_float32_matmul_precision("medium")
    
    dl_train, dl_test = get_mnist_dataloaders()
    
    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    
    configs = [
         {
            'enc_hidden_dim':enc_hiden_size,
            'dec_hidden_dim':dec_hidden_size,
            'codebook_slots':256,
        } for enc_hiden_size, dec_hidden_size in zip(enc_hidden_sizes, dec_hidden_sizes)
    ]
    
    
    return HQA2D_Lightning.train_full_stack(
        dl_train,
        dl_test,
        configs,
        trainer_args
    )

def train_hqa_imagenet100(**trainer_args):
    torch.set_float32_matmul_precision("medium")
    
    dl_train, dl_test = get_imagenet_dataloaders()
    
    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    
    configs = [
         {
            'enc_hidden_dim':enc_hiden_size,
            'dec_hidden_dim':dec_hidden_size,
            'codebook_slots':512,
        } for enc_hiden_size, dec_hidden_size in zip(enc_hidden_sizes, dec_hidden_sizes)
    ]
    
    
    return HQA2D_Lightning.train_full_stack(
        dl_train,
        dl_test,
        3,
        configs,
        trainer_args
    )


if __name__ == "__main__":
    model = train_hqa_imagenet100(max_epochs=50, num_sanity_val_steps=0)
    torch.save(model, "checkpoints/hqa_imagenet_512codebook.pt")
