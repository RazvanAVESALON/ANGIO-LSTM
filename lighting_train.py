import os, torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing import Any
import lightning as L
from cnn_lstm import CNNLSTM

network = CNNLSTM(num_classes=2) 


class LitAutoEncoder(L.LightningModule):
    def __init__(self, network):
        super().__init__()
        self.network= network

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, train_batch, batch_idx):
      

dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())

trainer = L.Trainer(max_steps=1000)
trainer.fit(LitAutoEncoder(encoder, decoder), data.DataLoader(dataset, batch_size=64))