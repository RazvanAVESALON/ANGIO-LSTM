import os, torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing import Any
import lightning as L
from cnn_lstm import CNNLSTM
from torchmetrics import MeanSquaredError
class LitAutoEncoder(L.LightningModule):
    def __init__(self, network):
        super().__init__()
        self.network= network
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, train_batch, batch_idx):
        
        inputs, targets, idx = train_batch
        output= self.network(inputs)
        loss =  nn.MSELoss(output,targets)
        metric = MeanSquaredError()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_MSE:",metric,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.network(inputs, target)
        loss = nn.MSELoss()
        metric = MeanSquaredError()
        self.log("val_loss", loss)
        self.log("val_MSE", metric) 
    
   
           
