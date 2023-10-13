import torch.nn as nn
import torch
import lightning as L

# from torcheval.metrics import MeanSquaredError
from torchmetrics import MeanSquaredError


class LitAngio(L.LightningModule):
    def __init__(self, network, opt_ch, lr):
        super().__init__()
        self.network = network
        self.opt_ch = opt_ch
        self.lr = lr

    def configure_optimizers(self):
        if self.opt_ch == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.opt_c == "SGD":
            opt = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.opt_c == "RMSprop":
            opt = torch.optim.RMSprop(self.parameters(), lr=self.lr)

        return opt

    def training_step(self, train_batch, batch_idx):
        inputs, targets, idx = train_batch
        output = self.network(inputs)
        output = output.to("cuda")
        print(output.type, targets.type)
        criterion = nn.MSELoss().to("cuda")
        loss = criterion(output, targets)
        mse_metric = MeanSquaredError().to("cuda")
        mse_train = mse_metric(output, targets)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )
        self.log(
            "train_mse",
            mse_train,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, valid_batch, batch_idx):
        inputs, target, idx = valid_batch
        output = self.network(inputs)
        output = output.to("cuda")
        print(output.shape, target.shape)
        criterion = nn.MSELoss().to("cuda")
        loss_val = criterion(output, target)
        mse_metric = MeanSquaredError().to("cuda")
        mse_val = mse_metric(output, target)
        print(output.shape, target.shape)
        self.log("val_loss", loss_val)
        self.log("mse_loss", mse_val)
        return loss_val
