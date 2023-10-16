import torch.nn as nn
import torch
import lightning as L

# from torcheval.metrics import MeanSquaredError
from torchmetrics import MeanSquaredError


class LitAngio(L.LightningModule):
    def __init__(self,hparams,network=None, opt_ch=None, lr=None,experiment=None):
        super().__init__()
        self.network = network
        self.opt_ch = opt_ch
        self.lr = lr
        self.experimet=experiment
        self.hparams = hparams

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
        self.experimet.log_metrics({f"Train_MSE": mse_train,
                        f"Train_loss": loss}, epoch=self.current_epoch)
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
    
    def test_step(self, test_batch, batch_idx):
        # this is the test loop
        inputs, targets, idx = test_batch
        self.network.eval()
        output = self.network(inputs)
        output = output.to("cuda")
        print(output.shape, targets.shape)
        criterion = nn.MSELoss().to("cuda")
        loss_test = criterion(output, targets)
        mse_metric = MeanSquaredError().to("cuda")
        mse_test = mse_metric(output, targets)
        self.log("test_loss", loss_test)
        self.log("self_loss", mse_test)
        return loss_test
