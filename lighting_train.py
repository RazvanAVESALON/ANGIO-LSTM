import torch
import torch.nn.functional as F
import lightning as L


class LitAutoEncoder(L.LightningModule):
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
        loss = F.mse_loss(output, targets)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, valid_batch, batch_idx):
        inputs, target, idx = valid_batch
        output = self.network(inputs)
        output = output.to("cuda")
        loss_val = F.mse_loss(output, target)
        print(output.shape, target.shape)
        self.log("val_loss", loss_val)
        return loss_val
