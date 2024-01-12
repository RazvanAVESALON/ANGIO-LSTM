from comet_ml import Experiment
import pandas as pd
import pathlib as pt
import yaml
import torch
import torchvision
import torchvision.transforms as T
import monai.transforms as TR
import torchmetrics
from datetime import datetime
from angio_class import AngioClass
from torchsummary import summary
#from cnn_lstm import CNNLSTM
import albumentations as A
from lighting_train import LitAngio
import lightning as L
from lightning.pytorch.tuner import Tuner
import pytorch_lightning as pl 
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt 
from cnn_lstm_rshp import CNNLSTM

def main():
    print(f"pyTorch version {torch.__version__}")
    print(f"torchvision version {torchvision.__version__}")
    print(f"torchmetrics version {torchmetrics.__version__}")
    print(f"CUDA available {torch.cuda.is_available()}")

    config = None
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    experiment = Experiment(
        api_key="wwQKu3dl9l1bRZOpeKs0y3r8S",
        project_name="general",
        workspace="razvanavesalon",
    )

    exp_name = f"Experiment_Dice_index{datetime.now().strftime('%m%d%Y_%H%M')}"

    exp_path = "D:\\Angio\\ANGIO-LSTM\\Experimente"
    exp_path = pt.Path(exp_path, exp_name)
    exp_path.mkdir(exist_ok=True)
    dir = "Weights"
    path = pt.Path(exp_path) / dir
    path.mkdir(exist_ok=True)
    ok = True
    network = CNNLSTM(num_classes=2)
    summary(network)
    experiment.log_parameters(config)

    yml_data = yaml.dump(config)
    f = open(f"{path}\\yaml_config.yml", "w+")
    f.write(yml_data)
    f.close()

    pixel_t=A.Compose([
        A.CLAHE(clip_limit=config['train']['clip_limit'], tile_grid_size=config['train']['tile_grid_size'], always_apply=False, p=config['train']['p_clahe']), 
        A.RandomGamma(gamma_limit=config['train']['gamma_limit'], eps=None, always_apply=False, p=config['train']['p']),
        A.OneOf([
        A.GaussianBlur(blur_limit=config['train']['blur_limit'], sigma_limit=config['train']['sigma_limit'], always_apply=False, p=config['train']['p_gauss_blur']), 
        A.MotionBlur (blur_limit=config['train']['b_limit'], p=config['train']['p']),
        A.MedianBlur (blur_limit=config['train']['b_limit'], p=config['train']['p']),
        ], p=1),
        ],keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))


    geometric_t= A.Compose([
        A.OneOf([
        A.CoarseDropout (max_holes=config['train']['coarse_max_holes'], max_height=config['train']['coarse_max_height'], max_width=config['train']['coarse_max_width'], fill_value=config['train']['fill_value'],p=config['train']['p']),
        A.PixelDropout (dropout_prob=0.01, drop_value=config['train']['fill_value'], p=config['train']['p'])
        ], p=1),
        A.Rotate(limit=config['train']['rotate_range']),
        A.Resize(height=config['data']['img_size'][0] , width=config['data']['img_size'][1])
        
    ],keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))

    geometric_resize= A.Compose([
        A.Resize(height=config['data']['img_size'][0] , width=config['data']['img_size'][1])
    ],keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))
    dataset_df = pd.read_csv(config["data"]["dataset_csv"])

    train_df = dataset_df.loc[dataset_df["subset"] == "train", :]
    train_ds = AngioClass(
        train_df, img_size=config["data"]["img_size"],nr_frm=config['data']['nr_frames'], geometrics_transforms=geometric_t
        ,pixel_transforms=pixel_t
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["train"]["bs"], shuffle=True, drop_last=True
    )

    valid_df = dataset_df.loc[dataset_df["subset"] == "valid", :]
    print(valid_df)
    valid_ds = AngioClass(
        valid_df, img_size=config["data"]["img_size"],nr_frm=config['data']['nr_frames'], geometrics_transforms=geometric_resize
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=config["train"]["bs"], shuffle=False, drop_last=True
    )
    if config["train"]["opt"] == "Adam":
        opt = torch.optim.Adam(network.parameters(), lr=config["train"]["lr"])
    elif config["train"]["opt"] == "SGD":
        opt = torch.optim.SGD(network.parameters(), lr=config["train"]["lr"])
    elif config["train"]["opt"] == "RMSprop":
        opt = torch.optim.RMSprop(network.parameters(), lr=config["train"]["lr"])


    checkpoint_callback = ModelCheckpoint(dirpath=path, save_top_k=50, monitor="val_loss")
    
    
    trainer = L.Trainer(callbacks=[checkpoint_callback],max_epochs=config['train']['epochs'], accelerator="gpu",)
  
    model= LitAngio(network, config["train"]["opt"], config["train"]["lr"], experiment)
 
    tuner=Tuner(trainer)
    lr_finder=tuner.lr_find(model,train_dataloaders=train_loader,val_dataloaders=valid_loader)
    print(lr_finder.results)
    path=pt.Path(path)/'lr_find.png'
    fig= lr_finder.plot(suggest=True)
    plt.savefig(path)
    #Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    #update hparams of the model
    model.hparams.lr = new_lr
    trainer.fit(
        model,
        train_loader,
        valid_loader,
        ckpt_path=r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index01032024_1429\Weights\epoch=9-step=3820.ckpt"
        
        
    )
    #trainer.test(LitAngio(network, config["train"]["opt"], config["train"]["lr"], experiment), dataloaders=test_loader)

if __name__ == "__main__":
    main()
