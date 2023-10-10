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
from comet_ml import Experiment
from torchsummary import summary 
from cnn_lstm import CNNLSTM
import albumentations as A 
from lighting_train import LitAutoEncoder
import lightning as L
#from cnn_lstm_rshp import CNNLSTM
def main():
    print(f"pyTorch version {torch.__version__}")
    print(f"torchvision version {torchvision.__version__}")
    print(f"torchmetrics version {torchmetrics.__version__}")
    print(f"CUDA available {torch.cuda.is_available()}")
    
    config = None
    with open('config.yaml') as f: 
        config = yaml.safe_load(f)
    
    experiment = Experiment(
        api_key="wwQKu3dl9l1bRZOpeKs0y3r8S",
        project_name="general",
        workspace="razvanavesalon",)

    exp_name = f"Experiment_Dice_index{datetime.now().strftime('%m%d%Y_%H%M')}"

    exp_path="D:\\Angio\\ANGIO-LSTM\\Experimente"
    exp_path = pt.Path(exp_path,exp_name)
    exp_path.mkdir(exist_ok=True)
    dir = "Weights"
    path = pt.Path(exp_path)/dir
    path.mkdir(exist_ok=True)

    network=CNNLSTM(num_classes=2)
    summary(network)
    experiment.log_parameters(config)

    yml_data = yaml.dump(config)
    f = open(f"{path}\\yaml_config.yml", "w+")
    f.write(yml_data)
    f.close()

    pixels = T.Compose([

        TR.ToTensord(keys="img"),
    ])
    geometric_t= A.Compose([
        A.Resize(height=config['data']['img_size'][0] , width=config['data']['img_size'][1])
    ],keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))

    dataset_df = pd.read_csv(config['data']['dataset_csv'])

    train_df = dataset_df.loc[dataset_df["subset"] == "train",:]
    train_ds = AngioClass(train_df, img_size=config['data']['img_size'],geometrics_transforms=geometric_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['train']['bs'], shuffle=True,drop_last=True)
    
    valid_df = dataset_df.loc[dataset_df["subset"] == "valid", :]
    print(valid_df)
    valid_ds = AngioClass(valid_df, img_size=config['data']['img_size'],geometrics_transforms=geometric_t)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=config['train']['bs'],shuffle=False,drop_last=True)
    if config['train']['opt'] == 'Adam':
        opt = torch.optim.Adam(network.parameters(), lr=config['train']['lr'])
    elif config['train']['opt'] == 'SGD':
        opt = torch.optim.SGD(network.parameters(), lr=config['train']['lr'])
    elif config['train']['opt'] == "RMSprop":
        opt = torch.optim.RMSprop(
            network.parameters(), lr=config['train']['lr'])
    
  
    
    trainer = L.Trainer(max_steps=100,accelerator='gpu')
    trainer.fit(LitAutoEncoder(network,config['train']['opt'],config['train']['lr']), train_loader,valid_loader)
   

if __name__ == "__main__":
    main()

