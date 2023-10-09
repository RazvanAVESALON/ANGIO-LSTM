import pandas as pd
import pathlib as pt
import yaml
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import monai.transforms as TR
import torchmetrics
from tqdm import tqdm
from datetime import datetime
from angio_class import AngioClass
from torchmetrics import MeanSquaredError
from comet_ml import Experiment
from torchsummary import summary 
from cnn_lstm import CNNLSTM
import albumentations as A 
from lighting_train import LitAutoEncoder
import lightning as L
from lightning.pytorch.accelerators import find_usable_cuda_devices
#from cnn_lstm_rshp import CNNLSTM
import torch.nn as nn 
def train(network, train_loader, valid_loader, exp, criterion, opt, epochs, thresh=0.5, weights_dir='weights', save_every_ep=50):

    total_loss = {'train': [], 'valid': []}
    total_dice = {'train': [], 'valid': []}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting training on device {device} ...")

    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    metric = MeanSquaredError()
    network.to(device)
    criterion.to(device)
    for ep in range(epochs):

        print(f"[INFO] Epoch {ep}/{epochs - 1}")

        print("-" * 20)
        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_average = 0.0

            if phase == 'train':
                network.train()
            else:
                network.eval()

            with tqdm(desc=phase, unit=' batch', total=len(loaders[phase].dataset)) as pbar:
                for data in loaders[phase]:
                    ins, tgs, idx = data
                    
                    print ('Input shape :',ins.shape, 'BF Point:' , tgs.shape)
                    ins = ins.to(device)
                    tgs = tgs.to(device)
                    opt.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        
                        output = network(ins)
                        output = output.to(device)
                        print (output.shape , tgs.squeeze())
                        loss = criterion(output, tgs.squeeze())
                        if 'cuda' in device.type:
                            output = output.cpu()
                            tgs = tgs.cpu().type(torch.int).squeeze()
                        else:
                            tgs = tgs.type(torch.int).squeeze()

                        mse = metric(output, tgs)

                        if phase == 'train':
                            loss.backward()
                            opt.step()

                    running_loss += loss.item() * ins.size(0)

                    running_average += mse.item() * ins.size(0)

                    if phase == 'valid':
                        if ep % save_every_ep == 0:
                            torch.save(
                                network, f"{weights_dir}/my_model{datetime.now().strftime('%m%d%Y_%H%M')}_e{ep}.pt")

                    pbar.update(ins.shape[0])


                total_loss[phase].append(
                    running_loss/len(loaders[phase].dataset))

                loss_value = running_loss/len(loaders[phase].dataset)
                mse_value = running_average/len(loaders[phase].dataset)

                total_dice[phase].append(
                    running_average/len(loaders[phase].dataset))

                postfix = f'error {total_loss[phase][-1]:.4f} MSE {mse*100:.2f}%'
                pbar.set_postfix_str(postfix)

                exp.log_metrics({f"{phase}MSE": mse_value,
                                f"{phase}loss": loss_value}, epoch=ep)



    return {'loss': total_loss, 'MSE': total_dice}

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
    
    
    # print (train_loader)
    # for data in train_loader:
        
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
    #     inputs , targets , idx = data 
        
    #     inputs = inputs.to(device)
    #     targets = targets.to(device)
    #     print ("Input:",inputs.shape,"Target:",targets.shape) 
    #     output = network(inputs)
    #     print("outpu:",output.shape)
    # for i, data in enumerate(train_loader):
    #     ins, tgs, idx = data
    #     print (f'{i} Input shape :',ins.shape, 'BF Point:' , tgs.shape)
    #     fig, ax = plt.subplots(1, 12, figsize=(20, 5))
    #     for j in range(ins.shape[1]):
    #         ax[j].imshow(ins[0][j][0], cmap='gray')
    #         ax[j].set_title(f"Frame {j}")
    #         print(f"frame {j}", ins[0][j].min(), ins[0][j].max())
            
    #         print(f"tg point frame {j}", tgs[0][j])
    #         # if (j + 1) % 5 == 0:
    #         #     break
    #     plt.show()
    #     if (i + 1) % 6 == 0:
    #         break
    
    
   

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
   
    # print(f"# Train: {len(train_ds)} # Valid: {len(valid_ds)}")
    # 

    # if config['train']['opt'] == 'Adam':
    #     opt = torch.optim.Adam(network.parameters(), lr=config['train']['lr'])
    # elif config['train']['opt'] == 'SGD':
    #     opt = torch.optim.SGD(network.parameters(), lr=config['train']['lr'])
    # elif config['train']['opt'] == "RMSprop":
    #     opt = torch.optim.RMSprop(
    #         network.parameters(), lr=config['train']['lr'])

    # history = train(network, train_loader, valid_loader, experiment, criterion, opt,
    #                 epochs=config['train']['epochs'], thresh=config['test']['threshold'], weights_dir=path)
    #plot_acc_loss(history, path)

if __name__ == "__main__":
    main()

