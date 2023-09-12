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
from angio_class import AngioClass, plot_acc_loss
from monai.losses import DiceCELoss
from torchmetrics import MeanSquaredError
from comet_ml import Experiment
import segmentation_models_pytorch as smp
from torchsummary import summary 
from torch import nn 
from cnn_lstm import CNNLSTM
import albumentations as A 
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
                network.train()  # Set model to training mode
            else:
                network.eval()   # Set model to evaluate mode

            with tqdm(desc=phase, unit=' batch', total=len(loaders[phase].dataset)) as pbar:
                for data in loaders[phase]:
                    ins, tgs, idx = data
                    ins = ins.to(device)
                    tgs = tgs.to(device)
                    opt.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        
                        output = network(ins)
                        print (output.shape , tgs.squeeze())
                        loss = criterion(output, tgs.squeeze())
                        if 'cuda' in device.type:
                            output = output.cpu()
                            tgs = tgs.cpu().type(torch.int).squeeze()
                        else:
                            tgs = tgs.type(torch.int).squeeze()

                        mse = metric(output, tgs)

                        if phase == 'train':
                            # se face backpropagation -> se calculeaza gradientii
                            loss.backward()
                            # se actualizează weights-urile
                            opt.step()

                    running_loss += loss.item() * ins.size(0)

                    running_average += mse.item() * ins.size(0)

                    if phase == 'valid':
                        # salvam ponderile modelului dupa fiecare epoca
                        if ep % save_every_ep == 0:
                            torch.save(
                                network, f"{weights_dir}/my_model{datetime.now().strftime('%m%d%Y_%H%M')}_e{ep}.pt")

                    pbar.update(ins.shape[0])

                # Calculam loss-ul pt toate batch-urile dintr-o epoca
                total_loss[phase].append(
                    running_loss/len(loaders[phase].dataset))

                loss_value = running_loss/len(loaders[phase].dataset)
                mse_value = running_average/len(loaders[phase].dataset)
                # Calculam acuratetea pt toate batch-urile dintr-o epoca
                total_dice[phase].append(
                    running_average/len(loaders[phase].dataset))

                postfix = f'error {total_loss[phase][-1]:.4f} MSE {mse*100:.2f}%'
                pbar.set_postfix_str(postfix)

                exp.log_metrics({f"{phase}MSE": mse_value,
                                f"{phase}loss": loss_value}, epoch=ep)

                # Resetam pt a acumula valorile dintr-o noua epoca

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
    print(train_df)
    train_ds = AngioClass(train_df, img_size=config['data']['img_size'],geometrics_transforms=geometric_t)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config['train']['bs'], shuffle=True,drop_last=True)

    valid_df = dataset_df.loc[dataset_df["subset"] == "valid", :]
    print(valid_df)
    valid_ds = AngioClass(valid_df, img_size=config['data']['img_size'],geometrics_transforms=geometric_t)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=config['train']['bs'], shuffle=False,drop_last=True)

    print(f"# Train: {len(train_ds)} # Valid: {len(valid_ds)}")
    criterion = nn.MSELoss()

    network=CNNLSTM(num_classes=2)

    if config['train']['opt'] == 'Adam':
        opt = torch.optim.Adam(network.parameters(), lr=config['train']['lr'])
    elif config['train']['opt'] == 'SGD':
        opt = torch.optim.SGD(network.parameters(), lr=config['train']['lr'])
    elif config['train']['opt'] == "RMSprop":
        opt = torch.optim.RMSprop(
            network.parameters(), lr=config['train']['lr'])

    history = train(network, train_loader, valid_loader, experiment, criterion, opt,
                    epochs=config['train']['epochs'], thresh=config['test']['threshold'], weights_dir=path)
    plot_acc_loss(history, path)

if __name__ == "__main__":
    main()
