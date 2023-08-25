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


def train(network, train_loader, valid_loader, experiment ,criterion, opt, epochs, thresh=0.5, weights_dir='weights', save_every_ep=10):

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
                    # print(data)
                    #print(type(ins), type(tgs))
                    ins = ins.to(device)
                    tgs = tgs.to(device)
                    #print (ins.size(),tgs.size())

                    # seteaza toti gradientii la zero, deoarece PyTorch acumuleaza valorile lor dupa mai multe backward passes
                    opt.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # se face forward propagation -> se calculeaza predictia
                        # plt.imshow(ins[0][0].cpu(),cmap='gray')
                        # plt.show()
                        output = network(ins)

                        # plt.imshow(output[0][0].cpu().detach().numpy(),cmap='gray')
                        # plt.show()
                        # print(output.size())

                        #second_output = Variable(torch.argmax(output,1).float(),requires_grad=True).cuda()
                        # output[:, 1, :, :] #=> 8 x 128 x 128
                        # tgs => 8 x 1 x 128 x 128
                        # tgs.squeeze() #=> 8 x 128 x 128

                        # se calculeaza eroarea/loss-ul

                        loss = criterion(output[:, :, :, :], tgs)

                        # deoarece reteaua nu include un strat de softmax, predictia finala trebuie calculata manual
                        current_predict = F.softmax(output, dim=1)[
                            :, 1].float()
                        current_predict[current_predict >= thresh] = 1.0
                        current_predict[current_predict < thresh] = 0.0

                        #plt.imshow(current_predict[0].cpu().detach().numpy(), cmap='gray')
                        # plt.show()

                        if 'cuda' in device.type:
                            current_predict = current_predict.cpu()
                            current_target = tgs.cpu().type(torch.int).squeeze()
                        else:
                            current_predict = current_predict
                            current_target = tgs.type(torch.int).squeeze()

                        # plt.imshow(current_target[0].cpu().detach().numpy())
                        # plt.show()

                        #print(current_predict.shape, current_target.shape)
                        #print(current_predict.dtype, current_target.dtype)

                        dice_idx = metric(current_predict, current_target)
                        print(dice_idx.item)

                        #print(dice_idx, dice_idx.shape, dice_idx.dtype )
                        # print(current_predict,current_target)
                        # print(f"\tAcc on batch {i}: {acc}")

                        if phase == 'train':
                            # se face backpropagation -> se calculeaza gradientii
                            loss.backward()
                            # se actualizează weights-urile
                            opt.step()

                    running_loss += loss.item() * ins.size(0)

                    running_average += dice_idx.item() * ins.size(0)

                    #print(running_average, dice_idx.item())
                    #print(running_loss, loss.item())

                    if phase == 'valid':
                        # salvam ponderile modelului dupa fiecare epoca
                        if ep % save_every_ep == 0:
                            torch.save(
                                network, f"{weights_dir}\\my_model{datetime.now().strftime('%m%d%Y_%H%M')}_e{ep}.pt")

                    #     model_path = f"{weights_dir}\\model_epoch{ep}.pth"
                    #     torch.save({'epoch': ep,
                    #                 'model_state_dict': network.state_dict(),
                    #                 'optimizer_state_dict': opt.state_dict(),
                    #                 'loss': total_loss,
                    #                 }, model_path)

                    pbar.update(ins.shape[0])

                # Calculam loss-ul pt toate batch-urile dintr-o epoca
                total_loss[phase].append(
                    running_loss/len(loaders[phase].dataset))

                # Calculam acuratetea pt toate batch-urile dintr-o epoca
                total_dice[phase].append(
                    running_average/len(loaders[phase].dataset))

                postfix = f'error {total_loss[phase][-1]:.4f} dice {dice_idx*100:.2f}%'
                pbar.set_postfix_str(postfix)

                experiment.log_metrics(
                    {f"{phase}_dice": total_dice[phase][-1], "loss": total_loss[-1]}, epoch=ep)

                # Resetam pt a acumula valorile dintr-o noua epoca

    return {'loss': total_loss, 'dice': total_dice}


def main():
    print(f"pyTorch version {torch.__version__}")
    print(f"torchvision version {torchvision.__version__}")
    print(f"torchmetrics version {torchmetrics.__version__}")
    print(f"CUDA available {torch.cuda.is_available()}")
    
    config = None
    with open('config.yaml') as f:  # reads .yml/.yaml files
        config = yaml.safe_load(f)
    
    experiment = Experiment(
        api_key="wwQKu3dl9l1bRZOpeKs0y3r8S",
        project_name="general",
        workspace="razvanavesalon",)

    exp_name = f"Experiment_Dice_index{datetime.now().strftime('%m%d%Y_%H%M')}"

    cwd=pt.Path.cwd()
    print (cwd)
    exp_path = cwd.parent/'experiments'
    exp_path = exp_path/ exp_name
    exp_path.mkdir(exist_ok=True)
    dir = "Weights"
    path = pt.Path(exp_path)/dir
    path.mkdir(exist_ok=True)

    #network = UNet(n_channels=1, n_classes=2,final_activation=nn.Softmax(dim=1))

    network = smp.Unet(
        encoder_name="efficientnet-b1",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # use `imagenet` pre-trained weights for encoder initialization
        encoder_weights="imagenet",
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        in_channels=1,
        # model output channels (number of classes in your dataset)
        classes=2,
    )
    summary(network)
  
  
    

    experiment.log_parameters(config)

    yml_data = yaml.dump(config)
    f = open(f"{path}\\yaml_config.yml", "w+")
    f.write(yml_data)
    f.close()

    pixels = T.Compose([

        TR.ToTensord(keys="img"),
    ])

    pixel_t = TR.Compose([
        TR.GaussianSmoothd(keys="img", sigma=config['train']['sigma']),
        TR.RandGibbsNoised(
            keys="img", prob=config['train']['gibbs_noise_prob'], alpha=config['train']['alpha']),
        TR.RandAdjustContrastd(
            keys="img", prob=config['train']['contrast_prob'], gamma=config['train']['contrast_gamma']),
    ])

    geometric_t = TR.Compose([

        TR.RandRotated(keys=["img", "seg"], prob=config['train']['rotate_prob'],
                       range_x=config['train']['rotate_range'], mode=['bilinear', 'nearest']),
        TR.RandFlipd(keys=["img", "seg"], prob=config['train']['flip_prob'],
                     spatial_axis=config['train']['flip_spatial_axis']),
        TR.RandZoomd(keys=["img", "seg"], prob=config['train']['zoom_prob'],
                     min_zoom=config['train']['min_zoom'], max_zoom=config['train']['max_zoom'])
        #TR.RandSpatialCropSamplesd(keys=["img", "seg"],num_samples=config['train']['rand_crop_samples'], roi_size=config['train']['rand_crop_size'],random_size=False),
    ])

    #path_construct = glob.glob(config["data"]['data_path'])
    #path_list = create_dataset_csv(path_construct)
    #dataset_df = pd.DataFrame(path_list)

    #dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
    # print(dataset_df.head(3))
    # dataset_df.to_csv(config['data']['dataset_csv'])

    dataset_df = pd.read_csv(config['data']['dataset_csv'])

    train_df = dataset_df.loc[dataset_df["subset"] == "train"]
    train_ds = AngioClass(train_df, img_size=config['data']['img_size'],
                          geometrics_transforms=geometric_t, pixel_transforms=pixel_t)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config['train']['bs'], shuffle=True,)
    print(train_loader)

    valid_df = dataset_df.loc[dataset_df["subset"] == "valid", :]
    valid_ds = AngioClass(
        valid_df, img_size=config['data']['img_size'], pixel_transforms=pixels)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=config['train']['bs'], shuffle=False)

    print(f"# Train: {len(train_ds)} # Valid: {len(valid_ds)}")

    criterion = DiceCELoss(to_onehot_y=True, batch=config['train']['bs'])

    if config['train']['opt'] == 'Adam':
        opt = torch.optim.Adam(network.parameters(), lr=config['train']['lr'])
    elif config['train']['opt'] == 'SGD':
        opt = torch.optim.SGD(network.parameters(), lr=config['train']['lr'])
    elif config['train']['opt'] == "RMSprop":
        opt = torch.optim.RMSprop(
            network.parameters(), lr=config['train']['lr'])

    history = train(network, train_loader, valid_loader,experiment,  criterion, opt,
                    epochs=config['train']['epochs'], thresh=config['test']['threshold'], weights_dir=path)
    plot_acc_loss(history, path)


if __name__ == "__main__":
    main()
