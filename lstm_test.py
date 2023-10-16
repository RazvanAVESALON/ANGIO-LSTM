
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
from lighting_train import LitAngio
import lightning as L


# from cnn_lstm_rshp import CNNLSTM
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

    network = CNNLSTM(num_classes=2)
    summary(network)
    experiment.log_parameters(config)

    yml_data = yaml.dump(config)
    f = open(f"{path}\\yaml_config.yml", "w+")
    f.write(yml_data)
    f.close()

    pixels = T.Compose(
        [
            TR.ToTensord(keys="img"),
        ]
    )
    geometric_t = A.Compose(
        [
            A.Resize(
                height=config["data"]["img_size"][0],
                width=config["data"]["img_size"][1],
            )
        ],
        keypoint_params=A.KeypointParams(format="yx", remove_invisible=False),
    )

    dataset_df = pd.read_csv(config["data"]["dataset_csv"])
    test_df = dataset_df.loc[dataset_df["subset"] == "test", :]
    test_ds = AngioClass(test_df, img_size=config["data"]["img_size"],geometrics_transforms=geometric_t)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["train"]["bs"], shuffle=False)
    
    model =L.load_from_checkpoint(
    checkpoint_path=r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index10162023_1454\Weights\lightning_logs\version_0\checkpoints\epoch=0-step=95.ckpt",
    weights_path=r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index10162023_1454\Weights\lightning_logs\version_0\checkpoints\epoch=0-step=95.ckpt",
    tags_csv=r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index10162023_1454\Weights\lightning_logs\version_0\metrics.csv",
    hparams_file=r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index10162023_1454\Weights\lightning_logs\version_0\hparams.yaml",
    on_gpu=True,
    map_location=None)

    # init trainer with whatever options
    trainer = L.Trainer()

    # test (pass in the model)
    trainer.test(model)
if __name__ == "__main__":
    main()


