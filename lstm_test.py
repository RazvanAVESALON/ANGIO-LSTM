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
import cv2
from skimage.color import gray2rgb
import numpy as np
import imageio
import os


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

    directory = f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
    parent_dir = config["data"]["parent_dir_exp"]
    test_path = pt.Path(parent_dir) / directory
    test_path.mkdir(exist_ok=True)

    csv_path = pt.Path(test_path) / r"Statistics.csv"
    gif_path = pt.Path(test_path) / r"Gif_prediction_overlap"
    gif_path.mkdir(exist_ok=True)
    overlap_pred_path = pt.Path(test_path) / r"Predictii_Overlap"
    overlap_pred_path.mkdir(exist_ok=True)

    network = CNNLSTM(num_classes=2)
    summary(network)
    experiment.log_parameters(config)

    yml_data = yaml.dump(config)
    f = open(f"{test_path}\\yaml_config.yml", "w+")
    f.write(yml_data)
    f.close()

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
    test_ds = AngioClass(
        test_df, img_size=config["data"]["img_size"], geometrics_transforms=geometric_t
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["train"]["bs"], shuffle=False
    )

    # init trainer with whatever options
    trainer = L.Trainer(max_epochs=1, accelerator="gpu")
    # test (pass in the model)
    trainer.test(
        LitAngio(
            network,
            config["train"]["opt"],
            config["train"]["lr"],
            experiment,
            overlap_pred_path,
            gif_path,
            csv_path,
        ),
        dataloaders=test_loader,
        ckpt_path=config["data"]["test_model_checkpoint"],
    )
    # prediction = trainer.predict(
    #     LitAngio(network, config["train"]["opt"], config["train"]["lr"], experiment),
    #     dataloaders=test_loader,
    #     ckpt_path=config["data"]["test_model_checkpoint"],
    # )


if __name__ == "__main__":
    main()
