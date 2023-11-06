from typing import Any
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch.nn as nn
import torch
import lightning as L
import numpy as np 
import cv2
# from torcheval.metrics import MeanSquaredError
from torchmetrics import MeanSquaredError
import imageio
from skimage.color import gray2rgb
from distances import calculate_distance,mm2pixels,pixels2mm
import pandas as pd 
import pathlib as pt 
import os
import json
class LitAngio(L.LightningModule):
    def __init__(self,network, opt_ch, lr,experiment,overlap_pth=None,gif_path=None,csv_path=None):
        super().__init__()
        self.network = network
        self.opt_ch = opt_ch
        self.lr = lr
        self.experimet=experiment
        self.overlap_pth= overlap_pth
        self.gif_path=gif_path
        self.csv_path=csv_path
        self.dict={"Patient":[], "Acq":[],"Frame":[],"Distance":[]}
    
    def configure_optimizers(self):
        if self.opt_ch == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.opt_ch == "SGD":
            opt = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.opt_ch == "RMSprop":
            opt = torch.optim.RMSprop(self.parameters(), lr=self.lr)

        return opt

    def overlap_3_chanels(self,gt, pred, input):
        # print(input.shape, input.dtype, input.min(), input.max())
        # print(pred.shape, pred.dtype, pred.min(), pred.max())
        print(input.shape)
        pred = pred.astype(np.int32)
        gt = gt.astype(np.int32)

        tp = gt & pred
        fp = ~gt & pred
        fn = gt & ~pred
        tn = ~gt & ~pred

        print(tp.min(), tp.max(), fp.min(), fp.max(), fn.min(), fn.max())

        img = np.zeros((512, 512, 3), np.float32)
        img[:, :, 1] = tp
        img[:, :, 2] = fp
        img[:, :, 0] = fn

        input = gray2rgb(input)
        print(img.min(), img.max(), img.dtype, img.shape)
        print(input.min(), input.max(), input.dtype, input.shape)
        dst = cv2.addWeighted(input, 0.7, img, 0.3, 0)
        # dst = cv2.resize(dst,dsize,interpolation=cv2.INTER_AREA)
        # plt.imshow(img)
        # plt.show()

        return dst
    
    def gif_maker(self,prediction,inputs,targets,index,test_loader,overlap_pred_path,gif_path):
        for step, (input_acq, gt, pred) in enumerate(zip(inputs, targets, prediction)):
                patient, acquisition, header, annotations = test_loader.dataset.csvdata(
                    (index[step].cpu().numpy())
                )
                
                movie_overlap_gif = []
                for frame_index, (frame, bf_gt, bf_pred) in enumerate(
                    zip(input_acq, gt, pred)
                ):
                    
                    frame = frame.cpu().detach().numpy() * 255
                    bf_gt = bf_gt.cpu().detach().numpy() * 512
                    bf_pred = bf_pred.cpu().detach().numpy() * 512
                    with open(header) as f:
                        angio_loader = json.load(f)
                    gt_coords_mm=pixels2mm(bf_gt,angio_loader['MagnificationFactor'],angio_loader['ImageSpacing'])
                    pred_cord_mm=pixels2mm(bf_pred,angio_loader['MagnificationFactor'],angio_loader['ImageSpacing'])
                    print('coord in mm ',pred_cord_mm,gt_coords_mm)
                    if not len(pred_cord_mm) and not len(pred_cord_mm):
                        self.dict["Distance"].append( str("Can't calculate Distance For this frame ( No prediction )" ) )
                        self.dict["Frame"].append(frame_index)
                        self.dict['Patient'].append(patient)
                        self.dict['Acq'].append(acquisition)
                    else:
                        distance=calculate_distance(gt_coords_mm,pred_cord_mm)
                    
                        self.dict["Distance"].append(distance)
                        self.dict["Frame"].append(frame_index)
                        self.dict['Patient'].append(patient)
                        self.dict['Acq'].append(acquisition)

                    
                    black = np.zeros(frame.shape[1:3])
                    masked_gt = cv2.circle(
                        black, (int(bf_gt[1]), int(bf_gt[0])), 5, [255, 255, 255], -1
                    )
                    black2 = np.zeros(frame.shape[1:3])
                    masked_pred = cv2.circle(
                        black2, (int(bf_pred[1]), int(bf_pred[0])), 5, [255, 255, 255], -1
                    )
                    print (frame.shape,masked_pred.shape,masked_gt.shape)
                    overlap_colors = self.overlap_3_chanels(masked_gt, masked_pred, frame[0])

                    frame = str(frame_index)
                    pat_id = str(patient)
                    acn_id = str(acquisition)

                    curr_overlap_path = str(
                        overlap_pred_path / f"OVERLAP_Colored_{pat_id}_{acn_id}-{frame}.png"
                    )
                    cv2.imwrite(curr_overlap_path, overlap_colors)
                    overlap_frame= imageio.imread(curr_overlap_path)
                    foo_Overlap = cv2.putText(
                    overlap_frame, 'Distance:', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
                    foo_Overlap = cv2.putText(foo_Overlap, f'{distance:.2f}', (5, 65), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
                    movie_overlap_gif.append(foo_Overlap)
                    
                imageio.mimsave(os.path.join(gif_path, 'OVERLAP_GIF'+'_' +
                            str(acquisition)+'.gif'), movie_overlap_gif, duration=1000 )    

    
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
        self.experimet.log_metrics({f"Validation_MSE": mse_val,
            f"Validation_loss": loss_val}, epoch=self.current_epoch)
        return loss_val 
    
    def test_step(self, test_batch, batch_idx,):
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
        test_dl = self.trainer.test_dataloaders
        self.gif_maker(output,inputs,targets,idx,test_dl,self.overlap_pth,self.gif_path)
        self.log("test_loss", loss_test)
        self.log("self_loss", mse_test)
        return loss_test
    def on_test_end(self):
        df = pd.DataFrame(self.dict)
        df.to_csv(self.csv_path)

    def predict_step(self, predict_batch, batch_idx):
        inputs,targets , idx = predict_batch
        #self.network.eval()
        y_pred = self.network(inputs.to(device='cuda:0'))

        return y_pred
