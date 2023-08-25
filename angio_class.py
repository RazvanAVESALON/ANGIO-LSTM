from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import torch


class AngioClass(torch.utils.data.Dataset):
    def __init__(self, dataset_df, img_size, geometrics_transforms=None, pixel_transforms=None):
        self.dataset_df = dataset_df.reset_index(drop=True)
        self.img_size = tuple(img_size)
        self.pixel_transforms = pixel_transforms
        self.geometrics_transforms = geometrics_transforms

    def __len__(self):

        return len(self.dataset_df)

    def csvdata(self, idx):

        patient = self.dataset_df['patient'][idx]
        acquisition = self.dataset_df['acquisition'][idx]
        frame = self.dataset_df['frames'][idx]
        header = self.dataset_df['angio_loader_header'][idx]
        annotations = self.dataset_df['annotations_path'][idx]

        return patient, acquisition, frame, header, annotations

    def crop_colimator(self, frame, gt, info):
        # schimbarea tipului de date
        img = frame.astype(np.float32)
        in_min = 0
        in_max = 2 ** info['BitsStored'] - 1
        out_min = 0
        out_max = 255
        
        if in_max != out_max:
            img = img.astype(np.float32)
            # aplicare normalizare 
            img = (img - in_min) * ((out_max - out_min) /
                                    (in_max - in_min)) + out_min
            img = np.rint(img)
            #schimbarea tipului de date 
            img.astype(np.uint8)

        # marginile imaginii fără colimator collimator
        img_edge = info['ImageEdges']
        
        # eliminarea colimatorului 
        img_c = img[..., img_edge[2]:img_edge[3]+1, img_edge[0]:img_edge[1]+1]
        # aplicare identică pe masca de segmentare
        new_gt = gt[..., img_edge[2]:img_edge[3]+1, img_edge[0]:img_edge[1]+1]

        return img_c, new_gt

    def __getitem__(self, idx):
        img = np.load(self.dataset_df['images_path'][idx])['arr_0']
        frame_param = self.dataset_df['frames'][idx]
        new_img = img[frame_param]
        with open(self.dataset_df['angio_loader_header'][idx]) as f:
            angio_loader = json.load(f)

        #deschiderea adnotării
        with open(self.dataset_df['annotations_path'][idx]) as f:
            clipping_points = json.load(f)                         

        #crearea unei imagini negre
        target = np.zeros(img.shape, dtype=np.uint8)
        #desenarea  punctului de bifurcație 
        target[frame_param] = cv2.circle(target[frame_param], [clipping_points[str(
            frame_param)][1], clipping_points[str(frame_param)][0]], 8, [255, 255, 255], -1)
        
    

        croped_colimator_img, croped_colimator_gt = self.crop_colimator(
            new_img, target[frame_param], angio_loader)

        new_img = cv2.resize(croped_colimator_img,
                             self.img_size, interpolation=cv2.INTER_AREA)
        new_target = cv2.resize(croped_colimator_gt,
                                self.img_size, interpolation=cv2.INTER_AREA)

        new_target = new_target/255
        new_img = new_img*1/255

        x = np.expand_dims(new_img, axis=0)
        y = np.expand_dims(new_target, axis=0)

        tensor_y = torch.from_numpy(y)
        tensor_x = torch.from_numpy(x)

        if self.pixel_transforms != None:

            data_pixel = {"img": tensor_x}
            tensor_x = self.pixel_transforms(data_pixel)["img"]

        if self.geometrics_transforms != None:
            data_geo = {"img": tensor_x, "seg": tensor_y}
            result = self.geometrics_transforms(data_geo)

            tensor_x = result["img"]
            tensor_y = result["seg"]

        #print (tensor_x.min(),tensor_y.min(),tensor_x.max(),tensor_y.max())

        #plt.imshow(tensor_x[0], cmap="gray")
        # plt.show()
        #plt.imshow(tensor_y[0] , cmap="gray")
        # plt.show()

        return tensor_x.float(), tensor_y.int(), idx


def plot_acc_loss(result, path):
    acc = result['dice']['train']
    loss = result['loss']['train']
    val_acc = result['dice']['valid']
    val_loss = result['loss']['valid']

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('DICE', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('DICE')
    plt.xlabel('Epoch')

    plt.subplot(122)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig(f"{path}\\Curbe de învățare")
