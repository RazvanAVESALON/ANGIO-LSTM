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

        img = frame.astype(np.float32)
        in_min = 0
        in_max = 2 ** info['BitsStored'] - 1
        out_min = 0
        out_max = 255
        
        if in_max != out_max:
            img = img.astype(np.float32) 
            img = (img - in_min) * ((out_max - out_min) /
                                    (in_max - in_min)) + out_min
            img = np.rint(img)
            img.astype(np.uint8)

        img_edge = info['ImageEdges']
        
        img_c = img[..., img_edge[2]:img_edge[3]+1, img_edge[0]:img_edge[1]+1]
        new_gt = gt[..., img_edge[2]:img_edge[3]+1, img_edge[0]:img_edge[1]+1]
        
        img_c= cv2.resize(img_c,self.img_size, interpolation=cv2.INTER_AREA)
        new_gt= cv2.resize(new_gt,self.img_size, interpolation=cv2.INTER_AREA)
        return img_c, new_gt

    def __getitem__(self, idx):
        img = np.load(self.dataset_df['images_path'][idx])['arr_0']
        print (img.shape)
        with open(self.dataset_df['annotations_path'][idx]) as f:
            clipping_points = json.load(f)   
        print (self.dataset_df['images_path'][idx])

        ann= np.zeros(img.shape)   
        print (ann)
        for n in range(img.shape[0]):
            print(n)
            if clipping_points.get(str(n)):
                ann[n]=cv2.circle(ann[n],[clipping_points[str(n)][1], clipping_points[str(n)][0]], 8, [255, 255, 255], -1)
          
        if img.shape[0]>=12:
            new_img=img[:12, :, :] 
            target=ann[:12,:,: ]
        else:
            new_img=np.zeros((12,512,512))
            target = np.zeros((12,512,512), dtype=np.uint8)
            new_img[:img.shape[0],:,:]=img[:,:,:]
            target[:ann.shape[0],:,:]=ann[:,:,:]
            print (new_img.shape,target.shape)
        with open(self.dataset_df['angio_loader_header'][idx]) as f:
            angio_loader = json.load(f)
            
        croped_colimator_img= np.zeros(new_img.shape, dtype=np.uint8)
        croped_colimator_gt= np.zeros(new_img.shape, dtype=np.uint8)
      
        for n in range(new_img.shape[0]):
            croped_colimator_img[n], croped_colimator_gt[n] = self.crop_colimator(new_img[n], target[n], angio_loader)
        croped_colimator_gt = croped_colimator_gt/255
        croped_colimator_img= croped_colimator_img/255

        x = np.zeros((12,3,512,512))
        y = np.zeros((12,3,512,512))
        x[:,0,:,:]=croped_colimator_img[:,:,:]
        y[:,0,:,:]=croped_colimator_gt[:,:,:]
        x[:,1,:,:]=croped_colimator_img[:,:,:]
        y[:,1,:,:]=croped_colimator_gt[:,:,:]
        x[:,2,:,:]=croped_colimator_img[:,:,:]
        y[:,2,:,:]=croped_colimator_gt[:,:,:]
        print (x.shape,y.shape)
        tensor_y = torch.from_numpy(y)
        tensor_x = torch.from_numpy(x)

        # if self.pixel_transforms != None:

        #     data_pixel = {"img": tensor_x}
        #     tensor_x = self.pixel_transforms(data_pixel)["img"]

        # if self.geometrics_transforms != None:
        #     data_geo = {"img": tensor_x, "seg": tensor_y}
        #     result = self.geometrics_transforms(data_geo)

        #     tensor_x = result["img"]
        #     tensor_y = result["seg"]
            
        return tensor_x.float(), tensor_y.float(), idx


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
