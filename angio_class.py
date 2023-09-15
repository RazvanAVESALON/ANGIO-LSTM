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

    def crop_colimator(self, frame,info,clipping_points,frame_nr):
        img = frame.astype(np.float32)
        in_min = 0
        in_max = 2 ** info['BitsStored'] - 1
        out_min = 0
        out_max = 255
        if in_max != out_max:
            img = img.astype(np.float32)
            img = (img - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
            img = np.rint(img)
            img.astype(np.uint8)
            
        # crop collimator
        img_edge = info['ImageEdges']
        print(img.shape)
        img_c = img[..., img_edge[2]:img_edge[3]+1, img_edge[0]:img_edge[1]+1]
        print (img_c.shape)
        bifurcation_point=[]
        if str(frame_nr) in clipping_points:
            bifurcation_point=clipping_points[str(frame_nr)]
            bifurcation_point[1]=bifurcation_point[1]-info['ImageEdges'][0]
            bifurcation_point[0]=bifurcation_point[0]-info['ImageEdges'][2]
            if self.geometrics_transforms != None:
                list_of_keypoints=[]
                list_of_keypoints.append(tuple(bifurcation_point))
                transformed=self.geometrics_transforms (image=img_c,keypoints=list_of_keypoints)
                img_rsz=transformed['image']
                print ('if',img_rsz.shape)
                bifurcation_point=transformed['keypoints'][0]
                print (bifurcation_point)
        else: 
            img_rsz= cv2.resize(img_c,self.img_size, interpolation=cv2.INTER_AREA)
            print ('else',img_rsz.shape)

        return img_rsz,bifurcation_point
    
    def __getitem__(self, idx):
        
        img = np.load(self.dataset_df['images_path'][idx])['arr_0']
        with open(self.dataset_df['annotations_path'][idx]) as f:
            clipping_points = json.load(f)   
        if img.shape[0]>=12:
            new_img=img[:12, :, :] 
        else:
            new_img=np.zeros((12,512,512))
            new_img[:img.shape[0],:,:]=img[:,:,:]
            
        with open(self.dataset_df['angio_loader_header'][idx]) as f:
            angio_loader = json.load(f)
            
        croped_colimator_img= np.zeros(new_img.shape, dtype=np.uint8)
        croped_colimator_gt= np.zeros(new_img.shape, dtype=np.uint8)

        for n in range(new_img.shape[0]):
           
            croped_colimator_img[n],clipping_points[str(n)]= self.crop_colimator(new_img[n],angio_loader,clipping_points,n )                

        print (clipping_points)    
        croped_colimator_gt = croped_colimator_gt/255
        for n in range(new_img.shape[0]):
            clipping_points[str(n)] =list(clipping_points[str(n)])
            clipping_points[str(n)][0]=clipping_points[str(n)][0]/512
            clipping_points[str(n)][1]=clipping_points[str(n)][1]/512
        print (clipping_points)  

        x = np.zeros((12,3,512,512))
        y = clipping_points
        x[:,0,:,:]=croped_colimator_img[:,:,:]
        x[:,1,:,:]=croped_colimator_img[:,:,:]
        x[:,2,:,:]=croped_colimator_img[:,:,:]
        
        
        
        
        
        
        
        tensor_y = torch.from_numpy(y)
        tensor_x = torch.from_numpy(x)
        
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
