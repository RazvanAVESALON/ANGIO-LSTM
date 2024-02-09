# import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import torch
import yaml
import matplotlib.pyplot as plt



class AngioClass(torch.utils.data.Dataset):
    def __init__(
        self, dataset_df, img_size,nr_frm, geometrics_transforms=None, pixel_transforms=None
    ):
        self.dataset_df = dataset_df.reset_index(drop=True)
        self.img_size = tuple(img_size)
        self.pixel_transforms = pixel_transforms
        self.geometrics_transforms = geometrics_transforms
        self.number_of_ch = 3
        self.nr_coordonates = 2
        self.nr_frm=nr_frm

    def __len__(self):
        return len(self.dataset_df)

    def csvdata(self, idx):
        patient = self.dataset_df["patient"][idx]
        acquisition = self.dataset_df["acquisition"][idx]
        header = self.dataset_df["angio_loader_header"][idx]
        annotations = self.dataset_df["annotations_path"][idx]
        img_pth=self.dataset_df["images_path"][idx]
        return patient, acquisition, header, annotations,img_pth

    def crop_colimator(self, frame, info, clipping_points, frame_nr):
        img = frame.astype(np.float32)

        in_min = 0
        in_max = 2 ** info["BitsStored"] - 1
        out_min = 0
        out_max = 255
        if in_max != out_max:
            img = img.astype(np.float32)
            img = (img - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
            img = np.rint(img)
            img.astype(np.uint8)

        img_edge = info["ImageEdges"]
        img_c = img[..., img_edge[2] : img_edge[3] + 1, img_edge[0] : img_edge[1] + 1]

        bifurcation_point = []
        if str(frame_nr) in clipping_points:
            if clipping_points[str(frame_nr)] :
                bifurcation_point = clipping_points[str(frame_nr)]
                bifurcation_point[1] = bifurcation_point[1] - info["ImageEdges"][0]
                bifurcation_point[0] = bifurcation_point[0] - info["ImageEdges"][2]
            else:
                bifurcation_point [1]=0
                bifurcation_point[0]=0
            
            if self.geometrics_transforms != None:
                    list_of_keypoints = []
                    list_of_keypoints.append(tuple(bifurcation_point))
                    transformed = self.geometrics_transforms(
                        image=img_c, keypoints=list_of_keypoints
                    )

                    img_rsz = transformed["image"]

                    if transformed["keypoints"]:
                      bifurcation_point = transformed["keypoints"][0]
                    else: 
                        bifurcation_point[1] = 0
                        bifurcation_point[0] = 0
                    
            img_rsz=img_rsz.astype(np.uint8)
        
            if self.pixel_transforms != None:

                list_of_keypoints=[]
                list_of_keypoints.append(tuple(bifurcation_point))
                transformed=self.pixel_transforms (image=img_rsz,keypoints=list_of_keypoints)
                img_rsz=transformed['image']
                bifurcation_point=transformed['keypoints'][0]
                
        else:
            img_rsz = cv2.resize(img_c, self.img_size, interpolation=cv2.INTER_AREA)

        return img_rsz, bifurcation_point

    def __getitem__(self, idx):
        img = np.load(self.dataset_df["images_path"][idx])["arr_0"]
        heatmap=np.load(self.dataset_df["vesselness"][idx])["arr_0"]

        with open(self.dataset_df["annotations_path"][idx]) as f:
            clipping_points = json.load(f)
        if img.shape[0] >= 12:
            new_img = img[: 12, :, :]
            new_heatmap=heatmap[: 12, :, :]
        else:
            new_img = np.zeros(
                (
                    12,
                    self.img_size[0],
                    self.img_size[1],
                )
            )
            new_heatmap = np.zeros(
                (
                    12,
                    self.img_size[0],
                    self.img_size[1],
                )
            )
            new_heatmap[: img.shape[0], :, :] = img[:, :, :]
            new_img[: img.shape[0], :, :] = img[:, :, :]

        with open(self.dataset_df["angio_loader_header"][idx]) as f:
            angio_loader = json.load(f)

        croped_colimator_img = np.zeros(new_img.shape, dtype=np.uint8)
        croped_colimator_heat = np.zeros(new_heatmap.shape, dtype=np.uint8)
        
        for n in range(new_img.shape[0]):
            croped_colimator_img[n], clipping_points[str(n)] = self.crop_colimator(
                new_img[n], angio_loader, clipping_points, n
            )
            croped_colimator_heat[n], clipping_points[str(n)] = self.crop_colimator(
                new_heatmap[n], angio_loader, clipping_points, n
            )
            

        # plt.imshow(croped_colimator_img[0])
        # plt.show()
        croped_colimator_img = croped_colimator_img / 255
        croped_colimator_heat=croped_colimator_heat / 255
        for n in range(new_img.shape[0]):
            clipping_points[str(n)] = list(clipping_points[str(n)])

        x = np.zeros(
            (
                self.nr_frm,
                self.number_of_ch,
                self.img_size[0],
                self.img_size[1],
            )
        )
        
        data = np.empty((self.nr_frm, self.nr_coordonates))
     
        for n in range(new_img.shape[0]):
            if clipping_points[str(n)] :
                input_memory=croped_colimator_img[n]
                data_memory=clipping_points[str(n)]
                break
        ok = 0 
        for n in range(new_img.shape[0]):
            
            if clipping_points[str(n)] :
                data[n] = clipping_points[str(n)]
                ok = 1
                save_frm_number= n   
            else: 
                if ok == 1 :
                    data[n]= clipping_points[str(save_frm_number)]
                    croped_colimator_img[n] = croped_colimator_img[save_frm_number]
                if ok == 0 :
                        data[n]= data_memory
                        croped_colimator_img[n]=input_memory
            
          
                    
           


        y = data / 512
      
        
        x[:, 0, :, :] = croped_colimator_img[:, :, :]
        x[:, 1, :, :] = croped_colimator_heat[:, :, :]
        x[:, 2, :, :] = croped_colimator_img[:, :, :]
        tensor_y = torch.from_numpy(y)
        tensor_x = torch.from_numpy(x)

        return tensor_x.float(), tensor_y.float(), idx
