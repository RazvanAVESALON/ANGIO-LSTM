import pandas as pd 
import imageio
import cv2
import yaml
from datetime import datetime
import pathlib as pt 
from skimage.color import gray2rgb
import numpy as np 
import os 
import json 
from angio_class import AngioClass
import torch
import albumentations as A 
import matplotlib.pyplot as plt 
def overlap_3_chanels(pred, input):
    # print(input.shape, input.dtype, input.min(), input.max())
    # print(pred.shape, pred.dtype, pred.min(), pred.max())
    print(input.shape)

 
    input=np.float32(input)
    pred=np.float32(pred)
    input = gray2rgb(input)

    print(input.min(), input.max(), input.dtype, input.shape)
    dst = cv2.addWeighted(input, 0.7, pred, 0.3, 0)
    # dst = cv2.resize(dst,dsize,interpolation=cv2.INTER_AREA)
    # plt.imshow(img)
    # plt.show()

    return dst

def main():

    csv_seq= pd.read_csv(r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index02092024_1325\Test02112024_1722\Statistics.csv")
    csv_rshp= pd.read_csv(r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index02062024_1810\Test02112024_1715\Statistics.csv")
    config = None
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    directory = f"Gif_overlap_Combined{datetime.now().strftime('%m%d%Y_%H%M')}"
    dir2="pred"
    parent_dir_seq = r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index02092024_1325\Test02112024_1722"
    parent_dir_rshp=   r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index02062024_1810\Test02112024_1715"
    test_path_seq = pt.Path(parent_dir_seq) / directory
    test_path_seq.mkdir(exist_ok=True)
    test_path_rshp = pt.Path(parent_dir_rshp) / directory
    test_path_rshp.mkdir(exist_ok=True)
    path_pred_seq = pt.Path(parent_dir_seq) / dir2
    path_pred_seq.mkdir(exist_ok=True)
    path_pred = pt.Path(parent_dir_rshp) / dir2
    path_pred.mkdir(exist_ok=True)
    patient=csv_seq['Patient'].unique()
    acq=csv_seq['Acq'].unique()

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
        test_df, img_size=config["data"]["img_size"],nr_frm=config['data']['nr_frames'], geometrics_transforms=geometric_t
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["train"]["bs"], shuffle=False
    )

    movie_overlap_gif=[]
    
    for batch_index, batch in enumerate(test_loader):
        x, y, index = iter(batch)

        for step, (input, gt) in enumerate(zip(x, y)):
            patient, acquisition, header, annotations,img_pth = test_loader.dataset.csvdata(
                    (index[step].cpu().numpy())
                )

            for i in range(len(csv_seq['Acq'])-1):
                print (acquisition == csv_seq['Acq'][i])
                print ("Acq:",type(acquisition),type(csv_seq['Acq'][i]))

                if patient == csv_seq['Patient'][i] and acquisition==str(csv_seq['Acq'][i]):
                    frame = str(csv_seq['Frame'][i])

                    img = input.cpu().numpy()*255
                    print (img.shape)
                    img= img[int(frame)]
                    pat_id = str(csv_seq['Patient'][i])
                    acn_id = str(csv_seq['Acq'][i])
                    bf_gt=csv_seq['Bifurcation_point'][i].strip('][ ').split(' ')

                    for f in range(len(bf_gt)):
                        if '' in bf_gt : 
                            bf_gt.remove('')
                    
                    bf_gt= [float(i) for i in bf_gt]
                    bf_pred=csv_seq['Prediction'][i].strip('] [ ').split(' ')

                    for f in range(len(bf_pred)):
                        if '' in bf_pred : 
                            bf_pred.remove('')
                    
                    bf_pred= [float(i) for i in bf_pred]
                    bf_pred2=csv_rshp['Prediction'][i].strip('][').split(' ')
                    for f in range(len(bf_pred2)):
                        if '' in bf_pred2 : 
                            bf_pred2.remove('')
                    bf_pred2= [float(i) for i in bf_pred2]
                    distance_seq=csv_seq['Distance'][i]
                    distance_rshp=csv_rshp['Distance'][i]
                
                    if csv_seq['Acq'][i]==csv_seq['Acq'][i+1]:
                        
                        black = np.zeros((512,512,3))

                        masked_gt = cv2.circle(black, (int(bf_gt[1]), int(bf_gt[0])), 5, [255, 0, 0], -1)
                        masked_pred_seq = cv2.circle(masked_gt, (int(bf_pred[1]), int(bf_pred[0])), 5, [0, 0, 255], -1)
                        masked_pred_rshp = cv2.circle(masked_pred_seq, (int(bf_pred2[1]), int(bf_pred2[0])), 5, [0, 128, 255], -1)
                        overlap_colors = overlap_3_chanels(masked_pred_rshp, img[0])
                        

                        curr_overlap_path = str(path_pred/ f"OVERLAP_Colored_{pat_id}_{acn_id}.png")
                        cv2.imwrite(curr_overlap_path, overlap_colors)
                        overlap_frame= imageio.imread(curr_overlap_path)

                        foo_Overlap = cv2.putText(
                        overlap_frame, 'DistanceSeq:', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0))
                        foo_Overlap = cv2.putText(foo_Overlap, f'{distance_seq:.2f}', (5, 58), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0))
                        foo_Overlap = cv2.putText(
                        overlap_frame, 'DistanceRshp:', (5, 75), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 128, 0))
                        foo_Overlap = cv2.putText(foo_Overlap, f'{distance_rshp:.2f}', (5, 88), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 128, 0))
                        movie_overlap_gif.append(foo_Overlap)
                                    
                        #imageio.mimsave(os.path.join(test_path_seq, 'OVERLAP_GIF'+'_' +str(acn_id)+'.gif'), movie_overlap_gif, duration=1000 ) 
                        
                        #imageio.mimsave(os.path.join(test_path_rshp, 'OVERLAP_GIF'+'_' +str(acn_id)+'.gif'), movie_overlap_gif, duration=1000 )
                    else: 
                        black = np.zeros((512,512,3))
                        masked_gt = cv2.circle(black, (int(bf_gt[1]), int(bf_gt[0])), 5, [255, 0, 0], -1)
                        masked_pred_seq = cv2.circle(masked_gt, (int(bf_pred[1]), int(bf_pred[0])), 5, [0, 0, 255], -1)
                        masked_pred_rshp = cv2.circle(masked_pred_seq, (int(bf_pred2[1]), int(bf_pred2[0])), 5, [0, 128, 255], -1)
                        overlap_colors = overlap_3_chanels(masked_pred_rshp, img[0])

                        
                        curr_overlap_path = str(path_pred/ f"OVERLAP_Colored_{pat_id}_{acn_id}.png")
                        cv2.imwrite(curr_overlap_path, overlap_colors)
                        overlap_frame= imageio.imread(curr_overlap_path)
                        foo_Overlap = cv2.putText(
                        overlap_frame, 'DistanceSeq:', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0))
                        foo_Overlap = cv2.putText(foo_Overlap, f'{distance_seq:.2f}', (5, 58), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0))
                        foo_Overlap = cv2.putText(
                        overlap_frame, 'DistanceRshp:', (5, 75), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 128, 0))
                        foo_Overlap = cv2.putText(foo_Overlap, f'{distance_rshp:.2f}', (5, 88), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 128, 0))
                        movie_overlap_gif.append(foo_Overlap)
                                    
                        imageio.mimsave(os.path.join(test_path_seq, 'OVERLAP_GIF'+'_' +str(acn_id)+'.gif'), movie_overlap_gif, duration=1000 ) 
                        
                        imageio.mimsave(os.path.join(test_path_rshp, 'OVERLAP_GIF'+'_' +str(acn_id)+'.gif'), movie_overlap_gif, duration=1000 )
                        movie_overlap_gif=[]
                    
                
    
    

        
if __name__ == "__main__":
    main()        