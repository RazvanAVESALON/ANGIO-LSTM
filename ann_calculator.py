import glob 
import os
import json
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
path = r"D:\date=angiografii\data\data\*"

path= glob.glob(path)
dictionary= {"Patient":[], 'Acq':[], "Frame":[],'Ann':[],'Percent':[] ,'Ann_start':[],'Ann_mid':[],'Ann_end':[]} 
pc= {'pc_per_acq':[]}
for patient in path :
    x = glob.glob(os.path.join(patient, r"*"))
    for acq in x:
        annotations = os.path.join(acq, "clipping_points.json")
        img = np.load(os.path.join(acq, "frame_extractor_frames.npz"))['arr_0']
        
        with open(annotations) as f:
            clipping_points = json.load(f)

        pt_tail= os.path.split(patient)
   
        ac_tail= os.path.split(acq)
        percent_per_acq=(len(clipping_points.keys())/ img.shape[0])*100
        pc['pc_per_acq'].append(percent_per_acq)
        print(pc)
        start ,mid, end= 0 , 0 ,0  
        sum = 0
        for n in range(img.shape[0]):
           if n == 0 and str(n) in clipping_points:
               start= 1
           elif str(n) not in clipping_points and start == 0: start = 1 
           elif str(n) in clipping_points and start == 1 : start =1 
           elif str(n) not in clipping_points and start==1 and  mid==0 : mid = 1
           elif str(n) in clipping_points and start == 1 and mid ==1: start ,mid = 1, 1  
           elif str(n) not in clipping_points and start == 1 and mid == 1: end = 1
           
        for n in range(img.shape[0]):
      
           percent = (len(clipping_points.keys())/ img.shape[0])*100

           if clipping_points:
               if str(n) in clipping_points:
                    dictionary['Patient'].append(pt_tail[1]) 
                    dictionary['Acq'].append(ac_tail[1])
                    dictionary['Frame'].append(n)
                    dictionary['Ann'].append(1)
                    dictionary['Percent'].append(percent)
                    dictionary['Ann_start'].append(start)
                    dictionary['Ann_mid'].append(mid)
                    dictionary['Ann_end'].append(end)
                    
               else:
                    dictionary['Patient'].append(pt_tail[1]) 
                    dictionary['Acq'].append(ac_tail[1])
                    dictionary['Frame'].append(n)
                    dictionary['Ann'].append(0)
                    dictionary['Percent'].append(percent)
                    dictionary['Ann_start'].append(start)
                    dictionary['Ann_mid'].append(mid)
                    dictionary['Ann_end'].append(end)

                
           else:   

                
                dictionary['Patient'].append(pt_tail[1]) 
                dictionary['Acq'].append(ac_tail[1])
                dictionary['Frame'].append(n)
                dictionary['Ann'].append(0)
                dictionary['Percent'].append(percent)
                dictionary['Ann_start'].append(start)
                dictionary['Ann_mid'].append(mid)
                dictionary['Ann_end'].append(end)

                    

      
    

df=pd.DataFrame(dictionary)

plt.hist(pc['pc_per_acq'])
plt.savefig(r'D:\Angio\ANGIO-LSTM\PERCENT_HIST.jpeg')
df.to_csv(r'D:\Angio\ANGIO-LSTM\Ann.csv')
       