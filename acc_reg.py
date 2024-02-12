
import pandas as pd 
import json 

df = pd.read_csv(r"D:\Angio\ANGIO-LSTM\Experimente\Experiment_Dice_index02092024_1325\Test02112024_1722\Statistics.csv")

print(df['Distance'].mean()) 

list_d_2mm=[]
list_d_5mm=[]
list_d_10mm=[] 
list_mean_distance=[]

for d in df['Distance']:
    if d<=2:
            
        list_d_2mm.append(1)
        #list_mean_distance.append(d[0])
    if d<=5:
        list_d_5mm.append(1)
        #list_mean_distance.append(d[0])
                
    if d<=10:
        list_d_10mm.append(1)
    else :
        list_d_2mm.append(0)
        list_d_5mm.append(0)
        list_d_10mm.append(0)
        #list_mean_distance.append(d[0])
    
print ((sum(list_d_2mm)/len(list_d_2mm))*100,(sum(list_d_5mm)/len(list_d_5mm))*100,(sum(list_d_10mm)/len(list_d_10mm))*100)#,(sum(list_mean_distance)/len(list_mean_distance)))