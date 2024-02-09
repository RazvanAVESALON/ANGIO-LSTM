import pandas as pd 
import os 

csv= pd.read_csv(r"D:\Angio\ANGIO-LSTM\CSV_angiografii_date_adaugate.csv")

csv2= csv.drop_duplicates(subset=['acquisition'])
df={'patient':[],'acquisition':[],'subset':[],'images_path':[],'annotations_path':[],'angio_loader_header':[],'vesselness':[]}
for path,subset in zip(csv2['images_path'],csv2['subset']):
    base=r"D:\date=angiografii\data\data"
    head, tail1 = os.path.split(path)
    head, tail2 = os.path.split(head)
    head , tail3 = os.path.split(head)
    base=os.path.join(base,tail3)
    base=os.path.join(base,tail2)
    df['images_path'].append(os.path.join(base,tail1))
    df['annotations_path'].append(os.path.join(base,'clipping_points.json'))
    df['angio_loader_header'].append(os.path.join(base,'angio_loader_header.json'))
    df['vesselness'].append(os.path.join(base,'vesselness_heatmaps.npz'))
    df['subset'].append(subset)
    df['patient'].append(tail3)
    df['acquisition'].append(tail2)

  
df=pd.DataFrame(df)  

df.to_csv(r"D:\Angio\ANGIO-LSTM\CSV_Heatmap.csv") 
