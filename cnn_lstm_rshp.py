import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2,bs=None):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3,batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.nr_frm=12
        self.bs=bs
       
    def forward (self, x_3d):
        hidden = None
        print (self.bs,type(self.nr_frm))
        unpacked=self.bs*self.nr_frm
        x_3d = x_3d.reshape(unpacked,3,512,512)
        
        print ('x_3d',x_3d.shape)
 
        with torch.no_grad():
            x = self.resnet(x_3d) 
        print(f"resnet out = {x.shape}")
        out, hidden = self.lstm(x.unsqueeze(1), hidden)
        print(f", out = {out.shape}, hidden = {len(hidden)}")       
       
        x = self.fc1(out[:, -1, :])
        x = F.relu(x)
        x = self.fc2(x)
        print ("X",x.shape)
        x= x.reshape (self.bs,12,2)
        return x
    
    # def forward (self, x_3d):
    #     hidden = None
    #     print (x_3d.shape)
    #     # Input: BS X Nr_frames X Nr_chn X H X W - 5 dims (BS + 4 dims)
    #     # reshape: BS * Nr_frames X Nr_chn X H X W - 4 dims (BS + 3 dims)
    #     seq= torch.empty((2,12,2))       
    #     for t in range(x_3d.size(1)):
    #         with torch.no_grad():
    #             x = self.resnet(x_3d[:, t, :, :, :]) 
    #         print(f"resnet out = {x.shape}")
    #         out, hidden = self.lstm(x.unsqueeze(1), hidden)
    #         print(f"t = {t}, out = {out.shape}, hidden = {len(hidden)}")       
       
    #         x = self.fc1(out[:, -1, :])
    #         x = F.relu(x)
    #         x = self.fc2(x)
    #         seq[:,t,:]=x
            
    #     print ("Seq.shape:",seq.shape)
    #     return seq