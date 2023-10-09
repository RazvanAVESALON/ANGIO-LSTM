import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(
            input_size=300, hidden_size=256, num_layers=3, batch_first=True
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x_3d):
        hidden = None
        bs = x_3d.shape[0]
        nr_frm = x_3d.shape[1]
        nr_ch = x_3d.shape[2]
        height = x_3d.shape[3]
        width = x_3d.shape[4]
        unpacked = bs * nr_frm
        x_3d = x_3d.reshape(unpacked, nr_ch, height, width)

        with torch.no_grad():
            x = self.resnet(x_3d)

        out, hidden = self.lstm(x.unsqueeze(1), hidden)

        x = self.fc1(out[:, -1, :])
        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(bs, nr_frm, self.num_classes)
        return x


