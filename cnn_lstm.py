import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


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
        seq = torch.empty((bs, nr_frm, self.num_classes))
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(1), hidden)
            x = self.fc1(out[:, -1, :])
            x = F.relu(x)
            x = self.fc2(x)
            seq[:, t, :] = x

        return seq


# if __name__ == "__main__":
#     net = CNNLSTM()
#     net.to("cuda")
#     net.train()

#     bs = 16
#     nr_frames = 12
#     height, width = 512, 512
#     ins = torch.zeros((bs, nr_frames, 3, height, width), device="cuda")

#     for i in range(1, 3):
#         out = net(ins)
#         print(f"out = {out.shape}")
