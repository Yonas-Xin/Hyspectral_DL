import torch
import torch.nn as nn
import torch.nn.functional as F

class deep_classfier(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(mid_channels, 1024)
        self.fc3 = nn.Linear(1024, out_channels)

    def forward(self, x):
        if x.dim() > 2:
            x = x.squeeze()
        x = F.relu(self.fc1(x),inplace=True)
        # x = self.dropout(x)
        x = F.relu(self.fc2(x),inplace=True) # 输出语义
        return self.fc3(x)