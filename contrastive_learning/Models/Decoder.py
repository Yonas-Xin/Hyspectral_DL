import torch.nn as nn
import torch.nn.functional as F

class Contrastive_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, mid_channels)
        self.fc3 = nn.Linear(mid_channels, out_channels, bias=False)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) # 输出语义
        return self.fc3(x)