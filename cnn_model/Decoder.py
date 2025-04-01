import torch
import torch.nn as nn
import torch.nn.functional as F

class simple_classfier(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=128):
        super(simple_classfier,self).__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x),inplace=True)
        x = self.fc2(x) # 输出语义
        return x

class deep_classfier(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=4096):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 128)
        self.fc2 = nn.Linear(128, mid_channels)
        self.fc3 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x),inplace=True)
        x = F.relu(self.fc2(x),inplace=True) # 输出语义
        return self.fc3(x)