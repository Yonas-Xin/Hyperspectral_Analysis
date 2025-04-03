import torch.nn as nn
import torch.nn.functional as F
import torch
import math
class common_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1), stride=1):
        super(common_3d,self).__init__()
        '''核大小为（kernel_size，kernel_size，kernel_size）'''
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.batch_norm(self.relu(self.conv(x)))

class SE_SpectralAttention_3d(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d((1, 1, bands))  # 压缩空间维度 (rows, cols) → (1, 1)
        self.fc = nn.Sequential(
            nn.Linear(bands, bands // 8),  # 降维减少计算量
            nn.ReLU(),
            nn.Linear(bands // 8, bands),  # 恢复原始维度
            nn.Sigmoid()  # 输出 [0,1] 的权重
        )
    def forward(self, x):
        # x.shape: (batch, 1, rows, cols, bands)
        batch, _, _, _, bands = x.shape
        gap = self.gap(x)
        gap = gap.view(batch, bands)
        weights = self.fc(gap)
        weights = weights.view(batch, 1, 1, 1, bands)
        return x * weights

class ECA_SpectralAttention_3d(nn.Module):
    def __init__(self, bands,gamma=2,b=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d((bands,1, 1))  # 压缩空间维度 (rows,cols) → (1,1)
        kernel_size = int(abs((math.log(bands, 2) + b) / gamma))
        if kernel_size%2==0:
            kernel_size+=1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (batch, 1, rows, cols, bands)
        batch, _, bands, _, _ = x.shape
        gap = self.gap(x)  # [batch, 1, 1, 1, bands]
        gap = gap.view(batch, 1, bands)  # [batch, 1, bands]
        attn_weights = self.conv(gap)  # 滑动窗口计算局部光谱关系
        # Sigmoid 归一化到 [0,1]
        attn_weights = self.sigmoid(attn_weights)  # [batch, 1, bands]
        # 恢复形状为 (batch,1,1,1,bands)
        attn_weights = attn_weights.view(batch, 1, bands, 1, 1)
        return x * attn_weights
class space_speactral_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,0), stride=1):
        super(space_speactral_3d,self).__init__()
        if in_channels != out_channels:
            self.use_conv1x1 = True
            self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), padding=0, stride=1, bias=False)
            self.batch_norm = nn.BatchNorm3d(out_channels)
        else: self.use_conv1x1 = False
        self.conv1 = common_3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = common_3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.use_conv1x1:
            x = self.batch_norm(self.conv1x1(x))
        return self.relu(x+y)

class SpectralSpatialCNN_3d(nn.Module):
    '''加入光谱注意力机制，不加入空间注意力'''
    def __init__(self,out_embedding=20):
        super().__init__()
        self.spectral_attention = ECA_SpectralAttention_3d(138, 2,1) # 全局平均池化
        self.conv_block1 = common_3d(1,64,(5,1,1),(2,0,0), 1)
        self.conv_block2 = space_speactral_3d(64,128,(3,3,3),(1,1,1),1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv_block3 = space_speactral_3d(128,256,(3,3,3),(1,1,1),1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv_block4 = space_speactral_3d(256,256,(3,3,3),(1,1,1),1)
        self.pool3 = nn.MaxPool3d(2)
        self.linear = nn.Linear(17408, out_features=out_embedding)
        self.dp = nn.Dropout(p=0.25) # 设置一个dropout层
    def forward(self, x):
        x = self.conv_block1(self.spectral_attention(x))
        x = self.pool1(self.conv_block2(x))
        x = self.pool2(self.conv_block3(x))
        x = self.pool3(self.conv_block4(x))
        x = x.view(x.shape[0], -1)
        return self.linear(self.dp(x))

class SpectralSpatialCNN(nn.Module):
    def __init__(self, spectral_channels, spatial_channels, out_features=128):
        """out_features: 输出特征数"""
        super(SpectralSpatialCNN, self).__init__()
        self.spectral_cnn = SpectralCNN(spectral_channels)
        self.space_cnn = SpaceCNN(spatial_channels)

        self.fc1 = nn.Linear(18048, out_features)
    def forward(self, spectral, spatial):
        spectral = self.spectral_cnn(spectral)
        spatial = self.space_cnn(spatial)
        return self.fc1(torch.cat((spatial, spectral), dim=1))

class SpectralCNN(nn.Module):
    def __init__(self, input_channels):
        super(SpectralCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        """输入：(batch_size, channels, width)"""
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = F.relu(self.conv3(x),inplace=True)
        x = F.relu(self.conv4(x),inplace=True)
        x = self.pool(self.conv5(x))
        x = x.view(x.size(0), -1)  # 展平
        return x

class SpaceCNN(nn.Module):
    def __init__(self, input_channels):
        super(SpaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
    def forward(self, x):
        """输入：(batch_size, channels, height, width)"""
        x = F.relu(self.conv1(x),inplace=True)
        x1 = F.relu(self.conv2(x),inplace=True)
        x1 = F.relu(self.conv3(x1),inplace=True)
        x1 = F.relu(self.conv4(x1),inplace=True)
        x1 = self.pool(self.conv5(x1+x))
        x1 = x1.reshape(x1.size(0), -1)  # 展平
        return x1