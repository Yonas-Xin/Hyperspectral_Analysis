from Encoder import SpectralSpatialCNN_3d
from Decoder import deep_classfier
import torch.nn as nn
import torch.nn.functional as F

class CNN_3d(nn.Module):
    '''编码器部分和F3FN_3一样，解码器部分需要重新训练'''
    def __init__(self, out_embedding=10, out_classes=8):
        super().__init__()
        self.encoder = SpectralSpatialCNN_3d(out_embedding=out_embedding)
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=4096)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(F.relu(x, inplace=True))
        return x