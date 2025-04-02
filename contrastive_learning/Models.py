import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import SpectralSpatialCNN,SpectralSpatialCNN_3d
from Decoder import simple_classfier,deep_classfier

class F3FN_3d(nn.Module):
    def __init__(self, out_embedding=10):
        super().__init__()
        self.encoder = SpectralSpatialCNN_3d(out_embedding=out_embedding)
        self.decoder = deep_classfier(out_embedding, 128, mid_channels=4096)
    def forward(self, x):
        embedding = self.encoder(x)
        x = self.decoder(F.relu(embedding, inplace=False))
        return embedding, x
    def predict(self, x):
        return self.encoder(x)

class F3FN(nn.Module):
    def __init__(self,spectral_channels, spatial_channels, out_embedding=128):
        super(F3FN,self).__init__()
        self.encoder = SpectralSpatialCNN(spectral_channels, spatial_channels, out_features=out_embedding)
        self.decoder = simple_classfier(out_embedding, 128)
    def forward(self, spectral, space):
        embedding = self.encoder(spectral, space)
        out = self.decoder(F.relu(embedding, inplace=False))
        return embedding, out
    def predict(self, spectral, space):
        return self.encoder(spectral, space)